"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round
from diffuseq.text_datasets import load_data_text
from diffuseq.utils.nn import mean_with_mask
from basic_utils import straightness

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0, td=0.0, rescale_max=1.0, cand_num=1)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    print(args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(dist_util.dev())

    tokenizer = load_tokenizer(args, is_eval=True)
    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size, 
        embedding_dim=args.hidden_dim, 
        _weight=model.word_embedding.weight.clone().cpu()
    ).eval().requires_grad_(False)


    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"{args.td}-ema{model_base_name.split('.ema')[1]}.samples-{time.strftime('%Y%m%d-%H:%M:%S')}")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    for cand in range(args.cand_num):
        if cand != 0:
            args.seed2 += 100

        set_seed(args.seed2)

        print("### Sampling...on", args.split)

        ## load data
        data_valid = load_data_text(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            deterministic=True,
            data_args=args,
            split=args.split,
            loaded_vocab=tokenizer,
            model_emb=model_emb.cpu(),  # using the same embedding wight with tranining data
            loop=False
        )


        # batch, cond = next(data_valid)
        # print(batch.shape)

        
        out_file = os.path.join(out_path, f"seed{args.seed2}_step{args.step}.json")
        # fout = open(out_path, 'a')

        all_test_data = []

        idx = 0

        try:
            while True:
                batch, cond = next(data_valid)
                # print(batch.shape)
                if idx % world_size == rank:  # Split data per nodes
                    all_test_data.append(cond)
                idx += 1

        except StopIteration:
            print('### End of reading iteration...')

        model_emb.to(dist_util.dev())

        if idx % world_size and rank >= idx % world_size:
            all_test_data.append({})  # Dummy data for Remainder : for dist.barrier()

        if rank == 0:
            from tqdm import tqdm
            iterator = tqdm(all_test_data)
        else:
            iterator = iter(all_test_data)

        for cond in iterator:

            if not cond:  # Barrier for Remainder
                for i in range(world_size):
                    dist.barrier()
                continue

            input_ids_x = cond.pop('input_ids').to(dist_util.dev())
            x_start = model.get_embeds(input_ids_x)
            input_ids_mask = cond.pop('input_mask').to(dist_util.dev())
            input_ids_mask_ori = input_ids_mask

            mse_mask = diffusion.get_mse_mask(input_ids_x, input_ids_mask, x_start, mask_type="remain y0, 1st pad, 2nd pad")

            noise = th.randn_like(x_start)
            input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
            x_noised = th.where(input_ids_mask == 0, x_start, noise)

            def probe(hidden_repr):
                decode_logits = th.softmax(model.get_logits(hidden_repr), dim=-1)
                decode_tokens = decode_logits.argmax(dim=-1)
                # cands = th.topk(decode_logits, k=5, dim=-1)
                # print(decode_logits.shape, cands.indices.shape)
                # for i in range(33,34):
                #     print(decode_logits[i, cands.indices[i]],"\n",cands.indices[i], flush=True)
                for decode_token in decode_tokens:
                    print(tokenizer.decode_token(decode_token))
            # probe(x_start)
            # exit(0)

            def rounding_embedding(hidden_repr):
                decode_logits = th.softmax(model.get_logits(hidden_repr), dim=-1)
                decode_tokens = decode_logits.argmax(dim=-1)
                return model.get_embeds(decode_tokens)

            def cal_input_t_1(t):
                t = min(1, t+args.td)
                return t


            def cal_input_t_2(t):
                if t > 1 - args.td:
                    return 1
                else:
                    return t / (1 - args.td)
                
            
            def cal_input_t_3(t):
                k = 1 - args.td
                return k * t + args.td

            def cal_input_t_4(t):
                if t < args.td:
                    return args.td
                return t.clone()

            

            model_kwargs = {}

            samples = [x_noised]

            x_start_hats = [th.where(input_ids_mask==0, x_start, 0).to(x_start.device)]
            
            start_t = time.time()

            with th.no_grad():
                ts = th.linspace(1, 0, args.step+1)
                for i, t in tqdm(enumerate(ts[:-1])):
                    dt = t - ts[i+1]
                    
                    input_t = cal_input_t_1(t)

                    # input_t = cal_input_t(t, args.td)
                    # input_t = t.clone()

                    input_t *= args.rescale_max
                    x_t = samples[-1]
                    # x_t = th.where(input_ids_mask==0, th.randn_like(x_start), x_start) # seq to be translated is noise
                    # x_t = th.where(input_ids_mask==0, x_start, x_start) # translated seq is noise
                    # if t == 1:
                    #     x_t = th.where(input_ids_mask.flip(dims=[0])==0, x_start.flip(dims=[0]), th.randn_like(x_start))
                    # elif t < 1:
                    #     x_t = th.where(input_ids_mask.flip(dims=[0])==0, x_start.flip(dims=[0]), x_t)
                    # probe(x_t)
                    if model.input_dims != model.output_dims:
                        x_start_hat = model(th.cat((x_t, x_start_hats[-1]), dim=-1), input_t*th.ones(x_t.size(0)).to(x_t.device), **cond)
                        # x_start_hat = model(th.cat((x_t, th.zeros_like(x_t)), dim=-1), input_t*th.ones(x_t.size(0)).to(x_t.device), **cond)
                    else:
                        x_start_hat = model(x_t, input_t*th.ones(x_t.size(0)).to(x_t.device), **cond)
                    x_start_hats.append(th.where(input_ids_mask==0, x_start, x_start_hat))
                    # from matplotlib import pyplot as plt
                    # fig, ax = plt.subplots()
                    # ax.plot(((x_t - x_start_hat)/t)[0].pow(2).mean(dim=-1).detach().cpu().numpy(), color="green")
                    # ax.plot(input_ids_mask_ori[0].detach().cpu().numpy(), color="blue")
                    # ax.plot((x_noised-x_start)[0].pow(2).mean(dim=-1).detach().cpu().numpy(), color="red")
                    # ax.set_ylim(0,2.5)
                    # fig.savefig(f"/home/pliu/workarea/ReflowSeq/tmpres/{t}.png")

                    # samples.append(x_start_hat)
                    probe(x_start_hat[:1])
                    time.sleep(1)
                    print(t, input_t)
                    print(mean_with_mask((x_start_hat-x_start)**2, mse_mask).mean())
                    # print(diffusion._token_discrete_loss(x_start_hat, model.get_logits, input_ids_x, mse_mask[:,:,0]).mean())
                    v_hat = (x_t - x_start_hat)/t
                    x_prev = x_t - dt*v_hat
                    # x_prev = th.randn_like(x_start_hat)*t+(1-t)*x_start_hat

                    x_prev = th.where(input_ids_mask == 0, x_start, x_prev)
                    # print(mean_with_mask((x_prev-x_start)**2, mse_mask).mean())

                    samples.append(x_prev)
            running_time = time.time() - start_t
            print(f"steps: {len(x_start_hats)}, curvature: ", straightness(samples))

            
            # if args.step == args.diffusion_steps:
            #     args.use_ddim = False
            #     step_gap = 1
            # else:
            #     args.use_ddim = True
            #     step_gap = args.diffusion_steps//args.step

            # sample_fn = (
            #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            # )

            # sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

            # samples = sample_fn(
            #     model,
            #     sample_shape,
            #     noise=x_noised,
            #     clip_denoised=args.clip_denoised,
            #     denoised_fn=partial(denoised_fn_round, args, model_emb),
            #     model_kwargs=model_kwargs,
            #     top_p=args.top_p,
            #     clamp_step=args.clamp_step,
            #     clamp_first=True,
            #     mask=input_ids_mask,
            #     x_start=x_start,
            #     gap=step_gap
            # )

            # print(samples[0].shape) # samples for each step
                
            def logAllSamples(**kargs):
                for key, arg in kargs.items():
                    for item in arg:
                        logits = model.get_logits(item)  # bsz, seqlen, vocab
                        cands = th.topk(logits, k=1, dim=-1)
                        word_lst_recover = []
                        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
                            len_x = args.seq_len - sum(input_mask).tolist()
                            length, tokens = tokenizer.decode_token_stop_at_sep(seq[len_x:], True)
                            word_lst_recover.append(f"{length}:{tokens}")
                        with open(out_file.replace(".json", f"-log-{key}.jsonl"), "a") as f:
                            f.write(str(word_lst_recover)+"\n")
            # logAllSamples(**{"xt":samples[1:], "x0":x_start_hats[1:]})

            sample = samples[-1]

            # print('decoding for seq2seq', )
            # print(sample.shape)

            logits = model.get_logits(sample)  # bsz, seqlen, vocab
            cands = th.topk(logits, k=1, dim=-1)

            # for x_start_hat in x_start_hats:
            #     # print(cands.indices.shape, input_ids_x.shape)
            #     print(mean_with_mask((x_start_hat-sample)**2, mse_mask).mean())
            #     print(diffusion._token_discrete_loss(x_start_hat, model.get_logits, cands.indices.squeeze(-1), mse_mask[:,:,0]).mean())
            # break

            # print(cands.indices[0])

            word_lst_recover = []
            word_lst_ref = []
            word_lst_source = []

            # tokenizer = load_tokenizer(args)
            gen_token_num = 0
            for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
                len_x = args.seq_len - sum(input_mask).tolist()
                tokens = tokenizer.decode_token_stop_at_sep(seq[len_x:])
                gen_token_num += len(tokens)
                word_lst_recover.append(tokens)

            for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
                # tokens = tokenizer.decode_token(seq)
                len_x = args.seq_len - sum(input_mask).tolist()
                word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
                word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

            for i in range(world_size):
                if i == rank:  # Write files sequentially
                    fout = open(out_file, 'a')
                    for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
                        print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
                    fout.close()
                dist.barrier()

        print('### Total takes {:.2f}s .....'.format(running_time))
        print('### {:.2f} tokens/second .....'.format(gen_token_num/running_time))

        print(f'### Written the decoded output to {out_file}')


if __name__ == "__main__":
    main()
