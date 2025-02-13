import argparse
import torch
import json, os
import time
import numpy as np
from transformers import AutoTokenizer

from tokenizers.processors import BertProcessing
from tokenizers import ByteLevelBPETokenizer, decoders

from diffuseq import gaussian_diffusion as gd
from diffuseq.gaussian_diffusion import SpacedDiffusion, space_timesteps
from diffuseq.transformer_model import TransformerNetModel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

class myTokenizer():
    """
    Load tokenizer from bert config or defined BPE vocab dict
    """
    ################################################
    ### You can custome your own tokenizer here. ###
    ################################################
    def __init__(self, args, is_eval=False):
        if args.vocab == 'bert':
            tokenizer = AutoTokenizer.from_pretrained(args.config_name)
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            # save
            if not is_eval:
                tokenizer.save_pretrained(args.checkpoint_path)
        else: 
            # load vocab from the path
            print('#'*30, 'load vocab from', args.vocab)
            vocab_dict = {'[START]': 0, '[END]': 1, '[UNK]':2, '[PAD]':3}
            with open(args.vocab, 'r', encoding='utf-8') as f:
                for row in f:
                    vocab_dict[row.strip().split(' ')[0]] = len(vocab_dict)
            self.tokenizer = vocab_dict
            self.rev_tokenizer = {v: k for k, v in vocab_dict.items()}
            self.sep_token_id = vocab_dict['[END]']
            self.pad_token_id = vocab_dict['[PAD]']
            # save
            if int(os.environ['LOCAL_RANK']) == 0 and not is_eval:
                path_save_vocab = f'{args.checkpoint_path}/vocab.json'
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)
                
        self.vocab_size = len(self.tokenizer)
        args.vocab_size = self.vocab_size # update vocab size in args
    
    def encode_token(self, sentences):
        if isinstance(self.tokenizer, dict):
            input_ids = [[0] + [self.tokenizer.get(x, self.tokenizer['[UNK]']) for x in seq.split()] + [1] for seq in sentences]
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            input_ids = self.tokenizer(sentences, add_special_tokens=True)['input_ids']
        else:
            assert False, "invalid type of vocab_dict"
        return input_ids
        
    def decode_token(self, seq):
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = self.tokenizer.decode(seq)
        else:
            assert False, "invalid type of vocab_dict"
        return tokens
    
    def decode_token_stop_at_sep(self, seq, return_length=None):
        length = 0
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).cpu().detach().numpy()
            idx = (seq == self.sep_token_id).argmax()
            length = idx+1
            if seq[idx] == self.sep_token_id:
                seq = seq[:idx+1]
            else:
                return self.decode_token(torch.tensor(seq).unsqueeze(-1))


            tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).cpu().detach().numpy()
            idx = (seq == self.sep_token_id).argmax()
            length = idx+1
            if seq[idx] == self.sep_token_id:
                seq = seq[:idx+1]
            else:
                return self.decode_token(torch.tensor(seq).unsqueeze(-1))
            
            tokens = self.tokenizer.decode(seq)
        else:
            assert False, "invalid type of vocab_dict"
        if return_length:
            return length, tokens
        else:
            return tokens
        
class mtTokenizer():
    def __init__(self, args):
        path = "./datasets/iwslt14-de-en/BPE"
        # 原本为<s>:0, <pad>:1, 需要将<s>改为1，<pad>改为0
        with open(f"{path}/vocab.json", "r") as f:
            vocab_dict = json.load(f)
        vocab_dict["<s>"], vocab_dict["<pad>"] = vocab_dict["<pad>"], vocab_dict["<s>"]
        with open(f"{path}/vocab_processed.json", "w") as f:
            json.dump(vocab_dict, f)
        
        tokenizer = ByteLevelBPETokenizer(
        f"{path}/vocab_processed.json",
        f"{path}/merges.txt",
    )

        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )

        # tokenizer.enable_truncation(max_length=512)


        # add length method to tokenizer object
        tokenizer.vocab_size = len(vocab_dict)

        # add length property to tokenizer object
        tokenizer.__len__ = property(lambda self: self.vocab_size)

        tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer = tokenizer
        self.sep_token_id = 2
        self.pad_token_id = 0
        self.vocab_size = len(vocab_dict)
        args.vocab_size = self.vocab_size # update vocab size in args
        # save
        # tokenizer.save_pretrained(args.checkpoint_path)
    def encode_token(self, sentences):
        input_ids = [self.tokenizer.encode(seq).ids for seq in sentences]
        return input_ids
    def decode_token(self, seq):
        seq = seq.squeeze(-1).tolist()
        while len(seq)>0 and seq[-1] == self.pad_token_id:
            seq.pop()
        tokens = self.tokenizer.decode(seq)
        return tokens
    def decode_token_stop_at_sep(self, seq):
        length = 0
        seq = seq.squeeze(-1).cpu().detach().numpy()
        idx = (seq == self.sep_token_id).argmax()
        length = idx+1
        if seq[idx] == self.sep_token_id:
            seq = seq[:idx+1]

        return self.decode_token(torch.tensor(seq).unsqueeze(-1))





def load_model_emb(args, tokenizer):
    ### random emb or pre-defined embedding like glove embedding. You can custome your own init here.
    model = torch.nn.Embedding(tokenizer.vocab_size, args.hidden_dim)
    path_save = '{}/random_emb.torch'.format(args.checkpoint_path)
    path_save_ind = path_save + ".done"
    if int(os.environ['LOCAL_RANK']) == 0:
        if os.path.exists(path_save):
            print('reload the random embeddings', model)
            model.load_state_dict(torch.load(path_save))
        else:
            print('initializing the random embeddings', model)
            torch.nn.init.normal_(model.weight)
            torch.save(model.state_dict(), path_save)
            os.sync()
            with open(path_save_ind, "x") as _:
                pass
    else:
        while not os.path.exists(path_save_ind):
            time.sleep(1)
        print('reload the random embeddings', model)
        model.load_state_dict(torch.load(path_save))

    return model, tokenizer


def load_tokenizer(args, is_eval=False):
    print(args.dataset)
    if args.dataset == "de2en":
        tokenizer = mtTokenizer(args)
    else:
        tokenizer = myTokenizer(args, is_eval=is_eval)
    return tokenizer

def load_defaults_config():
    """
    Load defaults for training args.
    """
    with open('diffuseq/config.json', 'r') as f:
        return json.load(f)


def create_model_and_diffusion(
    hidden_t_dim,
    hidden_dim,
    vocab_size,
    config_name,
    use_plm_init,
    dropout,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    notes,
    sc_rate,
    **kwargs,
):
    model = TransformerNetModel(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim*2),
        hidden_t_dim=hidden_t_dim,
        dropout=dropout,
        config_name=config_name,
        vocab_size=vocab_size,
        init_pretrained=use_plm_init,
        sc_rate=sc_rate
    )

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas,
        rescale_max=kwargs["rescale_max"],
        loss_mask=kwargs["loss_mask"]
    )

    return model, diffusion


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def straightness(traj):
    N = len(traj) - 1
    dt = 1 / N
    base = traj[0] - traj[-1]
    mse = []
    for i in range(1, len(traj)):
        v = (traj[i-1] - traj[i]) / dt
        mse.append(torch.mean((v - base) ** 2))
    return torch.mean(torch.stack(mse))
