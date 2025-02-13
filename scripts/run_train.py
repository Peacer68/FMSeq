import sys
import os
import argparse
import time
sys.path.append('.')
os.environ['NCCL_P2P_DISABLE']='1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--dataset', type=str, default='', help='name of training dataset')
    parser.add_argument('--data_dir', type=str, default='', help='path to training dataset')

    parser.add_argument('--noise_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'sqrt', 'trunc_cos', 'trunc_lin', 'pw_lin'], help='the distribution of noises')
    parser.add_argument('--diff_steps', type=int, default=4000, help='diffusion steps')
    parser.add_argument('--schedule_sampler', type=str, default='uniform', choices=['uniform', 'lossaware', 'fixstep'], help='schedule sampler of timesteps')

    parser.add_argument('--seq_len', type=int, default=128, help='max len of input sequence')
    parser.add_argument('--hidden_t_dim', type=int, default=128, help='hidden size of time embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden size of word embedding')
    parser.add_argument('--learning_steps', type=int, default=40000, help='total steps of learning')
    parser.add_argument('--save_interval', type=int, default=10000, help='save step')
    parser.add_argument('--resume_checkpoint', type=str, default='none', help='path to resume checkpoint, like xxx/xxx.pt')
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--bsz', type=int, default=64, help='batch size')
    parser.add_argument('--microbatch', type=int, default=64, help='microbatch size')
    parser.add_argument('--seed', type=int, default=101, help='random seed')

    parser.add_argument('--config_name', type=str, default='bert-base-uncased', help='config of pre-trained models')
    parser.add_argument('--vocab', type=str, default='bert', help='use bert vocab or load external vocab dict if given as path')
    parser.add_argument('--use_plm_init', type=str, default='no', choices=['no', 'bert'], help='load init parameter from the pre-trained lm')
    parser.add_argument('--sc_rate', type=float, default=0.0, help='the rate of self conditioning')
    parser.add_argument('--rescale_max', type=float, default=1.0, help='rescale range [0, rescale_max]')
    parser.add_argument('--loss_mask', type=str, default="remain y0, 1st pad, 2nd pad", help='the type of loss mask, ["remain all", "remain x0, y0", "remain y0, pad", "remain y0, 1st pad, 2nd pad", "remain y0"]')
    parser.add_argument('--merge_strategy', type=str, default="nonequal", help='strategy to merge x and y, equal means len(x)==len(y)')

    parser.add_argument('--notes', type=str, default='-', help='as training notes or specifical args')
    parser.add_argument('--app', type=str, default='', help='other input args')
    
    args = parser.parse_args()

    # set working dir to the upper folder
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname)
    os.chdir(dname)

    folder_name = "diffusion_models/"

    if int(os.environ['LOCAL_RANK']) == 0:
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

    Model_FILE = f"fmseq_{args.dataset}_h{args.hidden_dim}_lr{args.lr}" \
                f"_t{args.diff_steps}_{args.noise_schedule}_{args.schedule_sampler}" \
                f"_seed{args.seed}"
    if args.notes:
        args.notes += time.strftime("%Y%m%d-%H:%M:%S")
        Model_FILE = Model_FILE + f'_{args.notes}'
    Model_FILE = os.path.join(folder_name, Model_FILE)
    Log_FILE = os.path.join(Model_FILE, "train.log")

    if int(os.environ['LOCAL_RANK']) == 0:
        if not os.path.isdir(Model_FILE):
            os.mkdir(Model_FILE)
        os.system(f"touch {Log_FILE}")
    

    COMMANDLINE = f"OPENAI_LOGDIR={Model_FILE}  " \
                  f"TOKENIZERS_PARALLELISM=false " \
                  f"python train.py   " \
                  f"--checkpoint_path {Model_FILE} " \
                  f"--dataset {args.dataset} --data_dir {args.data_dir} --vocab {args.vocab} --use_plm_init {args.use_plm_init} " \
                  f"--lr {args.lr} " \
                  f"--dropout {args.dropout} " \
                  f"--batch_size {args.bsz} --microbatch {args.microbatch} " \
                  f"--diffusion_steps {args.diff_steps} " \
                  f"--noise_schedule {args.noise_schedule} " \
                  f"--schedule_sampler {args.schedule_sampler} --resume_checkpoint {args.resume_checkpoint} " \
                  f"--seq_len {args.seq_len} --hidden_t_dim {args.hidden_t_dim} --seed {args.seed} " \
                  f"--hidden_dim {args.hidden_dim} " \
                  f"--learning_steps {args.learning_steps} --save_interval {args.save_interval} " \
                  f"--config_name {args.config_name} --notes {args.notes} --sc_rate {args.sc_rate} "\
                  f"--rescale_max {args.rescale_max} --loss_mask '{args.loss_mask}' --merge_strategy '{args.merge_strategy}'"

    COMMANDLINE += " " + args.app

    print(COMMANDLINE)

    if int(os.environ['LOCAL_RANK']) == 0:
        with open(os.path.join(Model_FILE, 'saved_bash.sh'), 'w') as f:
            print(COMMANDLINE, file=f)
        os.system(f"{COMMANDLINE} | tee -a {Log_FILE}")
    else:
        os.system(COMMANDLINE)

    
    