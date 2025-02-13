import os, sys, glob
import argparse
import random
sys.path.append('.')
sys.path.append('..')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--checkpoint', type=str, default='', help='path to the checkpoint of diffusion model')
    parser.add_argument('--seed', type=int, default=101, help='random seed')
    parser.add_argument('--step', type=int, default=2000, help='if less than diffusion training steps, like 1000, use ddim sampling')

    parser.add_argument('--bsz', type=int, default=50, help='batch size')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'], help='dataset split used to decode')

    parser.add_argument('--top_p', type=int, default=-1, help='top p used in sampling, default is off')
    parser.add_argument('--pattern', type=str, default='ema', help='training pattern')
    parser.add_argument("--td", type=float, default=0, help="time difference")
    parser.add_argument("--cand_num", type=int, default=1, help="the number of candidate")
    
    args = parser.parse_args()

    # set working dir to the upper folder
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname)
    os.chdir(dname)

    output_lst = []
    model_dir = args.checkpoint.replace(args.checkpoint.split("/")[-1], "")
    for lst in glob.glob(model_dir):
        out_dir = 'generation_outputs'
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        COMMAND = f'python -m torch.distributed.launch --nproc_per_node=1 --master_port={12203 +random.randint(0,1000)+ int(args.seed)} --use_env sample_seq2seq_debug.py ' \
        f'--model_path {args.checkpoint} --step {args.step} ' \
        f'--batch_size {args.bsz} --seed2 {args.seed} --split {args.split} ' \
        f'--out_dir {out_dir} --top_p {args.top_p} --td {args.td} --cand_num {args.cand_num}'
        print(COMMAND)
        
        os.system(COMMAND)
    
    print('#'*30, 'decoding finished...')