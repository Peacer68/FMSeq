#!/bin/bash
# 数组定义
# CUDA_VISIBLE_DEVICES=0
tds=$(seq 0.0 0.1 0.0)

# 使用for循环遍历 seeds 中的元素
for td in $tds; do
    # if [[ $td == 0.5 ]]; then
    #     continue
    # fi
    python -u run_decode.py \
    --checkpoint diffusion_models/reflowseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed6868_test-qqp20240522-21:37:08/ema_0.9999_050000.pt \
    --seed 68 \
    --step 50 \
    --bsz 100 \
    --split test \
    --td $td \
    --cand_num 1
done
