python -m torch.distributed.launch --nproc_per_node=1 --master_port=12332 --use_env run_train.py \
--diff_steps 2000 \
--lr 0.0001 \
--config_name "bert-de2en" \
--dropout 0.3 \
--learning_steps 300000 \
--save_interval 2000 \
--seed 6868 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 1024 \
--microbatch 96 \
--dataset de2en \
--data_dir ./datasets/iwslt14-de-en \
--vocab de2en \
--seq_len 128 \
--schedule_sampler lossaware \
--resume_checkpoint "None" \
--notes test-de2en \
--sc_rate 0.5 \
--rescale_max 1000 \
--loss_mask "remain y0, pad" \
--merge_strategy "equal"