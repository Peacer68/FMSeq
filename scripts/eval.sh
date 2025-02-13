# CUDA_VISIBLE_DEVICES=5
generation_outputs=""
tds=$(seq 0.0 0.1 0.0)
for td in $tds; do
python eval_seq2seq.py --mbr\
 --folder $generation_outputs
done