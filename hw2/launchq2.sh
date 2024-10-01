
for b in 1000 5000 10000
do
  for lr in 0.005 0.01 0.02 0.03
  do
    CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
    --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $lr -rtg \
    --exp_name q2_b${b}_r${lr}
  done
done