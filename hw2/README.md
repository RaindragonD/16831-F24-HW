
# Q1 RTG/DSA, CartPole-v0
```bash
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na
```

# Q2 RTG/DSA, InvertedPendulum-v4
```bash

for b in 1000 5000 10000
do
  for lr in 0.005 0.01 0.02 0.03
  do
    CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
    --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $lr -rtg \
    --exp_name q2_b${b}_r${lr}
  done
done
```

# Q3 Baseline, LunarLanderContinuous-v2
```bash
python rob831/scripts/run_hw2.py \
--env_name LunarLanderContinuous-v2 --ep_len 1000 \
--discount 0.99 -n 100 -l 2 -s 64 -b 10000 -lr 0.005 \
--reward_to_go --nn_baseline --exp_name q3_b10000_r0.005
```

# Q4 Baseline, HalfCheetah-v4
```bash
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 \
--exp_name q4_search_b10000_lr0.02
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 -rtg \
--exp_name q4_search_b10000_lr0.02_rtg
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 --nn_baseline \
--exp_name q4_search_b10000_lr0.02_nnbaseline
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 -rtg --nn_baseline \
--exp_name q4_search_b10000_lr0.02_rtg_nnbaseline

for b in 10000 30000 50000
do
  for lr in 0.005 0.01 0.02
  do
    python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
    --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $lr -rtg --nn_baseline \
    --exp_name q4_b${b}_lr${lr}
  done
done
```

# Q5 GAE, Hopper-v4 
```bash
for gae_lambda in 0 0.95 0.99 1
do
  python rob831/scripts/run_hw2.py \
  --env_name Hopper-v4 --ep_len 1000 \
  --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 \
  --reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda $gae_lambda \
  --exp_name q5_b2000_r0.001_lambda${gae_lambda}
done
```

# Q6 Parallel Sampling, CartPole-v0
```bash
{ time python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --num_threads 1 --exp_name q6_1_thread; } 2>&1 | tee q6_1_thread_time.log
{ time python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --num_threads 32 --exp_name q6_32_threads; } 2>&1 | tee q6_32_threads_time.log
```