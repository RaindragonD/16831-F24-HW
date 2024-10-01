
{ time CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --num_threads 1 --exp_name q6_1_thread; } 2>&1 | tee q6_1_thread_time.log
{ time CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --num_threads 32 --exp_name q6_32_threads; } 2>&1 | tee q6_32_threads_time.log