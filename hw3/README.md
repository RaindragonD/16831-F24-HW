# Part I: Q-learning

```bash
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
--exp_name q1_dqn_1 --seed 1 --no_gpu
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
--exp_name q1_dqn_2 --seed 2 --no_gpu
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
--exp_name q1_dqn_3 --seed 3 --no_gpu
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
--exp_name q1_doubledqn_1 --double_q --seed 1 --no_gpu
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
--exp_name q1_doubledqn_2 --double_q --seed 2 --no_gpu
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 \
--exp_name q1_doubledqn_3 --double_q --seed 3 --no_gpu
```

# Part II: Actor-Critic

```bash
python rob831/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 \
-n 100 -b 1000 --exp_name q2_10_10 -ntu 10 -ngsptu 10 --no_gpu
python rob831/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 \
--exp_name q3_10_10 -ntu 10 -ngsptu 10 --no_gpu
```