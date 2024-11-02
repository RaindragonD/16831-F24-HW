
## Problem 1
```
python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch1x32 --env_name cheetah-hw4_part1-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 1 --size 32 --scalar_log_freq -1 --video_log_freq -1 --mpc_action_sampling_strategy 'random'

python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q1_cheetah_n5_arch2x250 --env_name cheetah-hw4_part1-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 5 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1 --mpc_action_sampling_strategy 'random'

python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch2x250 --env_name cheetah-hw4_part1-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
```

## Problem 2
```
python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q2_obstacles_singleiteration --env_name obstacles-hw4_part1-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
```

## Problem 3
```
python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q3_obstacles --env_name obstacles-hw4_part1-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --n_iter 12 --video_log_freq -1 --mpc_action_sampling_strategy 'random'

python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q3_reacher --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size_initial 5000 --batch_size 5000 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random' 

python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q3_cheetah --env_name cheetah-hw4_part1-v0 --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 20 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
```

## Problem 4
```
python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q4_reacher_horizon5 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 5 --mpc_action_sampling_strategy 'random' --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'

python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q4_reacher_horizon15 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 15 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'

python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q4_reacher_horizon30 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 30 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'

python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q4_reacher_numseq100 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --mpc_num_action_sequences 100 --mpc_action_sampling_strategy 'random'

python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q4_reacher_numseq1000 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_num_action_sequences 1000 --mpc_action_sampling_strategy 'random'

python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q4_reacher_ensemble1 --env_name reacher-hw4_part1-v0 --ensemble_size 1 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'

python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q4_reacher_ensemble3 --env_name reacher-hw4_part1-v0 --ensemble_size 3 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'

python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q4_reacher_ensemble5 --env_name reacher-hw4_part1-v0 --ensemble_size 5 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
```

## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install some Python packages; see [installation.md](../hw1/installation.md) from homework 1 for instructions.

2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badges below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeCAR-Lab/16831-F24-HW/blob/main/hw4/rob831/hw4_part1/scripts/run_hw4_mb.ipynb) **Part I (Model Based Learning)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeCAR-Lab/16831-F24-HW/blob/main/hw4/rob831/hw4_part2/scripts/run_hw4_expl.ipynb) **Part II (Exploration)**

## Complete the code

### Part 1
The following files have blanks to be filled with your solutions from homework 1. The relevant sections are marked with `TODO: get this from Piazza'.

- [rob831/hw4_part1/infrastructure/rl_trainer.py](rob831/hw4_part1/infrastructure/rl_trainer.py)
- [rob831/hw4_part1/infrastructure/utils.py](rob831/hw4_part1/infrastructure/utils.py)

You will then need to implement code in the following files:
- [rob831/hw4_part1/agents/mb_agent.py](rob831/hw4_part1/agents/mb_agent.py)
- [rob831/hw4_part1/models/ff_model.py](rob831/hw4_part1/models/ff_model.py)
- [rob831/hw4_part1/policies/MPC_policy.py](rob831/hw4_part1/policies/MPC_policy.py)
- [rob831/hw4_part1/infrastructure/rl_trainer.py](rob831/hw4_part1/infrastructure/rl_trainer.py)
- [rob831/hw4_part1/agents/mbpo_agent.py](rob831/hw4_part1/infrastructure/rl_trainer.py)

The relevant sections are marked with `TODO`.

You may also want to look through [scripts/run_hw4_mb.py](rob831/hw4_part1/scripts/run_hw4_mb.py) and [scripts/run_hw4_mbpo.py](rob831/hw4_part1/scripts/run_hw4_mbpo.py) (if running locally) or [scripts/run_hw4_mb.ipynb](rob831/hw4_part1/scripts/run_hw4_mb.ipynb) and [scripts/run_hw4_mbpo.ipynb](rob831/hw4_part1/scripts/run_hw4_mbpo.ipynb) (if running on Colab), though you will not need to edit this files beyond changing runtime arguments in the Colab notebook.

### Part 2
The following files have blanks to be filled with your solutions from homework 1 and 3. The relevant sections are marked with `TODO'. You can get solutions from Ed. 

- [rob831/hw4_part2/infrastructure/utils.py](rob831/hw4_part2/infrastructure/utils.py)
- [rob831/hw4_part2/infrastructure/rl_trainer.py](rob831/hw4_part2/infrastructure/rl_trainer.py)
- [rob831/hw4_part2/policies/MLP_policy.py](rob831/hw4_part2/policies/MLP_policy.py)
- [rob831/hw4_part2/policies/argmax_policy.py](rob831/hw4_part2/policies/argmax_policy.py)
- [rob831/hw4_part2/critics/dqn_critic.py](rob831/hw4_part2/critics/dqn_critic.py)

You will then need to implement code in the following files:

For RND :
- [rob831/hw4_part2/exploration/rnd_model.py](rob831/hw4_part2/exploration/rnd_model.py)
- [rob831/hw4_part2/agents/explore_or_exploit_agent.py](rob831/hw4_part2/agents/explore_or_exploit_agent.py)

See the [assignment PDF](hw4.pdf) for more details on what files to edit.

