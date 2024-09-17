# BC

## Part 2
I loaded and calulated the expert policy returns in the notebook.

## Part 3
```bash
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
--video_log_freq -1
```

```bash 
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1
```

## Part 4
```bash
for i in {64, 128, 256, 512, 1024}
do
    python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant_${i} --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --video_log_freq -1
done
```

# DAgger

```bash
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name dagger_ant --n_iter 100 \
--do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
--video_log_freq -1
```

```bash
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name dagger_humanoid --n_iter 100 \
--do_dagger --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1
```