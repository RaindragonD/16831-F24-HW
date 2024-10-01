from collections import OrderedDict
import pickle
import os
import sys
import time

import gym
from gym import wrappers
import numpy as np
import torch
from rob831.infrastructure import pytorch_util as ptu

from rob831.infrastructure import utils
from rob831.infrastructure.logger import Logger
from rob831.infrastructure.action_noise_wrapper import ActionNoiseWrapper

import multiprocessing
from functools import partial

# Add this at the beginning of the file, after imports
if torch.cuda.is_available():
    multiprocessing.set_start_method('spawn', force=True)

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        # Add new parameter for number of parallel threads
        self.num_threads = params.get('num_threads', 1)

        # We'll create environments in each worker process instead of here

        #############
        ## ENV
        #############

        # Make the gym environment
        self.envs = [self.create_env(i) for i in range(self.num_threads)]

        # import plotting (locally if 'obstacles' env)
        if not(self.params['env_name']=='obstacles-rob831-v0'):
            import matplotlib
            matplotlib.use('Agg')

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.envs[0].spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.envs[0].action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.envs[0].observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.envs[0].observation_space.shape if img else self.envs[0].observation_space.shape[0]
        ac_dim = self.envs[0].action_space.n if discrete else self.envs[0].action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.envs[0]):
            self.fps = 1/self.envs[0].model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.envs[0].env.metadata.keys():
            self.fps = self.envs[0].env.metadata['video.frames_per_second']
        else:
            self.fps = 10


        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.envs[0], self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.log_metrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(itr,
                                initial_expertdata, collect_policy,
                                self.params['batch_size'])
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            train_logs = self.train_agent()

            # log/save
            if self.log_video or self.log_metrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy, train_video_paths, train_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, load_initial_expertdata, collect_policy, batch_size):
        # TODO: get this from hw1
        if itr == 0:
            if load_initial_expertdata:
                paths = pickle.load(open(self.params['expert_data'], 'rb'))
                return paths, 0, None
            else:
                num_transitions_to_sample = self.params['batch_size_initial']
        else:
            num_transitions_to_sample = self.params['batch_size']

        print("\nCollecting data to be used for training...")
        
        if self.num_threads > 1:
            paths, envsteps_this_batch = self.parallel_sample_trajectories(
                collect_policy, num_transitions_to_sample, self.params['ep_len'])
        else:
            paths, envsteps_this_batch = utils.sample_trajectories(
                self.envs[0], collect_policy, num_transitions_to_sample, self.params['ep_len'])

        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    def parallel_sample_trajectories(self, collect_policy, num_transitions_to_sample, ep_len):
        num_cpus = multiprocessing.cpu_count()
        
        transitions_per_thread = num_transitions_to_sample // self.num_threads + 1

        sample_trajectory_partial = partial(
            utils.sample_trajectories,
            policy=collect_policy,
            min_timesteps_per_batch=transitions_per_thread,
            max_path_length=ep_len,
        )

        with multiprocessing.Pool(processes=min(self.num_threads, num_cpus)) as pool:
            results = []
            for i in range(self.num_threads):
                results.append(pool.apply_async(sample_trajectory_partial, (self.envs[i],)))

            pool.close()
            pool.join()

        paths = []
        envsteps_this_batch = 0
        for result in results:
            thread_paths, thread_envsteps = result.get()
            paths.extend(thread_paths)
            envsteps_this_batch += thread_envsteps

        return paths, envsteps_this_batch
    
    def train_agent(self):
        # TODO: get this from hw1
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.envs[0], eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.envs[0], eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

    def create_env(self, i):
        env = gym.make(self.params['env_name'])
        env.seed(self.params['seed']+i)
        if self.params['action_noise_std'] > 0:
            env = ActionNoiseWrapper(env, self.params['seed'], self.params['action_noise_std'])
        return env