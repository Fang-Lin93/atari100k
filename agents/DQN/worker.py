import random

import ray
import time
import torch

import numpy as np
from collections import deque
from scipy.stats import entropy
from torch.nn import L1Loss
from core.history import GameHistory
from core.config import BaseAtariConfig
from core.utils import prepare_observation_lst
from core.replay_buffer import ReplayBuffer
from core.storage import SharedStorage, QueueStorage
from agents.DQN.dqn import DQNet


@ray.remote
class PushWorker(object):
    """
    fetch data from the replay buffer and send it to the queue
    """

    def __init__(self, batch_queue: QueueStorage, replay_buffer: ReplayBuffer, model_storage: SharedStorage,
                 config: BaseAtariConfig):
        self.config = config
        self.storage = model_storage
        self.replay_buffer = replay_buffer
        self.batch_queue = batch_queue

    def run(self):
        """
        create data batch and send to the queue
        """
        start = False
        init_beta = self.config.priority_prob_beta
        beta_warm_step = self.config.prio_beta_warm_step

        while True:
            # wait for starting
            if not start:
                start = ray.get(self.storage.get_start_signal.remote())
                time.sleep(1)
                continue

            train_step = ray.get(self.storage.get_counter.remote())
            # increase beta: 0.4 -> 1
            beta = init_beta + (1-init_beta)*train_step/beta_warm_step if train_step < beta_warm_step else 1
            self.make_batch(self.config.batch_size, beta, self.config.n_td_steps)
            time.sleep(1)

    def make_batch(self, batch_size, beta, n_td_steps):
        """
        compute n-step return targets and form a batch of data
        push the data to the queue waiting for training
        decay_weights = 1, gamma, gamma**2,...
        obs: current obs
        actions: action int
        n_step_reward: value prefix
        next_obs: obs after value prefix, use target model to eval
        next_obs_pos_in_batch: whether it is required to use
        """
        if ray.get(self.replay_buffer.get_total_len.remote()) < batch_size:
            return

        # print('fetch!!')

        game_lst, game_pos_lst, indices_lst, weights_lst, make_time = \
            ray.get(self.replay_buffer.prepare_batch_context.remote(batch_size, beta))
        obs, actions, greedy_actions, n_step_reward, next_obs, next_obs_pos_in_batch, is_w = [], [], [], [], [], [], []
        for i, (game_prefix, pos, weight) in enumerate(zip(game_lst, game_pos_lst, weights_lst)):
            game, prefix = game_prefix
            obs.append(game[pos])
            actions.append(game.actions[pos])
            n_step_reward.append(prefix[pos])
            is_w.append(weight)

            # once need bootstrapping
            if pos + n_td_steps < len(game):
                greedy_actions.append(game.greedy_actions[pos + n_td_steps])
                next_obs.append(game[pos + n_td_steps])
                next_obs_pos_in_batch.append(i)
        # obs = [i:i+stack_obs_num] be (-3, -2, -1, now) as obs.

        if self.config.image_based:
            obs, next_obs = prepare_observation_lst(obs), prepare_observation_lst(next_obs)
            obs, next_obs = torch.from_numpy(obs).float() / 255.0, torch.from_numpy(next_obs).float() / 255.0
        else:
            obs, next_obs = torch.from_numpy(np.array(obs)), torch.from_numpy(np.array(next_obs))

        batch = (obs, actions, n_step_reward, next_obs, greedy_actions, next_obs_pos_in_batch, indices_lst, is_w, make_time)
        self.batch_queue.push(batch)


# @ray.remote(num_gpus=0.125)
@ray.remote
class DataWorker(object):
    def __init__(self, rank, replay_buffer: ReplayBuffer, storage: SharedStorage, config: BaseAtariConfig):
        """Data Worker for collecting data through self-play
        Parameters
        ----------
        rank: int
            id of the worker
        replay_buffer: ReplayBuffer
            Shared Replay buffer for the central learner
        storage: Any
            The model storage for the central learner
        """
        self.rank = rank
        self.config = config
        self.model_storage = storage
        self.replay_buffer = replay_buffer
        # double buffering when data is sufficient
        self.local_buffer = []
        self.pool_size = 1
        self.device = self.config.device
        self.last_model_index = -1

        self.eps_greedy = 1.
        # to scale n future rewards

    def put(self, data):
        # put a game history into the pool
        # a sequence of history  <--> a sequence of priorities (1-1 correspondence)
        self.local_buffer.append(data)

    def get_local_buffer(self):
        return self.local_buffer

    def len_pool(self):
        # current pool size
        return len(self.local_buffer)

    def add_data_to_remote(self):
        # push the local buffer to the remote(central) buffer
        if self.len_pool() >= self.pool_size:
            self.replay_buffer.save_pools.remote(self.local_buffer)
            del self.local_buffer[:]

    def get_priorities(self, i, pred_values_lst, boostrap_values_lst, value_prefix_lst):
        # obtain the priorities at index i
        # print(f'pred_(N)_={len(pred_values_lst[i])}, '
        #       f'greedy_(N-n)_={len(boostrap_values_lst[i])}, '
        #       f'prefix_(N)={len(value_prefix_lst[i])}')
        assert len([pred_values_lst[i]]) == len([value_prefix_lst[i]])
        if self.config.use_priority:
            pred_values = np.array(pred_values_lst[i])
            if len(boostrap_values_lst[i]) > 0:  # the episode is too short! no boostrap required  TODO
                target_values = np.pad(boostrap_values_lst[i], (0, self.config.n_td_steps)) \
                                + np.array(value_prefix_lst[i])
            else:
                target_values = np.array(value_prefix_lst[i])

            priorities = abs(pred_values - target_values) + self.config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities

    def run(self):
        # number of parallel mcts
        env_nums = self.config.num_env  # num of envs for each actor
        n_td_step = self.config.n_td_steps
        discount = self.config.discount_factor
        n_step_weights = np.array([self.config.discount_factor ** i for i in range(self.config.n_td_steps)])
        model = DQNet(self.config.obs_shape,
                      self.config.action_space_size,
                      self.config.out_mlp_hidden_dim,
                      self.config.num_blocks,
                      self.config.res_out_channels)
        model.to(self.device)
        model.eval()

        start_training = False  # trace whether the remote model is available to train
        # different seed for different games
        # envs = [self.config.new_game(self.config.seed + self.rank * i) for i in range(env_nums)]
        envs = [self.config.new_game() for _ in range(env_nums)]  # remove seed

        def _get_max_entropy(action_space):
            p = 1.0 / action_space
            ep = - action_space * p * np.log2(p)
            return ep

        max_visit_entropy = _get_max_entropy(self.config.action_space_size)
        # determine the 100k benchmark: total_transitions <= 100K
        total_transitions = 0
        # used to control the replay_ratio speed
        # replay_ratio = trained_steps/trans_collected
        # trans_collected = step_counter * num_envs * num_actors
        # If trans_collected * replay_ratio > trained_steps: slow down !
        replay_ratio_multiplier = self.config.replay_ratio * self.config.num_env * self.config.num_actors
        # max transition to collect for this data worker
        max_transitions = self.config.total_transitions // self.config.num_actors  # for each actor
        with torch.no_grad():
            while True:
                # loops for rollout a batch of envs
                trained_steps = ray.get(self.model_storage.get_counter.remote())

                # record the running replay ratio

                # training finished if reaching max training steps or max interaction steps (100K)
                if trained_steps >= self.config.training_steps:
                    print('reach max training/action steps')
                    time.sleep(5)
                    break

                init_obses = [env.reset() for env in envs]
                dones = np.array([False for _ in range(env_nums)])
                game_histories = [GameHistory(self.config.num_stack_obs,
                                              initial_obs=init_obses[i],
                                              cvt_string=self.config.cvt_string) for i in range(env_nums)]

                # used to calculate n-step td priorities# TODO
                pred_values_lst = [[] for _ in range(env_nums)]
                boostrap_values_lst = [[] for _ in range(env_nums)]
                value_prefix_lst = [[] for _ in range(env_nums)]
                reward_window_lst = [deque(maxlen=n_td_step) for _ in range(env_nums)]

                # some logs
                eps_ori_reward_lst, eps_clip_reward_lst, eps_steps_lst, action_entropies_lst = np.zeros(
                    env_nums), np.zeros(
                    env_nums), np.zeros(env_nums), np.zeros(env_nums)
                step_counter = 0

                self_play_clip_rewards = 0.
                self_play_ori_rewards = 0.
                self_play_moves = 0.
                self_play_episodes = 0.

                self_play_clip_rewards_max = - np.inf
                self_play_moves_max = 0

                self_play_visit_entropy = []

                other_distribution = {}

                # one loop: rollout N envs to finish all
                # max_moves is to limit the maximal moves for each loop (rollout one batch of envs)
                while step_counter <= self.config.max_moves:
                    if not start_training:
                        # is central learner available to train?
                        start_training = ray.get(self.model_storage.get_start_signal.remote())

                    # remote model reaches max number of training times
                    trained_steps = ray.get(self.model_storage.get_counter.remote())
                    if trained_steps >= self.config.training_steps:
                        # training is finished
                        time.sleep(1)
                        return

                    # data efficient demand !
                    # replay-ratio = trained_steps / env_trans collected
                    # if start_training and (replay_ratio_multiplier*step_counter > trained_steps > 0):
                    #     # self-play is faster than training speed or finished
                    #     # wait the learner for some time
                    #     # print(f'saved_transitions={total_transitions}/{max_transitions*self.config.num_actors},'
                    #     #       f' trained_steps={self.config.batch_size*trained_steps}/{self.config.training_steps}'
                    #     #       f' self-play suspended, waiting learning...')
                    #     time.sleep(1)
                    #     continue

                    # data inefficient but fast learning
                    # if start_training and ray.get(self.replay_buffer.is_full.remote()):
                    #     # print('replay_buffer is full... waiting training')
                    #     time.sleep(1)
                    #     continue

                    _temperature = np.ones(env_nums)

                    # update the model to the remote one in self-play every checkpoint_interval
                    new_model_index = trained_steps // self.config.checkpoint_interval
                    if new_model_index > self.last_model_index:
                        self.last_model_index = new_model_index
                        # update model
                        weights = ray.get(self.model_storage.get_weights.remote())
                        model.set_weights(weights)
                        model.to(self.device)
                        model.eval()

                        # log if more than 1 env in parallel because env will reset in this loop.
                        if env_nums > 1:
                            if len(self_play_visit_entropy) > 0:
                                visit_entropies = np.array(self_play_visit_entropy).mean()
                                visit_entropies /= max_visit_entropy
                            else:
                                visit_entropies = 0.

                            if self_play_episodes > 0:
                                log_self_play_moves = self_play_moves / self_play_episodes
                                log_self_play_rewards = self_play_clip_rewards / self_play_episodes
                                log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                            else:
                                log_self_play_moves = 0
                                log_self_play_rewards = 0
                                log_self_play_ori_rewards = 0

                            self.model_storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                           log_self_play_ori_rewards,
                                                                           log_self_play_rewards,
                                                                           self_play_clip_rewards_max,
                                                                           _temperature.mean(),
                                                                           visit_entropies, 0,
                                                                           other_distribution)
                            self_play_clip_rewards_max = - np.inf  # reset for each new model weights!

                    step_counter += 1
                    for i in range(env_nums):
                        # reset env and its history if finished
                        if dones[i]:
                            # store current block trajectory
                            # pad zeros to value-prefix where boostrap is not required
                            for s_ in range(n_td_step - 1):
                                reward_window_lst[i].append(0)
                                padded_r = np.pad(reward_window_lst[i], (0, n_td_step - len(reward_window_lst[i])))
                                value_prefix_lst[i].append(n_step_weights @ padded_r)

                            priorities = self.get_priorities(i, pred_values_lst, boostrap_values_lst, value_prefix_lst)
                            game_histories[i].game_over()

                            self.put((game_histories[i], value_prefix_lst[i], priorities))  # add to local buffer
                            self.add_data_to_remote()

                            # reset the finished env and new a env
                            envs[i].close()
                            init_obs = envs[i].reset()
                            game_histories[i] = GameHistory(self.config.num_stack_obs,
                                                            initial_obs=init_obs,
                                                            cvt_string=self.config.cvt_string)

                            # log
                            self_play_clip_rewards_max = max(self_play_clip_rewards_max, eps_clip_reward_lst[i])
                            self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                            self_play_clip_rewards += eps_clip_reward_lst[i]
                            self_play_ori_rewards += eps_ori_reward_lst[i]
                            self_play_visit_entropy.append(action_entropies_lst[i] / eps_steps_lst[i])
                            self_play_moves += eps_steps_lst[i]
                            self_play_episodes += 1

                            # reset priority log
                            pred_values_lst[i] = []
                            boostrap_values_lst[i] = []
                            value_prefix_lst[i] = []
                            reward_window_lst[i] = deque(maxlen=n_td_step)

                            # reset episode log
                            eps_steps_lst[i] = 0
                            eps_clip_reward_lst[i] = 0
                            eps_ori_reward_lst[i] = 0
                            action_entropies_lst[i] = 0

                    # stack obs for model inference
                    stack_obs = [game.step_obs() for game in game_histories]
                    if self.config.image_based:
                        stack_obs = prepare_observation_lst(stack_obs)
                        stack_obs = torch.from_numpy(stack_obs).to(self.device).float() / 255.0
                    else:
                        stack_obs = [game_history.step_obs() for game_history in game_histories]
                        stack_obs = torch.from_numpy(np.array(stack_obs)).to(self.device)

                    network_output = model(stack_obs.float())  # batch size = number of games...

                    # one-step interaction to the envs.
                    for i in range(env_nums):
                        env = envs[i]
                        action_probs = network_output[i]

                        # print(network_output.shape)  # = (B=num_envs, action_space)
                        # print('prob', action_probs)

                        greedy_act = action_probs.argmax().item()

                        if random.random() < self.eps_greedy:  # epsilon-greedy
                            action = np.random.randint(env.action_space_size)
                        else:
                            action = greedy_act

                        # for log only
                        visit_entropy = entropy(action_probs, base=2)
                        action_entropies_lst[i] += visit_entropy

                        # make the action
                        obs, ori_reward, done, info = env.step(action)
                        # clip the reward
                        clip_reward = np.sign(ori_reward) if self.config.clip_reward else ori_reward

                        # store data
                        game_histories[i].add(action, greedy_act, obs, clip_reward)

                        eps_clip_reward_lst[i] += clip_reward
                        eps_ori_reward_lst[i] += ori_reward
                        dones[i] = done

                        eps_steps_lst[i] += 1
                        total_transitions += 1  # add 1 for each action
                        # 1 -> 0.1 at ,max_transitions/2
                        self.eps_greedy = max(1-1.8*total_transitions/max_transitions, 0.1)

                        # value prefix
                        reward_window_lst[i].append(clip_reward)
                        if len(reward_window_lst[i]) >= n_td_step:
                            value_prefix_lst[i].append(n_step_weights @ reward_window_lst[i])

                        if self.config.use_priority and start_training:
                            # predicted Q value for that act
                            pred_values_lst[i].append(network_output[i][action].item())
                            # n-step boostrap for n-previous Q estimation
                            if len(pred_values_lst[i]) > n_td_step:
                                boostrap_values_lst[i].append(discount ** n_td_step * max(network_output[i]))

                # Once max_moves achieved -> save the data to the replay buffer
                for i in range(env_nums):
                    env = envs[i]
                    env.close()

                    if dones[i]:
                        for s_ in range(n_td_step - 1):
                            reward_window_lst[i].append(0)
                            padded_r = np.pad(reward_window_lst[i], (0, n_td_step - len(reward_window_lst[i])))
                            value_prefix_lst[i].append(n_step_weights @ padded_r)

                        priorities = self.get_priorities(i, pred_values_lst, boostrap_values_lst, value_prefix_lst)
                        game_histories[i].game_over()

                        self.put((game_histories[i], value_prefix_lst[i], priorities))
                        self.add_data_to_remote()

                        self_play_clip_rewards_max = max(self_play_clip_rewards_max, eps_clip_reward_lst[i])
                        self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                        self_play_clip_rewards += eps_clip_reward_lst[i]
                        self_play_ori_rewards += eps_ori_reward_lst[i]
                        self_play_visit_entropy.append(action_entropies_lst[i] / eps_steps_lst[i])
                        self_play_moves += eps_steps_lst[i]
                        self_play_episodes += 1
                    else:
                        # if the final game history is not finished, we will not save this data.
                        total_transitions -= len(game_histories[i])  # len(gamehistory) = #(actions)

                # logs
                visit_entropies = np.array(self_play_visit_entropy).mean()
                visit_entropies /= max_visit_entropy

                if self_play_episodes > 0:
                    log_self_play_moves = self_play_moves / self_play_episodes
                    log_self_play_rewards = self_play_clip_rewards / self_play_episodes
                    log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                else:
                    log_self_play_moves = 0
                    log_self_play_rewards = 0
                    log_self_play_ori_rewards = 0

                other_distribution = {}
                # send logs
                self.model_storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                               log_self_play_ori_rewards, log_self_play_rewards,
                                                               self_play_clip_rewards_max, _temperature.mean(),
                                                               visit_entropies, 0,
                                                               other_distribution)
