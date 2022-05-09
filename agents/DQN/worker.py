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
from core.storage import SharedStorage
from agents.DQN.dqn import DQNet


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

        self.eps_greedy = 1
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

    def get_priorities(self, i, pred_values_lst, greedy_values_lst, value_prefix_lst):
        # obtain the priorities at index i
        # print(f'pred_(N)_={len(pred_values_lst[i])}, '
        #       f'greedy_(N-n)_={len(greedy_values_lst[i])}, '
        #       f'prefix_(N)={len(value_prefix_lst[i])}')
        assert len([pred_values_lst[i]]) == len([value_prefix_lst[i]])
        if self.config.use_priority:
            pred_values = np.array(pred_values_lst[i])
            target_values = np.pad(greedy_values_lst[i], (0, self.config.td_steps)) + np.array(value_prefix_lst[i])

            priorities = abs(pred_values-target_values) + self.config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities

    def run(self):
        # number of parallel mcts
        env_nums = self.config.num_env  # num of envs for each actor
        n_td_step = self.config.td_steps
        discount = self.config.discount_factor
        n_step_weights = np.array([self.config.discount_factor ** i for i in range(self.config.td_steps)])
        model = DQNet((3 * 4, 96, 96), self.config.action_space_size, 32, 2, 64)
        model.to(self.device)
        model.eval()

        start_training = True  # trace whether the remote model is available to train
        # different seed for different games
        envs = [self.config.new_game(self.config.seed + self.rank * i) for i in range(env_nums)]

        def _get_max_entropy(action_space):
            p = 1.0 / action_space
            ep = - action_space * p * np.log2(p)
            return ep

        max_visit_entropy = _get_max_entropy(self.config.action_space_size)
        # determine the 100k benchmark: total_transitions <= 100K
        total_transitions = 0
        # max transition to collect for this data worker
        max_transitions = self.config.total_transitions // self.config.num_actors # for each actor
        with torch.no_grad():
            while True:
                # loops for rollout a batch of envs
                trained_steps = ray.get(self.model_storage.get_counter.remote())
                # training finished if reaching max training steps or max interaction steps (100K)
                if trained_steps >= self.config.training_steps or total_transitions > self.config.total_transitions:  # TODO training steps need to be used
                    print('reach max training/action steps')
                    time.sleep(30)
                    break

                init_obses = [env.reset() for env in envs]
                dones = np.array([False for _ in range(env_nums)])
                game_histories = [GameHistory(self.config.num_stack_obs,
                                              initial_obs=init_obses[i]) for i in range(env_nums)]

                # used to calculate n-step td priorities# TODO
                pred_values_lst = [[] for _ in range(env_nums)]
                greedy_values_lst = [[] for _ in range(env_nums)]
                value_prefix_lst = [[] for _ in range(env_nums)]
                reward_window_lst = [deque(maxlen=n_td_step) for _ in range(env_nums)]

                # some logs
                eps_ori_reward_lst, eps_clip_reward_lst, eps_steps_lst, action_entropies_lst = np.zeros(env_nums), np.zeros(
                    env_nums), np.zeros(env_nums), np.zeros(env_nums)
                step_counter = 0

                self_play_clip_rewards = 0.
                self_play_ori_rewards = 0.
                self_play_moves = 0.
                self_play_episodes = 0.

                self_play_clip_rewards_max = - np.inf
                self_play_moves_max = 0

                self_play_visit_entropy = []
                other_dist = {}

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
                        print('training is finished')
                        time.sleep(30)
                        return

                    # if start_training and (total_transitions / max_transitions) > (
                    #         trained_steps / self.config.training_steps):
                    #     # self-play is faster than training speed or finished
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
                                                                           self_play_clip_rewards_max, _temperature.mean(),
                                                                           visit_entropies, 0,
                                                                           other_dist)
                            self_play_clip_rewards_max = - np.inf  # reset for each new model weights!

                    step_counter += 1
                    for i in range(env_nums):
                        # reset env and its history if finished
                        if dones[i]:
                            # store current block trajectory
                            # pad zeros to value-prefix where boostrap is not required
                            if self.config.use_priority and start_training:
                                for s_ in range(n_td_step - 1):
                                    reward_window_lst[i].append(0)
                                    padded_r = np.pad(reward_window_lst[i], (0, n_td_step-len(reward_window_lst[i])))
                                    value_prefix_lst[i].append(n_step_weights @ padded_r)

                            priorities = self.get_priorities(i, pred_values_lst, greedy_values_lst, value_prefix_lst)
                            game_histories[i].game_over()

                            self.put((game_histories[i], priorities))  # add to local buffer
                            self.add_data_to_remote()

                            # reset the finished env and new a env
                            envs[i].close()
                            init_obs = envs[i].reset()
                            game_histories[i] = GameHistory(self.config.num_stack_obs, initial_obs=init_obs)

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
                            greedy_values_lst[i] = []
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

                        if random.random() < self.eps_greedy:  # epsilon-greedy
                            action = np.random.randint(env.action_space_size)
                        else:
                            action = action_probs.argmax().item()

                        # for log only
                        visit_entropy = entropy(action_probs, base=2)
                        action_entropies_lst[i] += visit_entropy

                        # make the action
                        obs, ori_reward, done, info = env.step(action)
                        # clip the reward
                        clip_reward = np.sign(ori_reward) if self.config.clip_reward else ori_reward

                        # store data
                        game_histories[i].add(action, obs, clip_reward)

                        eps_clip_reward_lst[i] += clip_reward
                        eps_ori_reward_lst[i] += ori_reward
                        dones[i] = done

                        eps_steps_lst[i] += 1
                        total_transitions += 1  # add 1 for each action

                        if self.config.use_priority and start_training:
                            # predicted Q value for that act
                            pred_values_lst[i].append(network_output[i][action].item())
                            reward_window_lst[i].append(clip_reward)
                            # value prefix
                            if len(reward_window_lst[i]) >= n_td_step:
                                value_prefix_lst[i].append(n_step_weights@reward_window_lst[i])
                            # n-step boostrap for n-previous Q estimation
                            if len(pred_values_lst[i]) > n_td_step:
                                greedy_values_lst[i].append(discount**n_td_step*max(network_output[i]))

                # Once max_moves achieved -> save the data to the replay buffer
                for i in range(env_nums):
                    env = envs[i]
                    env.close()

                    if dones[i]:
                        if self.config.use_priority and start_training:
                            for s_ in range(n_td_step - 1):
                                reward_window_lst[i].append(0)
                                padded_r = np.pad(reward_window_lst[i], (0, n_td_step - len(reward_window_lst[i])))
                                value_prefix_lst[i].append(n_step_weights @ padded_r)

                        priorities = self.get_priorities(i, pred_values_lst, greedy_values_lst, value_prefix_lst)
                        game_histories[i].game_over()

                        self.put((game_histories[i], priorities))
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

                other_dist = {}
                # send logs
                self.model_storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                               log_self_play_ori_rewards, log_self_play_rewards,
                                                               self_play_clip_rewards_max, _temperature.mean(),
                                                               visit_entropies, 0,
                                                               other_dist)
