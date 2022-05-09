import ray
import torch
from agents.DQN.dqn import DQNet
from core.config import BaseAtariConfig
from core.storage import SharedStorage
from agents.DQN.worker import DataWorker
from core.replay_buffer import ReplayBuffer

game_config = BaseAtariConfig(training_steps=5,
                              max_moves=200,  # moves for overall playing, atari 100k + some buffer to finish the game
                              total_transitions=100,  # atari 100k
                              test_max_moves=12000,
                              gray_scale=False,
                              episode_life=True,
                              cvt_string=False,
                              image_based=True,
                              frame_skip=4,  # sticky actions...
                              num_env=10,
                              num_actors=1,
                              checkpoint_interval=10,
                              use_priority=True,
                              prioritized_replay_eps=1e-6,
                              td_steps=5,
                              num_stack_obs=4,
                              clip_reward=True,
                              exp_path='results',
                              device='cpu')  # it controls the test max move

game_config.set_game('SpaceInvadersNoFrameskip-v4')

# obs_shape, action_dim, out_mlp_hidden_dim, num_blocks, res_out_channels
policy_model = DQNet((3 * 4, 96, 96), game_config.action_space_size, 32, 2, 64)
target_model = DQNet((3 * 4, 96, 96), game_config.action_space_size, 32, 2, 64)
storage = SharedStorage.remote(policy_model, target_model)

replay_buffer = ReplayBuffer.remote(config=game_config)

data_workers = [DataWorker.remote(rank, replay_buffer, storage, game_config) for rank in
                range(0, game_config.num_actors)]
workers = [worker.run.remote() for worker in data_workers]
ray.wait(workers)

# ! need queue.push and queue.get to send replay buffer data to the queue waiting to be used

self = replay_buffer


# if __name__ == '__main__':
#     self = data_workers[0]
#     g, p = ray.get(self.get_local_buffer.remote())[0]
#
#     imgs = ray.get(g.obs_history)
#     from matplotlib import pyplot as plt
#     plt.imshow(imgs[0])
#     plt.show()

