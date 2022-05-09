
import ray
import torch
from agents.DQN.dqn import DQNet
from agents.DQN.test import _test
from core.config import BaseAtariConfig
from core.storage import SharedStorage, QueueStorage
from agents.DQN.worker import DataWorker, PushWorker
from core.replay_buffer import ReplayBuffer


game_config = BaseAtariConfig(training_steps=5,
                              max_moves=200,  # moves for overall playing, atari 100k + some buffer to finish the game
                              total_transitions=1000,  # atari 100k
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
                              n_td_steps=5,
                              num_stack_obs=4,
                              clip_reward=True,
                              exp_path='results',
                              device='cpu',
                              game_name='SpaceInvadersNoFrameskip-v4'
                              )  # it controls the test max move

# game_config.set_game('SpaceInvadersNoFrameskip-v4')

# obs_shape, action_dim, out_mlp_hidden_dim, num_blocks, res_out_channels
model = DQNet((3 * 4, 96, 96), game_config.action_space_size, 32, 2, 64)
target_model = DQNet((3 * 4, 96, 96), game_config.action_space_size, 32, 2, 64)

model_storage = SharedStorage.remote(model, target_model)
model_storage.set_start_signal.remote()

batch_queue = QueueStorage(15, 20)
replay_buffer = ReplayBuffer.remote(game_config)

data_workers = [DataWorker.remote(rank, replay_buffer, model_storage, game_config) for rank in
                range(0, game_config.num_actors)]
push_worker = PushWorker.remote(batch_queue, replay_buffer, model_storage, game_config)
workers = [worker.run.remote() for worker in data_workers] + [push_worker.run.remote()]

workers += [_test.remote(game_config, model_storage)]

ray.wait(workers)
print('training...')

# workers.append(push_worker.run.remote())
# ray.wait(workers)

# ! need queue.push and queue.get to send replay buffer data to the queue waiting to be used

# self = replay_buffer

# if __name__ == '__main__':
#     self = data_workers[0]
#     g, p = ray.get(self.get_local_buffer.remote())[0]
#
#     imgs = ray.get(g.obs_history)
#     from matplotlib import pyplot as plt
#     plt.imshow(imgs[0])
#     plt.show()

