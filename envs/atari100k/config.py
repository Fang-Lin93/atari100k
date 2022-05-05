#
# import numpy as np
# from envs.atari100k.game import make_atari, EpisodicLifeEnv, WarpFrame, AtariGame
#
#
# class DiscreteSupport(object):
#     """
#     Distributional rewards.
#     Then using classification loss rather than regression loss
#     """
#     def __init__(self, min_: int, max_: int, delta=1.):
#         assert min_ < max_
#         self.min = min_
#         self.max = max_
#         self.range = np.arange(min_, max_ + 1, delta)
#         self.size = len(self.range)
#         self.delta = delta
#
#
# class BaseConfig(object):
#     """
#     Basic Atari env. config for ANY algorithms
#     This config keeps the basic observation setting
#     """
#
#     def __init__(self,
#                  max_moves: int = 108000,
#                  test_max_moves: int = 12000,
#                  gray_scale: bool = False,
#                  episode_life: bool = False,
#                  cvt_string: bool = False,
#                  image_based: bool = True,
#                  frame_skip: int = 4,  # sticky actions...
#                  stacked_observations: int = 4):
#
#         """Base Config for EfficientZero
#         Parameters
#         ----------
#         max_moves: int
#             max number of moves for an episode
#         test_max_moves: int
#             max number of moves for an episode during testing (in training stage),
#             set this small to make sure the game will end faster.
#         gray_scale: bool
#             True -> use gray image observation
#         episode_life: bool
#             True -> one life in atari100k games
#         cvt_string: bool
#             True -> convert the observation into string in the replay buffer
#         image_based: bool
#             True -> observation is image based
#         frame_skip: int
#             number of frame skip
#         stacked_observations: int
#             number of frame stack
#         """
#
#         # Self-Play
#         self.max_moves = max_moves
#         self.test_max_moves = test_max_moves
#         self.frame_skip = frame_skip
#         self.stacked_observations = stacked_observations
#         self.episode_life = episode_life
#
#         # env
#         self.seed = None
#         self.env_name = None
#         self.action_space_size = None
#         self.image_channel = 3
#         self.image_based = image_based
#         self.obs_shape = None
#         self.gray_scale = gray_scale
#         self.cvt_string = cvt_string
#
#     def set_game(self, env_name):
#
#         self.env_name = env_name
#         # gray scale
#         if self.gray_scale:
#             self.image_channel = 1
#         obs_shape = (self.image_channel, 96, 96)
#         self.obs_shape = (obs_shape[0] * self.stacked_observations, obs_shape[1], obs_shape[2])
#         self.action_space_size = self.new_game().action_space_size
#
#     def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False,
#                  final_test=False):
#         """
#         make_atari: random noop_init + sticky_actions & max frame + time limits
#         WarpFrame: crop to 84x84
#         EpisodicLifeEnv: One life -> one episode  (default = True)
#         Monitor: record the video. But this can only run in the terminal not console
#         AtariWrapper: it create game object from env. with:
#             accessibility to legal actions
#             observation to str to save memory
#         """
#
#         if test:
#             if final_test:
#                 max_moves = 108000 // self.frame_skip
#             else:
#                 max_moves = self.test_max_moves
#             env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=max_moves)
#         else:
#             env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=self.max_moves)
#
#         if self.episode_life and not test:
#             env = EpisodicLifeEnv(env)
#         env = WarpFrame(env, width=self.obs_shape[1], height=self.obs_shape[2], grayscale=self.gray_scale)
#
#         if seed is not None:
#             env.seed(seed)
#
#         if save_video:
#             from gym.wrappers import Monitor
#             env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
#         return AtariGame(env, env.action_space.n, cvt_string=self.cvt_string)
#
#
# if __name__ == '__main__':
#
#     config = BaseConfig()
#     config.set_game('BreakoutNoFrameskip-v4')
#     envs = [config.new_game(), config.new_game()]
#     init_obs = [env.reset() for env in envs]
#
#
#
#
