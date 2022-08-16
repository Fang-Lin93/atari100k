
import numpy as np
from core.wrap import make_atari, EpisodicLifeEnv, WarpFrame, AtariGame


class DiscreteSupport(object):
    """
    Distributional rewards.
    Then using classification loss rather than regression loss
    """

    def __init__(self, min_: int, max_: int, delta=1.):
        assert min_ < max_
        self.min = min_
        self.max = max_
        self.range = np.arange(min_, max_ + 1, delta)
        self.size = len(self.range)
        self.delta = delta


class BaseAtariConfig(object):
    """
    Basic Atari env. config for ANY algorithms
    This config keeps the basic observation setting
    """

    def __init__(self,
                 # file
                 exp_path: str = '',
                 model_path: str = 'models',
                 save_ckpt_interval=1000,

                 # self-play
                 training_steps: int = 100 * 1000,
                 max_moves: int = 108000,  # moves for play only, it works only if total_transitions not works
                 total_transitions: int = 100 * 1000,  # atari 100k
                 test_max_moves: int = 12000,
                 episode_life: bool = True,
                 frame_skip: int = 4,  # sticky actions...
                 num_env: int = 10,  # number of env. for each worker
                 num_actors: int = 1,
                 start_transitions=8,

                 # replay buffer
                 use_priority: bool = True,
                 prioritized_replay_eps=1e-6,
                 priority_prob_alpha=0.6,
                 priority_prob_beta=0.4,
                 prio_beta_warm_step=20000,  # when will beta be 1
                 rb_transition_size=1000,  # number of transitions permitted in replay buffer
                 replay_ratio=0.1,

                 # training
                 checkpoint_interval: int = 100,
                 target_model_interval: int = 200,
                 n_td_steps=5,  # >= 1
                 batch_size=256,
                 discount_factor=0.997,
                 weight_decay=1e-4,
                 momentum=0.9,
                 max_grad_norm=5,

                 # testing
                 test_interval: int = 10000,  # test after trained ? times
                 num_test_episodes=8,
                 save_test_video=False,

                 # learning rate
                 lr_init=0.01,
                 lr_warm_step=1000,
                 lr_decay_rate=0.1,
                 lr_decay_steps=100000,

                 # env
                 num_stack_obs: int = 4,
                 gray_scale: bool = False,
                 clip_reward: bool = True,
                 cvt_string: bool = False,
                 image_based: bool = True,
                 game_name: str = 'SpaceInvadersNoFrameskip-v4',
                 device='cpu',

                 # model
                 out_mlp_hidden_dim=32,
                 num_blocks=2,
                 res_out_channels=64,

                 # PBER
                 back_factor=0.1,
                 back_step=1
                 ):

        """Base Config for Wrapped Atari games
        Parameters
        ----------
        training_steps: int
            training steps while collecting data
        max_moves: int
            max number of moves for an episode
        test_max_moves: int
            max number of moves for an episode during testing (in training stage),
            set this small to make sure the game will end faster.
        gray_scale: bool
            True -> use gray image observation
        episode_life: bool
            True -> one life is treated as an episode
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
        image_based: bool
            True -> observation is image based
        frame_skip: int
            number of frame skip
        num_stack_obs: int
            number of frame stack
        """
        # file
        self.exp_path = exp_path
        self.model_path = model_path
        self.save_ckpt_interval = save_ckpt_interval

        # Self-Play
        self.training_steps = training_steps
        self.total_transitions = total_transitions  # Atari100k
        self.max_moves = max_moves
        self.test_max_moves = test_max_moves
        self.frame_skip = frame_skip
        self.num_stack_obs = num_stack_obs
        self.episode_life = episode_life
        self.num_env = num_env
        self.num_actors = num_actors
        self.start_transitions = start_transitions

        # replay buffer
        self.use_priority = use_priority
        self.prioritized_replay_eps = prioritized_replay_eps
        self.rb_transition_size = rb_transition_size
        self.priority_prob_alpha = priority_prob_alpha
        self.priority_prob_beta = priority_prob_beta
        self.prio_beta_warm_step = prio_beta_warm_step
        self.replay_ratio = replay_ratio

        # training
        self.checkpoint_interval = checkpoint_interval  # when to update the worker's weights as the shared
        self.target_model_interval = target_model_interval
        self.n_td_steps = n_td_steps  # number of steps for value bootstrap
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.max_grad_norm = max_grad_norm

        # testing
        self.test_interval = test_interval
        self.num_test_episodes = num_test_episodes
        self.save_test_video = save_test_video

        # learning rate
        self.lr_init = lr_init
        self.lr_warm_step = lr_warm_step
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

        # env
        self.seed = 0
        self.env_name = None
        self.action_space_size = None
        self.image_channel = 3
        self.image_based = image_based
        self.obs_shape = None
        self.gray_scale = gray_scale
        self.cvt_string = cvt_string
        self.clip_reward = clip_reward

        # device
        self.device = device

        # game
        self.game_name = game_name
        self.set_game(game_name)

        # model, see the dqn.py file
        self.out_mlp_hidden_dim = out_mlp_hidden_dim
        self.num_blocks = num_blocks
        self.res_out_channels = res_out_channels

        # PBER
        self.back_factor = back_factor
        self.back_step = back_step

    def set_game(self, env_name):

        self.env_name = env_name
        # gray scale
        if self.gray_scale:
            self.image_channel = 1
        obs_shape = (self.image_channel, 96, 96)
        self.obs_shape = (obs_shape[0] * self.num_stack_obs, obs_shape[1], obs_shape[2])
        self.action_space_size = self.new_game().action_space_size

    def new_game(self, seed=None, test=False, save_video=False, save_path=None, uid='', final_test=False):
        """
        make_atari = random noop_init + sticky_actions & max frame + time limits
        WarpFrame: crop to 96x96
        EpisodicLifeEnv: One life -> one episode  (default = True)
        Monitor: record the video. But this can only run in the terminal not console
        AtariWrapper: it create game object from env. with:
            accessibility to legal actions
            observation to str to save memory
        """

        if test:
            if final_test:
                max_moves = 108000 // self.frame_skip  # not 100000 (100k)...?
            else:
                max_moves = self.test_max_moves
            env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=max_moves)
        else:
            env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=self.max_moves)

        if self.episode_life and not test:
            env = EpisodicLifeEnv(env)
        env = WarpFrame(env, width=self.obs_shape[1], height=self.obs_shape[2], grayscale=self.gray_scale)

        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers.record_video import RecordVideo
            env = RecordVideo(env, video_folder=save_path, name_prefix=uid)

        return AtariGame(env, env.action_space.n, cvt_string=self.cvt_string)


if __name__ == '__main__':
    config = BaseAtariConfig()
    config.set_game('BreakoutNoFrameskip-v4')
    envs = [config.new_game(), config.new_game()]
    init_obs = [env.reset() for env in envs]
    from matplotlib import pyplot as plt

    plt.imshow(init_obs[0])


# import numpy as np
# from core.wrap import make_atari, EpisodicLifeEnv, WarpFrame, AtariGame
#
#
# class DiscreteSupport(object):
#     """
#     Distributional rewards.
#     Then using classification loss rather than regression loss
#     """
#
#     def __init__(self, min_: int, max_: int, delta=1.):
#         assert min_ < max_
#         self.min = min_
#         self.max = max_
#         self.range = np.arange(min_, max_ + 1, delta)
#         self.size = len(self.range)
#         self.delta = delta
#
#
# class AtariConfig(object):
#     """
#     Basic Atari env. config for ANY algorithms
#     This config keeps the basic observation setting
#     """
#
#     def __init__(self,
#                  # env
#                  num_stack_obs: int = 4,
#                  episode_life: bool = True,
#                  frame_skip: int = 4,  # sticky actions...
#                  gray_scale: bool = False,
#                  clip_reward: bool = True,
#                  cvt_string: bool = False,
#                  image_based: bool = True,
#                  game_name: str = 'SpaceInvadersNoFrameskip-v4',
#                  device='cpu',
#
#                  # number of moveds
#                  test_max_moves: int = 1000,
#                  train_max_moves: int = 100_000,
#                  ):
#         # env
#         self.seed = 0
#         self.env_name = None
#         self.action_space_size = None
#         self.image_channel = 3
#         self.image_based = image_based
#         self.obs_shape = None
#         self.gray_scale = gray_scale
#         self.cvt_string = cvt_string
#         self.clip_reward = clip_reward
#         self.episode_life = episode_life
#
#         # device
#         self.device = device
#
#         # obs
#         self.num_stack_obs = num_stack_obs
#         self.frame_skip = frame_skip
#         self.test_max_moves = test_max_moves
#         self.train_max_moves = train_max_moves
#
#         # game
#         self.game_name = game_name
#         self.set_game(game_name)
#
#     def set_game(self, env_name):
#
#         self.env_name = env_name
#         # gray scale
#         if self.gray_scale:
#             self.image_channel = 1
#         obs_shape = (self.image_channel, 96, 96)
#         self.obs_shape = (obs_shape[0] * self.num_stack_obs, obs_shape[1], obs_shape[2])
#         self.action_space_size = self.new_game().action_space_size
#
#     def new_game(self, seed=None, test=False, save_video=False, save_path=None, uid='', final_test=False):
#         """
#         make_atari = random noop_init + sticky_actions & max frame + time limits
#         WarpFrame: crop to 96x96
#         EpisodicLifeEnv: One life -> one episode  (default = True)
#         Monitor: record the video. But this can only run in the terminal not console
#         AtariWrapper: it create game object from env. with:
#             accessibility to legal actions
#             observation to str to save memory
#         """
#
#         if test:
#             env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=self.test_max_moves)
#         else:
#             env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=self.train_max_moves)
#
#         if self.episode_life and not test:
#             env = EpisodicLifeEnv(env)
#         env = WarpFrame(env, width=self.obs_shape[1], height=self.obs_shape[2], grayscale=self.gray_scale)
#
#         if seed is not None:
#             env.seed(seed)
#
#         if save_video:
#             from gym.wrappers.record_video import RecordVideo
#             env = RecordVideo(env, video_folder=save_path, name_prefix=uid)
#
#         return AtariGame(env, env.action_space.n, cvt_string=self.cvt_string)
#
#
# if __name__ == '__main__':
#     from pber.storage import ReplayBuffer
#     from core.history import GameHistory
#
#     atari = AtariConfig()
#     atari.set_game('BreakoutNoFrameskip-v4')
#     env_nums = 2
#     envs = [atari.new_game() for _ in range(env_nums)]
#     init_obs = [env.reset() for env in envs]
#
#     games = [GameHistory(atari.num_stack_obs, initial_obs=init_obs[i], cvt_string=atari.cvt_string) for i in
#              range(env_nums)]
#     dones = [False for _ in range(env_nums)]
#
#     while not all(dones):
#         for i in range(env_nums):
#             if dones[i]:
#                 games[i].game_over()
#             else:
#                 env = envs[i]
#                 obs, ori_reward, done, info = env.step(1)
#                 # clip the reward
#                 # store data
#                 games[i].add(1, 2, obs, ori_reward)
#                 dones[i] = done
#
#     buffer_config = {'beta': 0.1,
#                      'back_step': 1,
#                      'batch_size': 10,
#                      'priority_prob_alpha': 0.4,
#                      'rb_transition_size': 1000}
#
#     rb = ReplayBuffer(buffer_config)
#
#     rb.save_pools(zip(games, [np.array([1]*len(games[0])), np.array([0.5]*len(games[1]))]))
#
#     t = rb.prepare_batch_context(5, 1)
#     from matplotlib import pyplot as plt
#
#     plt.imshow(init_obs[0])
