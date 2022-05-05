
import cv2
import numpy as np
import gym
import copy
import ray
from core.utils import arr_to_str, str_to_arr


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=96, height=96, grayscale=False, dict_space_key=None):

        """
        See https://github.com/YeWR/EfficientZero/blob/main/core/utils.py

        Warp frames to 96x96, not 84x84
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:  # TODO
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space  # it's a gym.space.Box
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        """
        this is a bulit-in function for gym.ObservationWrapper, to change the observations
        """
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)  # frame = (W, H) -> (W, H, 1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    """
    gym.Wrapper is used to create customized env using original envs

    rewrite 'reset' and 'step'

    """
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.

        It will return done=True once a life is lost
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        Here reset means you use your new life to play the game;
        NOT truly new game.

        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class NoopResetEnv(gym.Wrapper):
    """
    rewrite 'reset'
    """
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0. (act=0 for several times)
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)  # if dead, just go back to the normal reset
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    """
     rewrite 'step', 'render'
    """
    def __init__(self, env, skip=4):
        """
        Return only every `skip`-th frame

        sticky actions for 'skip' times;
        sum the reward inside this time;
        use maximum frame of the last two frames as observation
        """

        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        # (2,) + (210, 160, 3) = (2, 210, 160, 3), 2 refers to max (last 2 frames)
        self.viewer = None
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done, info = None, None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:  # the frame before the last frame
                self._obs_buffer[0] = obs
            if i == self._skip - 1:  # last frame
                self._obs_buffer[1] = obs
            total_reward += reward  # summed up all rewards inside those frames
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        # self.max_frame = self._obs_buffer.max(axis=0)  # ??
        self.max_frame = np.max(self._obs_buffer, axis=0)  # why do they use maximum frame of the last two?

        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        """
        here is how to re-shape the image and view the new image in the game screen

        Here the resize is only for visualization ! not for computation !
        """
        img = self.max_frame
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA).astype(np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


class TimeLimit(gym.Wrapper):
    """
    set time limit to the env given by max number of (sticky & true) actions
    """
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


def make_atari(env_id, skip=4, max_episode_steps=None):
    """Make Atari games
    Parameters
    ----------
    env_id: str
        name of environment
    skip: int
        frame skip
    max_episode_steps: int
        max moves for an episode
    """
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id  # require full version
    env = NoopResetEnv(env, noop_max=30)  # random no-op at the very beginning
    env = MaxAndSkipEnv(env, skip=skip)  # sticky actions & max_frame
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)  # count & truncate interactions
    return env


class AtariGame(object):
    """
    - give access to legal actions
    - modify observations: use np.uint8 or string
    """

    def __init__(self, env, action_space_size: int, config=None, cvt_string=True):
        """Atari Wrapper Parameters
        ----------
        env: Any
            another env wrapper
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
            using cv2.img
        """
        self.env = env
        self.action_space_size = action_space_size
        self.config = config
        self.cvt_string = cvt_string

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation

    def close(self):
        self.env.close()

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)


class GameHistory:
    """
    A block of game history from a full trajectories.
    The horizons of Atari games are quite large.
    Split the whole trajectory into several history blocks.
    """
    def __init__(self, action_space, max_length=200, config=None):
        """
        Parameters
        ----------
        action_space: int
            action space
        max_length: int
            max transition number of the history block
        """
        self.action_space = action_space
        self.max_length = max_length
        self.config = config

        self.stacked_observations = config.num_stack_obs
        self.discount = config.discount
        self.action_space_size = config.action_space_size
        self.zero_obs_shape = (config.obs_shape[-2], config.obs_shape[-1], config.image_channel)

        self.child_visits = []
        self.root_values = []

        self.actions = []
        self.obs_history = []
        self.rewards = []

    def init(self, init_observations):
        """Initialize a history block, stack the previous stacked_observations frames.
        Parameters
        ----------
        init_observations: list
            list of the stack observations in the previous time steps
        """
        self.child_visits = []
        self.root_values = []

        self.actions = []
        self.obs_history = []
        self.rewards = []
        self.target_values = []
        self.target_rewards = []
        self.target_policies = []

        assert len(init_observations) == self.stacked_observations

        for observation in init_observations:
            self.obs_history.append(copy.deepcopy(observation))

    def pad_over(self, next_block_observations, next_block_rewards, next_block_root_values, next_block_child_visits):
        """To make sure the correction of value targets, we need to add (o_t, r_t, etc) from the next history block
        , which is necessary for the bootstrapped values at the end states of this history block.
        Eg: len = 100; target value v_100 = r_100 + gamma^1 r_101 + ... + gamma^4 r_104 + gamma^5 v_105,
            but r_101, r_102, ... are from the next history block.
        Parameters
        ----------
        next_block_observations: list
            o_t from the next history block
        next_block_rewards: list
            r_t from the next history block
        next_block_root_values: list
            root values of MCTS from the next history block
        next_block_child_visits: list
            root visit count distributions of MCTS from the next history block
        """
        assert len(next_block_observations) <= self.config.num_unroll_steps
        assert len(next_block_child_visits) <= self.config.num_unroll_steps
        assert len(next_block_root_values) <= self.config.num_unroll_steps + self.config.td_steps
        assert len(next_block_rewards) <= self.config.num_unroll_steps + self.config.td_steps - 1

        # notice: next block observation should start from (stacked_observation - 1) in next trajectory
        for observation in next_block_observations:
            self.obs_history.append(copy.deepcopy(observation))

        for reward in next_block_rewards:
            self.rewards.append(reward)

        for value in next_block_root_values:
            self.root_values.append(value)

        for child_visits in next_block_child_visits:
            self.child_visits.append(child_visits)

    def is_full(self):
        # history block is full
        return self.__len__() >= self.max_length

    def legal_actions(self):
        return [_ for _ in range(self.action_space.n)]

    def append(self, action, obs, reward):
        # append a transition tuple
        self.actions.append(action)
        self.obs_history.append(obs)
        self.rewards.append(reward)

    def obs(self, i, extra_len=0, padding=False):
        """To obtain an observation of correct format: o[t, t + stack frames + extra len]
        Parameters
        ----------
        i: int
            time step i
        extra_len: int
            extra len of the obs frames
        padding: bool
            True -> padding frames if (t + stack frames) are out of trajectory
        """
        frames = ray.get(self.obs_history)[i:i + self.stacked_observations + extra_len]
        if padding:
            pad_len = self.stacked_observations + extra_len - len(frames)
            if pad_len > 0:
                pad_frames = [frames[-1] for _ in range(pad_len)]
                frames = np.concatenate((frames, pad_frames))
        if self.config.cvt_string:
            frames = [str_to_arr(obs, self.config.gray_scale) for obs in frames]
        return frames

    def zero_obs(self):
        # return a zero frame
        return [np.zeros(self.zero_obs_shape, dtype=np.uint8) for _ in range(self.stacked_observations)]

    def step_obs(self):
        # return an observation of correct format for model inference
        index = len(self.rewards)
        frames = self.obs_history[index:index + self.stacked_observations]
        if self.config.cvt_string:
            frames = [str_to_arr(obs, self.config.gray_scale) for obs in frames]
        return frames

    def get_targets(self, i):
        # return the value/rewrad/policy targets at step i
        return self.target_values[i], self.target_rewards[i], self.target_policies[i]

    def game_over(self):
        # post processing the data when a history block is full
        # obs_history should be sent into the ray memory. Otherwise, it will cost large amounts of time in copying obs.
        self.rewards = np.array(self.rewards)
        self.obs_history = ray.put(np.array(self.obs_history))
        self.actions = np.array(self.actions)
        self.child_visits = np.array(self.child_visits)
        self.root_values = np.array(self.root_values)

    def store_search_stats(self, visit_counts, root_value, idx: int = None):
        # store the visit count distributions and value of the root node after MCTS
        sum_visits = sum(visit_counts)
        if idx is None:
            self.child_visits.append([visit_count / sum_visits for visit_count in visit_counts])
            self.root_values.append(root_value)
        else:
            self.child_visits[idx] = [visit_count / sum_visits for visit_count in visit_counts]
            self.root_values[idx] = root_value

    def __len__(self):
        return len(self.actions)
