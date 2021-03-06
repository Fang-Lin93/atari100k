
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


