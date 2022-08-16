
import numpy as np
import ray
from core.utils import str_to_arr


class GameHistory(object):
    """
    It's not replay buffer
    used to generate stacked observations for model inference,
    here I imitate the code in EfficientZero but remove MCTS needs
    The memorized traj may be used for replay buffer later.
    """
    def __init__(self, num_stack_obs, capacity=100000, cvt_string=False, gray_scale=False, initial_obs=None):
        self.num_stack_obs = num_stack_obs  # stack consecutive k images as a state
        self.capacity = capacity
        self.cvt_string = cvt_string  # whether store cvt_string in the buffer
        self.gray_scale = gray_scale

        self.obs_history = []
        self.actions = []
        self.greedy_actions = []
        self.rewards = []

        if initial_obs is not None:
            self.init(initial_obs)

    def init(self, initial_obs):
        self.obs_history = [initial_obs for _ in range(self.num_stack_obs)]

    def add(self, action, greedy_act, next_state, next_reward):
        """
        greedy action is used to compute Q-value
        """
        self.actions.append(action)
        self.greedy_actions.append(greedy_act)
        self.obs_history.append(next_state)
        self.rewards.append(next_reward)

    def step_obs(self):
        index = len(self.rewards)
        frames = self.obs_history[index:index + self.num_stack_obs]
        if self.cvt_string:
            frames = [str_to_arr(obs, self.gray_scale) for obs in frames]
        return frames

    def game_over(self):
        # post processing the data when a history block is full
        # obs_history should be sent into the ray memory. Otherwise, it will cost large amounts of time in copying obs.
        self.rewards = np.array(self.rewards)
        self.obs_history = ray.put(np.array(self.obs_history))
        self.actions = np.array(self.actions)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        # return stacked history!
        frames = ray.get(self.obs_history)[i:i+self.num_stack_obs]
        if self.cvt_string:
            return [str_to_arr(obs, self.gray_scale) for obs in frames]
        return frames


if __name__ == '__main__':
    import time
    from matplotlib import pyplot as plt
    from core.wrap import make_atari, EpisodicLifeEnv

    env_id_ = 'BreakoutNoFrameskip-v4'
    env = make_atari(env_id_, skip=4, max_episode_steps=1000)
    env = EpisodicLifeEnv(env)
    obs = env.reset()
    history = GameHistory(num_stack_obs=4)
    history.init(obs)
    done = False
    i_ = 0

    while not done:
        env.render()
        i_ += 1
        act = env.action_space.sample()
        obs, r, done, info = env.step(act)

        history.add(act, act, obs, r)
        print('reward=', r)
        time.sleep(0.1)

    plt.imshow(obs)


