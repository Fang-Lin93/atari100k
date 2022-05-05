
from core.utils import str_to_arr


class GameHistory(object):
    """
    used to generate stacked observations for model inference,
    here I imitate the code in EfficientZero but remove MCTS needs
    The memorized traj may be used for replay buffer later.
    """
    def __init__(self, num_stack_obs, capacity=100000, cvt_string=False, gray_scale=False):
        self.num_stack_obs = num_stack_obs
        self.capacity = capacity
        self.cvt_string = cvt_string
        self.gray_scale = gray_scale

        self.obs_history = []
        self.actions = []
        self.rewards = []

    def init(self, initial_obs):
        self.obs_history = [initial_obs for _ in range(self.num_stack_obs)]

    def add(self, action, next_state, next_reward):
        self.actions.append(action)
        self.obs_history.append(next_state)
        self.rewards.append(next_reward)

    def step_obs(self):
        index = len(self.rewards)
        frames = self.obs_history[index:index + self.num_stack_obs]
        if self.cvt_string:
            frames = [str_to_arr(obs, self.gray_scale) for obs in frames]
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
    i = 0

    while not done:
        env.render()
        i += 1
        act = env.action_space.sample()
        obs, r, done, info = env.step(act)

        history.add(act, obs, r)
        print('reward=', r)
        time.sleep(0.1)

    plt.imshow(obs)


