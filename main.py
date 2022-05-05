import time
from core.wrap import make_atari, EpisodicLifeEnv
# from agents.DQN.dqn import RandAgent


env_id_ = 'BreakoutNoFrameskip-v4'
env = make_atari(env_id_, skip=4, max_episode_steps=1000)
env = EpisodicLifeEnv(env)
obs = env.reset()
done = False
i = 0

while not done:
    env.render()
    i += 1
    obs, r, done, info = env.step(env.action_space.sample())
    print('reward=', r)
    time.sleep(0.1)

# env.close()

