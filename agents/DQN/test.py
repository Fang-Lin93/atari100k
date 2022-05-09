import os
import random

import torch

import numpy as np

from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
from core.utils import prepare_observation_lst
from core.history import GameHistory


def test(model, config, counter, test_episodes, device, render, save_video=False, final_test=False, use_pb=False):
    """evaluation test
    Parameters
    ----------
    model: any
        common for evaluation
    config: core.config.BaseAtariConfig
        set the game information, such as max moves and the pre-processing of image inputs.
    counter: int
        current training step counter
    test_episodes: int
        number of test episodes
    device: str
        'cuda' or 'cpu'
    render: bool
        True -> render the image during evaluation
    save_video: bool
        True -> save the videos during evaluation
    final_test: bool
        True -> this test is the final test, and the max moves would be 108k/skip
    use_pb: bool
        True -> use tqdm bars
    """

    model.to(device)
    model.eval()
    save_path = os.path.join(config.exp_path, 'recordings', 'step_{}'.format(counter))

    if use_pb:
        pb = tqdm(np.arange(config.test_max_moves), leave=True)

    with torch.no_grad():
        # new games
        envs = [config.new_game(seed=i, test=True, save_video=save_video, save_path=save_path, uid=i,
                                final_test=final_test) for i in range(test_episodes)]
        # initializations
        init_obses = [env.reset() for env in envs]
        dones = np.array([False for _ in range(test_episodes)])
        game_histories = [GameHistory(num_stack_obs=config.num_stack_obs) for _ in range(test_episodes)]

        for i in range(test_episodes):
            game_histories[i].init(init_obses[i])

        step = 0
        ep_ori_rewards = np.zeros(test_episodes)  # original
        ep_clip_rewards = np.zeros(test_episodes)
        # loop
        while not dones.all():
            if render:
                for i in range(test_episodes):
                    envs[i].render()

            if config.image_based:
                stack_obs = []
                for game_history in game_histories:
                    stack_obs.append(game_history.step_obs())
                stack_obs = prepare_observation_lst(stack_obs)
                stack_obs = torch.from_numpy(stack_obs).to(device).float() / 255.0
            else:
                stack_obs = [game_history.step_obs() for game_history in game_histories]
                stack_obs = torch.from_numpy(np.array(stack_obs)).to(device)

            with autocast():
                q_values = model(stack_obs.float())
            actions = q_values.argmax(dim=1)

            for i in range(test_episodes):
                if dones[i]:
                    continue

                action = actions[i]  # greedy

                obs, ori_reward, done, info = envs[i].step(action)

                if config.clip_reward:
                    clip_reward = np.sign(ori_reward)
                else:
                    clip_reward = ori_reward

                game_histories[i].add(action, obs, clip_reward)

                dones[i] = done
                ep_ori_rewards[i] += ori_reward
                ep_clip_rewards[i] += clip_reward

            step += 1
            if use_pb:
                pb.set_description('{} In step {}, scores: {}(max: {}, min: {}) currently.'
                                   ''.format(config.env_name, counter,
                                             ep_ori_rewards.mean(), max(ep_ori_rewards), min(ep_ori_rewards)))
                pb.update(1)

        for env in envs:
            env.close()

    return ep_ori_rewards, save_path


if __name__ == '__main__':
    import torch
    from agents.DQN.dqn import DQNet
    from core.config import BaseAtariConfig

    config = BaseAtariConfig(test_max_moves=50)  # it controls the test max move
    config.set_game('BreakoutNoFrameskip-v4')

    # obs_shape, action_dim, out_mlp_hidden_dim, num_blocks, res_out_channels
    m = DQNet((3*4, 96, 96), config.action_space_size, 32, 2, 64)

    rs, ps = test(
        model=m,
        config=config,
        counter=1,
        test_episodes=3,
        device='cpu',
        render=True,
        save_video=False,  # True for final test to save videos
        final_test=False,
        use_pb=True
    )
