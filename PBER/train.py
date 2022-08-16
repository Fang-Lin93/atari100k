import os
import ray
import torch
import time
import torch.optim as optim
import numpy as np
from torch.nn import L1Loss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pber.dqn import DQNet
from pber.test import test_
from pber.config import BaseAtariConfig
from core.storage import SharedStorage, QueueStorage
from pber.workers import DataWorker, PushWorker
from pber.replaybuffer import ReplayBuffer
from core.log import log
from loguru import logger


def adjust_lr(config, optimizer, step_count):
    # adjust learning rate, step lr every lr_decay_steps
    if step_count < config.lr_warm_step:
        lr = config.lr_init * step_count / config.lr_warm_step
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = config.lr_init * config.lr_decay_rate ** ((step_count - config.lr_warm_step) // config.lr_decay_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def _train(model, target_model, replay_buffer,
           model_storage: SharedStorage,
           batch_queue: QueueStorage,
           config: BaseAtariConfig,
           summary_writer):
    gamma = config.discount_factor
    td_step = config.n_td_steps

    # central behavior model
    model = model.to(config.device)
    # central target model for evaluation
    # model_storage is used to save local model !
    target_model = target_model.to(config.device)

    optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
                          weight_decay=config.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)

    model.train()
    target_model.eval()

    # wait until collecting enough data to start
    while not (ray.get(replay_buffer.get_total_len.remote()) >= config.start_transitions):
        time.sleep(1)
        pass

    logger.info('Begin training...')

    model_storage.set_start_signal.remote()

    step_count = 0
    # Note: the interval of the current model and the target model is between x and 2x. (x = target_model_interval)
    # recent_weights is the param of the target model

    loss_log = {'td_loss': [], 'lr': []}

    # while loop
    while step_count < config.training_steps:
        # remove data if the replay buffer is full. (more data settings)

        # # remove if more data
        replay_buffer.remove_to_fit.remote()

        # if step_count % 100 == 0:
        #     # replay_buffer.clear_buffer.remote()
        #     replay_buffer.remove_to_fit.remote()

        # obtain a batch
        batch = batch_queue.pop()
        if batch is None:
            time.sleep(0.3)
            continue

        model_storage.incr_counter.remote()
        lr = adjust_lr(config, optimizer, step_count)

        # update model for self-play: send to the local model
        if step_count % config.checkpoint_interval == 0:
            model_storage.set_weights.remote(model.get_weights())

        # update target model for evaluation
        if step_count % config.target_model_interval == 0:
            target_model.set_weights(model.get_weights())

        # preprocessing the data
        # greedy_actions is from the policy network, but evaluated using target network (double DQN)
        obs, actions, n_step_reward, next_obs, greedy_actions, next_obs_pos_in_batch, \
        indices_lst, is_weights, make_time = batch
        td_target = torch.from_numpy(np.array(n_step_reward)).to(config.device)

        if len(next_obs_pos_in_batch) > 0:
            with torch.no_grad():
                bootstrapping = target_model(next_obs.to(config.device))
                bootstrapping = bootstrapping[range(len(next_obs_pos_in_batch)), greedy_actions]
                td_target[next_obs_pos_in_batch] += gamma ** td_step * bootstrapping

        pred_q = model(obs.to(config.device))[range(len(actions)), actions]

        # update new_priority TODO: try anticipated priority
        l1_dist = L1Loss(reduction='none')(pred_q, td_target)
        new_priority = l1_dist.data.cpu().numpy() + config.prioritized_replay_eps
        replay_buffer.update_priorities.remote(indices_lst, new_priority, make_time)

        optimizer.zero_grad()
        # loss = F.smooth_l1_loss(pred_q, td_target, reduction='none') # huber loss
        loss = 0.5 * l1_dist ** 2
        loss = (loss.view(-1) * torch.from_numpy(np.array(is_weights)).to(config.device)).mean()  # IS weighted
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        loss_log['lr'].append(lr)
        loss_log['td_loss'].append(loss.item())
        step_count += 1

        # save common
        if step_count % config.save_ckpt_interval == 0:
            model_path = os.path.join(config.model_path, 'model_{}.pth'.format(step_count))
            torch.save(model.state_dict(), model_path)

        logger.info(f'step_count={step_count}, td_loss={loss.item():.5f}, lr={lr:.5f}')
        if step_count % 1 == 0:
            # config, step_count, log_data, model, replay_buffer,  shared_storage, summary_writer, vis_result=True
            log(config, step_count, (loss.item(), lr), model, replay_buffer, model_storage, summary_writer)

    model_storage.set_weights.remote(model.get_weights())
    print('Training finished!')
    return model.get_weights()


def train():
    config = BaseAtariConfig(
        # file
        exp_path='results',
        model_path='models',
        save_ckpt_interval=10000,

        # Self-Play
        # TODO: change to strict 100k
        training_steps=100 * 1000,
        max_moves=105 * 1000,  # moves for play only, it works only if total_transitions not works
        total_transitions=100 * 1000,  # atari 100k
        test_max_moves=12000,
        episode_life=True,
        frame_skip=4,  # sticky actions...
        num_env=10,  # number of env. for each worker
        num_actors=1,
        start_transitions=8,

        # replay buffer
        use_priority=True,
        prioritized_replay_eps=1e-6,
        priority_prob_beta=0.4,
        prio_beta_warm_step=20000,
        rb_transition_size=50000,  # number of transitions permitted in replay buffer
        replay_ratio=0.1,

        # training
        checkpoint_interval=100,  # send common behavior model to local
        target_model_interval=200,  # update target model to local
        n_td_steps=5,  # >= 1
        batch_size=64,
        discount_factor=0.997,
        weight_decay=1e-4,
        momentum=0.9,
        max_grad_norm=5,

        # testing
        test_interval=100,  # test after trained ? times
        num_test_episodes=5,

        # learning rate
        lr_init=0.1,
        lr_warm_step=1000,
        lr_decay_rate=0.95,
        lr_decay_steps=1000,

        # env
        num_stack_obs=4,
        gray_scale=False,
        clip_reward=True,
        cvt_string=True,
        image_based=True,
        device='cpu',

        # game
        game_name='SpaceInvadersNoFrameskip-v4',
        # game_name='BreakoutNoFrameskip-v4',

        # model
        out_mlp_hidden_dim=32,
        num_blocks=2,
        res_out_channels=64,

        # PBER
        back_factor=0.1,
        back_step=1

    )  # it controls the test max move

    model = DQNet(config.obs_shape,
                  config.action_space_size,
                  config.out_mlp_hidden_dim,
                  config.num_blocks,
                  config.res_out_channels)

    target_model = DQNet(config.obs_shape,
                         config.action_space_size,
                         config.out_mlp_hidden_dim,
                         config.num_blocks,
                         config.res_out_channels)

    model_storage = SharedStorage.remote(model, target_model)
    model_storage.set_start_signal.remote()

    batch_queue = QueueStorage(15, 20)
    replay_buffer = ReplayBuffer.remote(config)

    data_workers = [DataWorker.remote(rank, replay_buffer, model_storage, config) for rank in
                    range(0, config.num_actors)]
    push_worker = PushWorker.remote(batch_queue, replay_buffer, model_storage, config)
    workers = [worker.run.remote() for worker in data_workers] + [push_worker.run.remote()]

    workers += [test_.remote(config, model_storage)]

    # ray.wait(workers)
    print('training...')

    summary_writer = SummaryWriter(config.exp_path, flush_secs=10)
    final_weights = _train(model, target_model, replay_buffer, model_storage, batch_queue, config, summary_writer)


if __name__ == '__main__':
    train()
