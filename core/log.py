

import ray
import numpy as np
from loguru import logger


def log(config, step_count, log_data, model, replay_buffer,  shared_storage, summary_writer, vis_result=True):
    # step_count: trained steps
    loss_data, lr = log_data
    replay_trans_collected, replay_episodes_collected, replay_buffer_size, priorities, total_num, trans_collected, worker_logs = \
        ray.get([replay_buffer.transitions_collected.remote(),
                 replay_buffer.episodes_collected.remote(), replay_buffer.size.remote(),
                 replay_buffer.get_priorities.remote(), replay_buffer.get_total_len.remote(),
                 replay_buffer.transitions_collected.remote(),
                 shared_storage.get_worker_logs.remote()])

    running_replay_ratio = step_count/trans_collected

    worker_ori_reward, worker_reward, worker_reward_max, worker_eps_len, worker_eps_len_max, test_counter, test_dict, temperature, visit_entropy, priority_self_play, distributions = worker_logs

    if summary_writer is not None:
        tag = 'Train'
        if vis_result:
            summary_writer.add_histogram('{}_replay_data/replay_buffer_priorities'.format(tag),
                                         priorities,
                                         step_count)

        summary_writer.add_scalar('{}/episodes_collected'.format(tag), replay_episodes_collected, step_count)
        summary_writer.add_scalar('{}/transitions_collected'.format(tag), replay_trans_collected, step_count)
        summary_writer.add_scalar('{}/replay_buffer_len'.format(tag), replay_buffer_size, step_count)
        summary_writer.add_scalar('{}/total_node_num'.format(tag), total_num, step_count)
        summary_writer.add_scalar('{}/lr'.format(tag), lr, step_count)
        summary_writer.add_scalar('{}/td_loss'.format(tag), loss_data, step_count)
        summary_writer.add_scalar('{}/replay_ratio'.format(tag), running_replay_ratio, step_count)

        if worker_reward is not None:
            summary_writer.add_scalar('workers/ori_reward', worker_ori_reward, step_count)
            summary_writer.add_scalar('workers/clip_reward', worker_reward, step_count)
            summary_writer.add_scalar('workers/clip_reward_max', worker_reward_max, step_count)
            summary_writer.add_scalar('workers/eps_len', worker_eps_len, step_count)
            summary_writer.add_scalar('workers/eps_len_max', worker_eps_len_max, step_count)
            summary_writer.add_scalar('workers/temperature', temperature, step_count)
            summary_writer.add_scalar('workers/visit_entropy', visit_entropy, step_count)
            summary_writer.add_scalar('workers/priority_self_play', priority_self_play, step_count)
            for key, val in distributions.items():
                if len(val) == 0:
                    continue

                val = np.array(val).flatten()
                summary_writer.add_histogram('workers/{}'.format(key), val, step_count)

        if test_dict is not None:
            for key, val in test_dict.items():
                summary_writer.add_scalar('train/{}'.format(key), np.mean(val), test_counter)