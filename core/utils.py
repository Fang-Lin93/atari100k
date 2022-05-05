
import os
import cv2
import shutil
import numpy as np
import gym
from scipy.stats import entropy


def arr_to_str(arr):
    """To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function encodes the observation numpy arr to the jpeg strings
    """
    img_str = cv2.imencode('.jpg', arr)[1].tobytes()
    return img_str


def str_to_arr(s, gray_scale=False):
    """To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function decodes the observation numpy arr from the jpeg strings
    Parameters
    ----------
    s: string
        the inputs
    gray_scale: bool
        True -> the inputs observation is gray not RGB.
    """
    nparr = np.frombuffer(s, np.uint8)
    if gray_scale:
        arr = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        arr = np.expand_dims(arr, -1)
    else:
        arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return arr


def make_results_dir(exp_path, args):
    # make the result directory
    os.makedirs(exp_path, exist_ok=True)
    if args.opr == 'train' and os.path.exists(exp_path) and os.listdir(exp_path):
        if not args.force:
            raise FileExistsError('{} is not empty. Please use --force to overwrite it'.format(exp_path))
        else:
            print('Warning, path exists! Rewriting...')
            shutil.rmtree(exp_path)
            os.makedirs(exp_path)
    log_path = os.path.join(exp_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'model'), exist_ok=True)
    return exp_path, log_path


def select_action(visit_counts, temperature=1, deterministic=True):
    """select action from the root visit counts.
    Parameters
    ----------
    visit_counts:
         MCTS counts
    temperature: float
        the temperature for the distribution
    deterministic: bool
        True -> select the argmax
        False -> sample from the distribution
    """
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
    total_count = sum(action_probs)
    assert total_count > 0
    action_probs = [x / total_count for x in action_probs]
    if deterministic:
        action_pos = np.argmax([v for v in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    count_entropy = entropy(action_probs, base=2)
    return action_pos, count_entropy  # actions_pos = 0, 1, 2,... as in atari100k env


def prepare_observation_lst(observation_lst):
    """Prepare the observations to satisfy the input format of torch
    [B, S, W, H, C] -> [B, S x C, W, H]
    batch, stack num, width, height, channel

    Default reshaping order: [A, B-fast] <==> order = 'C'
    So [S, C, W, H] => [S x C, W, H] <==> stack Sx[C, W, H], Here channel changes faster than history stacks
    """
    # B, S, W, H, C
    observation_lst = np.array(observation_lst, dtype=np.uint8)
    observation_lst = np.moveaxis(observation_lst, -1, 2)  # move one axis to another place

    shape = observation_lst.shape
    observation_lst = observation_lst.reshape((shape[0], -1, shape[-2], shape[-1]))

    return observation_lst


if __name__ == '__main__':
    """
    [W, H, C] cannot be re-shaped to [C, W, H] to torch
    
    Check it by the first channel[0, :, :]
    
    The right way to do it is using np.moveaxis()
    
    """

    from core.wrap import make_atari
    from matplotlib import pyplot as plt

    env = make_atari('BreakoutNoFrameskip-v4')

    obs = env.reset()

    # (reshape.ravel() == obs.ravel()).all()
    # (move.ravel() != obs.ravel()).all()

    plt.imshow(obs)
    plt.show()

    # one channel
    plt.imshow(obs[:, :, 0])
    plt.show()

    # not a channel for reshape ! be careful not to use reshape in CNN...
    reshape = obs.reshape(3, 210, 160)  # reshape follows the ravel rank
    plt.imshow(reshape[0])
    plt.show()

    # is a channel
    move = np.moveaxis(obs, -1, 0)  # same as  np.transpose(obs, (2, 0, 1))
    plt.imshow(move[0])
    plt.show()

    string = arr_to_str(obs)
    re_img = str_to_arr(string)
    plt.imshow(re_img)
    plt.show()






