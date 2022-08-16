import time
import sys
import random

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import ray

from loguru import logger

logger.remove()
logger.add(sys.stderr, level='INFO')


def softmax(p, alpha):
    prob = [i ** alpha for i in p]
    norm = sum(prob)
    return [_ / norm for _ in prob]


class BlindCliffWalk(object):
    """
    optimal actions: 0 -> (0) -> 1 -> (1) -> 2 -> (0) -> 3 -> (1) -> ...
    next_v = 0 if ns < 0 else max(q_table[ns]) for terminate !
    odd state + odd action -> larger state, otherwise smaller state
    """

    def __init__(self, n: int, r: int = 1, gamma=None, mid_r_pos: int = 0):
        if gamma is None:
            gamma = 1 - 1 / n

        self.n = n  # number of states
        self.r = r  # the reward
        self.mid_r_pos = mid_r_pos  # the reward is given at the state self.n - 1 - self.mid_r_pos
        self.gamma = gamma
        self._flat_opt_q_tbl = self.get_flat_opt_q_tbl()
        # self._flat_q_tbl = [_ for i in self.opt_q_tbl for _ in i]
        self.lr = 1 / 4  # follow the PER

        # replaced by  self.flat_seq_rb
        # self.rb = []
        # self.rb_unique = []
        # # 2^n trajectories with transitions: [ s0x(2^n)/2 act=(0) ] + [ s0x(2^n)/2 act=(1) ] + (other states)...
        # for s_ in range(n):
        #     tran_0, tran_1 = (s_, 0) + self.step(s_, 0), (s_, 1) + self.step(s_, 1)
        #     self.rb += [tran_0] * (2 ** (n - 1 - s_))
        #     self.rb += [tran_1] * (2 ** (n - 1 - s_))
        #     self.rb_unique += [tran_0, tran_1]

        # record sequences [()]
        self.seq_rb, self.seq_trans_idx = [], []
        seq = []
        for s_ in range(n):
            seq_0 = seq + [(s_, 0) + self.step(s_, 0)]
            seq_1 = seq + [(s_, 1) + self.step(s_, 1)]
            if seq_0[-1][-1] == -1:  # remove the end episode
                self.seq_trans_idx += [(len(self.seq_rb) + seq_, step_)
                                       for seq_ in range(2 ** (n - 1 - s_)) for step_ in range(s_ + 1)]
                self.seq_rb += [seq_0] * 2 ** (n - 1 - s_)

                seq = seq_1
            if seq_1[-1][-1] == -1:
                self.seq_trans_idx += [(len(self.seq_rb) + seq_, step_)
                                       for seq_ in range(2 ** (n - 1 - s_)) for step_ in range(s_ + 1)]
                self.seq_rb += [seq_1] * 2 ** (n - 1 - s_)
                seq = seq_0

        self.flat_seq_rb = [j for i in self.seq_rb for j in i]
        self.num_trans = len(self.flat_seq_rb)
        self.rb_unique = set(self.flat_seq_rb)
        # self.rb_unique = list(set(self.replay_buffer))  # for oracle only, it removes frequencies

    def next_s(self, s, a):
        if s == self.n - 1 or s % 2 != a:
            return -1  # episode terminates
        return s + 1

    def get_flat_opt_q_tbl(self):
        """
        Q-table for the optimal policy!
        wrong action leads to immediate stop of an episode
        """
        q_tbl = []

        for s in range(self.n):
            val = [0, 0]
            val[s % 2] = self.gamma ** (self.n - s - 1)
            if self.mid_r_pos != 0 and s <= self.n - 1 - self.mid_r_pos:
                val[s % 2] += self.gamma ** (self.n - s - 1 - self.mid_r_pos)

            q_tbl += val
        return q_tbl

    def step(self, s, a):
        """
        :return: reward, next_s
        -1 is the terminal state
        """

        if s % 2 == a:
            # n_s = s + 1 if s < self.n - 1 else -1
            # r = self.r if s == self.n - 1 - self.mid_reward else 0
            # return r, n_s
            if s < self.n - 1:
                return self.r * (s == self.n - 1 - self.mid_r_pos), s + 1  # middle state reward
            return self.r, -1
        return 0, -1

    def is_convergent(self, q_tbl, threshold=1e-3):
        error = 0
        for s in range(self.n):
            error += (self._flat_opt_q_tbl[s][0] - q_tbl[s][0]) ** 2 + (self._flat_opt_q_tbl[s][1] - q_tbl[s][1]) ** 2
        return error / 2 / self.n < threshold

    @staticmethod
    def is_greedy_opt(q_tbl):
        for i, q in enumerate(q_tbl):
            sign = 1 if i % 2 == 0 else -1
            if sign * q[0] <= sign * q[1]:
                return False
        return True

    def q_tbl_mse(self, q_tbl):
        """
        MSE of all transitions
        """
        flat_q_tbl = [i for _ in q_tbl for i in _]
        sum_error = sum([(i - j) ** 2 for i, j in zip(flat_q_tbl, self._flat_opt_q_tbl)])
        return sum_error / self.n / 2

    def init_q_tbl(self):
        # return [[random.normalvariate(0, 1), random.normalvariate(0, 1)] for _ in range(self.n)]
        return [[0, 0] for _ in range(self.n)]
        # return [[random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)] for _ in range(self.n)]

    def td_update(self, q_table, i, is_weight):
        """
        :return: old TD-error, new TD-error
        """
        s, a, r, ns = self.flat_seq_rb[i]
        next_v = 0 if ns < 0 else max(q_table[ns])
        # importance sampling weigths
        old_td = r + self.gamma * next_v - q_table[s][a]
        q_table[s][a] += self.lr * old_td * is_weight
        # PER
        return abs(old_td), abs(r + self.gamma * next_v - q_table[s][a])

    def uniform(self, max_iter: int, interval=100):
        q_table = self.init_q_tbl()
        mse_s = []
        steps2opt, met_opt = 0, False
        for n_ in range(max_iter + 1):

            if n_ % interval == 0:
                mse_s.append(self.q_tbl_mse(q_table))

            s, a, r, ns = random.choice(self.flat_seq_rb)  # sample
            next_v = 0 if ns < 0 else max(q_table[ns])  # Bellman optimality equation
            q_table[s][a] += self.lr * (r + self.gamma * next_v - q_table[s][a])  # update

            if not self.is_greedy_opt(q_table) and not met_opt:
                steps2opt += 1
            else:
                met_opt = True
        return steps2opt, q_table, mse_s

    def per(self, max_iter: int, alpha=0.5, eps=0.0001, is_factor=0, init_p=None, interval=100, init_current_td=False):
        """
        prioritized exp. replay
        beta should be changed from 0
        """

        q_table = self.init_q_tbl()

        if init_p is None:
            init_p = 1

        if init_current_td:
            p = [abs(r + self.gamma * max(q_table[ns]) - q_table[s][a]) + eps if ns > -1 else abs(r - q_table[s][a])
                 for s, a, r, ns in self.flat_seq_rb]
            # p = [r+eps for _, _, r, _ in self.flat_seq_rb]
        else:
            # max
            p = [init_p] * len(self.flat_seq_rb)
        idx = range(len(p))
        mse_s = []

        steps2opt, met_opt = 0, False

        for n_ in range(max_iter + 1):

            if n_ % interval == 0:
                mse_s.append(self.q_tbl_mse(q_table))

            prob = softmax(p, alpha=alpha)
            i = random.choices(idx, weights=prob)[0]
            is_weight = 1 / (self.num_trans * prob[i]) ** is_factor

            _, new_td = self.td_update(q_table, i, is_weight)
            p[i] = new_td + eps

            if not self.is_greedy_opt(q_table) and not met_opt:
                steps2opt += 1
            else:
                met_opt = True

        return steps2opt, q_table, mse_s

    def pser(self, max_iter: int, alpha=0.5, eps=0.0001, is_factor=0, rho=0.4, init_p=None, interval=100,
             init_current_td=False):
        # init_current_td cannot be used for this simple question!
        """
        prioritized sequential exp. replay
        beta should be changed from 0
        """
        q_table = self.init_q_tbl()

        if init_p is None:
            init_p = 1
        # p = [init_p] * len(self.seq_trans_idx)
        # current TD
        if init_current_td:
            p = [abs(r + self.gamma * max(q_table[ns]) - q_table[s][a]) + eps if ns > -1 else abs(r - q_table[s][a])
                 for s, a, r, ns in self.flat_seq_rb]
            # p = [r+eps for _, _, r, _ in self.flat_seq_rb]
        else:
            # max
            p = [init_p] * len(self.flat_seq_rb)

        idx = range(len(p))

        # initialize priorities with decayed values: here it's not useful for all zero...
        for i in idx:
            seq_, loc_ = self.seq_trans_idx[i]
            for back_ in range(loc_):
                p[i - 1 - back_] = max(p[i] * rho ** (back_ + 1), p[i - 1 - back_])

        mse_s = []
        steps2opt, met_opt = 0, False

        for n_ in range(max_iter + 1):

            if n_ % interval == 0:
                mse_s.append(self.q_tbl_mse(q_table))

            # select transitions based on priorities
            prob = softmax(p, alpha=alpha)
            i = random.choices(idx, weights=prob)[0]  # from seq_trans_idx
            is_weight = 1 / (self.num_trans * prob[i]) ** is_factor

            old_td, new_td = self.td_update(q_table, i, is_weight)
            p[i] = new_td + eps

            # update priorities, decay along the sequence
            # p[i] = abs(r + self.gamma * next_v - q_table[s][a]) + eps
            seq_, loc_ = self.seq_trans_idx[i]
            for back_ in range(loc_):
                p[i - 1 - back_] = max((old_td + eps) * rho ** (back_ + 1), p[i - 1 - back_])

            if not self.is_greedy_opt(q_table) and not met_opt:
                steps2opt += 1
            else:
                met_opt = True

        return steps2opt, q_table, mse_s

    def pber(self, max_iter: int, alpha=0.5, eps=0.0001, is_factor=0, rho=0.4, back_factor=0.01, init_p=None,
             interval=100, sweep=False, seq_back_step: int = 1, fitted_number: int = 1, b_re_weight=False,
             init_current_td=False):
        # init_current_td cannot be used for this simple question!
        """
        prioritized backward exp. replay
        beta should be changed from 0
        """
        q_table = self.init_q_tbl()

        if init_p is None:
            init_p = 1  # > eps
        # fast backward decay gives better results
        backward_p = [back_factor ** (len(s_) - loc_ - 1) for s_ in self.seq_rb for loc_, _ in enumerate(s_)]

        # current TD
        if init_current_td:
            p = [p_ * abs(r + self.gamma * max(q_table[ns]) - q_table[s][a]) + eps if ns > -1 else abs(
                r - q_table[s][a])
                 for (s, a, r, ns), p_ in zip(self.flat_seq_rb, backward_p)]
        else:
            # max
            p = [init_p * _ for _ in backward_p]  # very important initialization!

        mse_s = []

        steps2opt, met_opt = 0, False

        for n_ in range(max_iter + 1):

            if n_ % interval == 0:
                mse_s.append(self.q_tbl_mse(q_table))
            #
            # # select transitions based on priorities
            prob = softmax(p, alpha=alpha)
            i = random.choices(range(len(p)), weights=prob)[0] if not sweep else p.index(max(p))  # from seq_trans_idx
            is_weight = 1 / (self.num_trans * prob[i]) ** is_factor

            # fitted Q
            old_td, new_td = self.td_update(q_table, i, is_weight)
            for _ in range(fitted_number - 1):
                old_, new_td = self.td_update(q_table, i, is_weight)
            # new_priority = new_td + eps

            # PBER TODO: add backward_p[i] can solely increase the performance
            # p[i] = new_td * backward_p[i]  # re-weight with backwards
            p[i] = new_td + eps
            if b_re_weight:
                p[i] = backward_p[i] * new_td + eps

            # PSER: back-propagate priorities to the previous transitions
            if seq_back_step > 0:
                seq_, loc_ = self.seq_trans_idx[i]  # (episode_id, trans_id)
                for back_ in range(loc_):
                    if back_ > seq_back_step:
                        break
                    if b_re_weight:
                        # p[i - 1 - back_] is already re-weighted
                        p[i - 1 - back_] = max(backward_p[i] * old_td * rho ** (back_ + 1) + eps, p[i - 1 - back_])
                    else:
                        # p[i - 1 - back_] = max((old_td + eps) * rho ** (back_ + 1), p[i - 1 - back_])
                        p[i - 1 - back_] = (old_td + eps) * rho ** (back_ + 1)

            if not self.is_greedy_opt(q_table) and not met_opt:
                steps2opt += 1
            else:
                met_opt = True

        return steps2opt, q_table, mse_s

    def beta_er(self, max_iter: int, alpha=0.5, eps=0.0001, is_factor=0, rho=0.4, back_factor=0.01, init_p=None,
                interval=100, sweep=False, seq_back_step: int = 1, fitted_number: int = 1, init_current_td=False):
        # init_current_td cannot be used for this simple question!
        """
        prioritized backward exp. replay
        beta should be changed from 0
        """
        q_table = self.init_q_tbl()

        if init_p is None:
            init_p = 1  # > eps
        # fast backward decay gives better results
        backward_p = [back_factor ** (len(s_) - loc_ - 1) for s_ in self.seq_rb for loc_, _ in enumerate(s_)]
        betas = backward_p[:]

        # current TD
        if init_current_td:
            p = [abs(r + self.gamma * max(q_table[ns]) - q_table[s][a]) if ns > -1 else abs(r - q_table[s][a])
                 for s, a, r, ns in self.flat_seq_rb]
            # p = [r+eps for _, _, r, _ in self.flat_seq_rb]
        else:
            # max
            p = [init_p] * len(self.flat_seq_rb)

        mse_s = []

        steps2opt, met_opt = 0, False

        for n_ in range(max_iter + 1):

            if n_ % interval == 0:
                mse_s.append(self.q_tbl_mse(q_table))
            #
            # # select transitions based on priorities
            # prob = softmax([i*j for i, j in zip(p, betas)], alpha=alpha)
            prob = softmax([i*j for i, j in zip(p, betas)], alpha=alpha)
            i = random.choices(range(len(p)), weights=prob)[0] if not sweep else p.index(max(p))  # from seq_trans_idx
            is_weight = 1 / (self.num_trans * prob[i]) ** is_factor

            # fitted Q
            old_td, new_td = self.td_update(q_table, i, is_weight)
            for _ in range(fitted_number - 1):
                old_, new_td = self.td_update(q_table, i, is_weight)
            # new_priority = new_td + eps

            # PBER TODO: add backward_p[i] can solely increase the performance
            # p[i] = new_td * backward_p[i]  # re-weight with backwards
            p[i] = new_td + eps

            # reset current transition's beta
            betas[i] = backward_p[i]
            # count number of updates

            # PSER: back-propagate priorities to the previous transitions
            if seq_back_step > 0:
                seq_, loc_ = self.seq_trans_idx[i]  # (episode_id, trans_id)
                for back_ in range(loc_):
                    if back_ > seq_back_step:
                        break
                    # back-propagate the priority (PSER)
                    # p[i - 1 - back_] = max(old_td * rho ** (back_ + 1) + eps, p[i - 1 - back_])
                    # update earlier transitions too early is not a good idea...
                    p[i - 1 - back_] = (old_td + eps) * rho ** (back_ + 1)
                    # back-propagate previous transitions' beta: 1, beta, beta^2, ...
                    # betas[i - 1 - back_] = 1
                    betas[i - 1 - back_] = max(back_factor ** back_, betas[i - 1 - back_])

            if not self.is_greedy_opt(q_table) and not met_opt:
                steps2opt += 1
            else:
                met_opt = True

        return steps2opt, q_table, mse_s

    def oracle(self, use_td=True):
        q_table = self.init_q_tbl()
        num_updates = 0

        while True:
            best = 0
            best_trans = None

            if use_td:
                for (s, a, r, ns) in self.rb_unique:
                    next_v = 0 if ns < 0 else max(q_table[ns])  # TODO
                    td = abs(r + self.gamma * next_v - q_table[s][a])
                    if td > best:
                        best_trans = (s, a, r, ns)
                        best = td
            else:
                for (s, a, r, ns) in self.rb_unique:  # largest error to the oracle??
                    error = abs(self._flat_opt_q_tbl[s][a] - q_table[s][a])
                    if error > best:
                        best_trans = (s, a, r, ns)
                        best = error

            (s, a, r, ns) = best_trans
            next_v = 0 if ns < 0 else max(q_table[ns])
            q_table[s][a] += self.lr * (r + self.gamma * next_v - q_table[s][a])  # update
            num_updates += 1
            if self.is_convergent(q_table):
                break

        return num_updates

    def uniform_FA(self):  # TODO
        """
        tabular + bias does not converge and unstable
        suppose feature = [state]xn + [action]x2 + [1] (const.), all zero means wrong action
        """
        theta_s = [random.normalvariate(0, 0.1) for _ in range(self.n)]
        theta_a = [random.normalvariate(0, 0.1) for _ in range(2)]
        bias = random.normalvariate(0, 0.1)
        num_updates = 0
        while True:
            s, a, r, ns = random.choice(self.flat_seq_rb)  # sample
            next_v = 0 if ns < 0 else theta_s[ns] + max(theta_a) + bias  # max only effect action
            current_v = theta_s[s] + theta_a[a] + bias
            delta = r + self.gamma * next_v - current_v
            theta_s[s] += self.lr * delta
            theta_a[a] += self.lr * delta
            bias += self.lr * delta
            num_updates += 1

            q_tbl = [[theta_s[s_] + theta_a[0] + bias, theta_s[s_] + theta_a[1] + bias] for s_ in range(self.n)]
            if self.is_convergent(q_tbl):  # TODO
                break

        return num_updates


ray.init()

n = 12  # should be odd
num_trials = 15
max_num = 20000
iter_interval = 100
init_p = None


@ray.remote
def run_training(config: dict, idx: int):
    game = BlindCliffWalk(n=n)
    s_t = time.time()
    n_opt_, _, mse_ = game.__getattribute__(config['func'])(**config['kwargs'])
    print(f'{config["desc"]}-{idx} Done! ({time.time() - s_t:.3f})s  opt={n_opt_}')
    return n_opt_, mse_, config["desc"]


configs = [
    {
        'desc': 'Uniform',
        'func': 'uniform',
        'kwargs': {'max_iter': max_num}
    },
    {
        'desc': 'PER',
        'func': 'per',
        'kwargs': {'max_iter': max_num, 'init_p': init_p}
    }, {
        'desc': 'PSER',
        'func': 'pser',
        'kwargs': {'max_iter': max_num, 'init_p': init_p}
    }, {
        'desc': 'PBER (back=1)',
        'func': 'pber',
        'kwargs': {'max_iter': max_num, 'init_p': init_p}
    }, {
        'desc': 'PBER w.o. propagation',
        'func': 'pber',
        'kwargs': {'max_iter': max_num, 'init_p': init_p, 'seq_back_step': 0},
    },
    {
        'desc': 'beta_ER',
        'func': 'beta_er',
        'kwargs': {'max_iter': max_num, 'init_p': init_p, 'back_factor': 0.1}
    },
    # {
    #     'desc': 'sweeping',
    #     'func': 'beta_er',
    #     'kwargs': {'max_iter': max_num, 'init_p': init_p, 'back_factor': 0.0001}
    # },
]

# TODO fitted_number: should be directed by the theory? so that the compounding error is small

futures = [run_training.remote(config_, i) for i in range(num_trials) for config_ in configs]

num_iter = list(range(0, max_num + 1, iter_interval))
x, mse, m = [], [], []
s_time = time.time()

steps_to_opt = []
for n_opt, res, desc in ray.get(futures):
    x += num_iter
    mse += res
    m += [desc] * len(num_iter)
    steps_to_opt.append([n_opt, desc])

# prepare data: MSE, number of steps to get the optimal policy
df = pd.DataFrame(columns=['num_iter', 'mse', 'method'])
df['num_iter'] = x
df['mse'] = mse
df['method'] = m
opt_df = pd.DataFrame(data=steps_to_opt, columns=['greedy_opt', 'method'])

fig, ax = plt.subplots(2, 1, figsize=(14, 16))
sns.lineplot(data=df, x="num_iter", y="mse", hue="method", ax=ax[0])
sns.kdeplot(data=opt_df, x="greedy_opt", hue="method", ax=ax[1])

# if sweeping is not random
# sweeping_opt_step = opt_df[opt_df['method'] == 'Backward sweeping']['greedy_opt'].mean()
# ax[1].scatter(x=sweeping_opt_step, y=0, s=200, marker='^', c='purple')
ax[1].set_xbound(0)

fig.show()

print('time used =', time.time() - s_time)
