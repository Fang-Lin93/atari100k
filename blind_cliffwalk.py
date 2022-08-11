import time
import sys
import random
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

    def __init__(self, n: int, r: int = 1, gamma=None):
        if gamma is None:
            gamma = 1 - 1 / n

        self.n = n  # number of states
        self.r = r  # the reward
        self.gamma = gamma
        self._flat_opt_q_tbl = self.get_flat_opt_q_tbl()
        # self._flat_q_tbl = [_ for i in self.opt_q_tbl for _ in i]
        self.lr = 1 / 4  # follow the PER

        self.rb = []
        self.rb_unique = []
        # 2^n trajectories with transitions: [ s0x(2^n)/2 act=(0) ] + [ s0x(2^n)/2 act=(1) ] + (other states)...
        for s_ in range(n):
            tran_0, tran_1 = (s_, 0) + self.step(s_, 0), (s_, 1) + self.step(s_, 1)
            self.rb += [tran_0] * (2 ** (n - 1 - s_))
            self.rb += [tran_1] * (2 ** (n - 1 - s_))
            self.rb_unique += [tran_0, tran_1]

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

        self.num_trans = len(self.rb)
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
            val[s % 2] = self.gamma ** (self.n - s - 1)  # 'right' action
            # if s <= (self.n-1)/2:
            #     val[s % 2] += self.gamma ** ((self.n-1)/2 - s)
            # q_tbl.append(val)
            q_tbl += val
        return q_tbl

    def step(self, s, a):
        """
        :return: reward, next_s
        -1 is the terminal state
        """

        if s % 2 == a:
            if s < self.n - 1:
                return 0, s + 1  # middle state?
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
            sign = 1 if i%2 == 0 else -1
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
        # return [[random.normalvariate(0, 0.1), random.normalvariate(0, 0.1)] for _ in range(self.n)]
        return [[0, 0] for _ in range(self.n)]

    def uniform(self, max_iter: int, interval=100):
        q_table = self.init_q_tbl()
        mse_s = []
        num_opt = 0
        for n_ in range(max_iter + 1):

            if n_ % interval == 0:
                mse_s.append(self.q_tbl_mse(q_table))

            s, a, r, ns = random.choice(self.rb)  # sample
            next_v = 0 if ns < 0 else max(q_table[ns])  # Bellman optimality equation
            q_table[s][a] += self.lr * (r + self.gamma * next_v - q_table[s][a])  # update

            if not self.is_greedy_opt(q_table):
                num_opt += 1

        return num_opt, q_table, mse_s

    def per(self, max_iter: int, alpha=0.5, eps=0.0001, beta=0, init_p=None, interval=100):
        """
        prioritized exp. replay
        beta should be changed from 0
        """
        q_table = self.init_q_tbl()

        if init_p is None:
            init_p = eps
        p = [init_p] * len(self.rb)
        idx = range(len(p))
        mse_s = []

        num_opt = 0

        for n_ in range(max_iter + 1):

            if n_ % interval == 0:
                mse_s.append(self.q_tbl_mse(q_table))

            prob = softmax(p, alpha=alpha)
            i = random.choices(idx, weights=prob)[0]
            s, a, r, ns = self.rb[i]
            next_v = 0 if ns < 0 else max(q_table[ns])

            # td = r + self.gamma * next_v - q_table[s][a]

            is_weight = 1 / (self.num_trans * prob[i]) ** beta
            q_table[s][a] += self.lr * (r + self.gamma * next_v - q_table[s][a]) * is_weight

            # update priorities using new td
            p[i] = abs(r + self.gamma * next_v - q_table[s][a]) + eps

            if not self.is_greedy_opt(q_table):
                num_opt += 1

        return num_opt, q_table, mse_s

    def pser(self, max_iter: int, alpha=0.5, eps=0.0001, beta=0, rho=0.4, init_p=None, interval=100):
        """
        prioritized sequential exp. replay
        beta should be changed from 0
        """
        q_table = self.init_q_tbl()

        if init_p is None:
            init_p = eps
        p = [init_p] * len(self.seq_trans_idx)
        idx = range(len(p))
        mse_s = []

        num_opt = 0

        for n_ in range(max_iter + 1):

            if n_ % interval == 0:
                mse_s.append(self.q_tbl_mse(q_table))

            # select transitions based on priorities
            prob = softmax(p, alpha=alpha)
            i = random.choices(idx, weights=prob)[0]  # from seq_trans_idx
            s, a, r, ns = self.flat_seq_rb[i]
            next_v = 0 if ns < 0 else max(q_table[ns])

            # td = r + self.gamma * next_v - q_table[s][a]

            # importance sampling weigths
            is_weight = 1 / (self.num_trans * prob[i]) ** beta
            q_table[s][a] += self.lr * (r + self.gamma * next_v - q_table[s][a]) * is_weight

            # update priorities, decay along the sequence
            p[i] = abs(r + self.gamma * next_v - q_table[s][a]) + eps
            seq_, loc_ = self.seq_trans_idx[i]
            for back_ in range(loc_):
                p[i - 1 - back_] = max(p[i] * rho ** (back_ + 1), p[i - 1 - back_])

            if not self.is_greedy_opt(q_table):
                num_opt += 1

        return num_opt, q_table, mse_s

    def pber(self, max_iter: int, alpha=0.5, eps=0.0001, beta=0, rho=0.4, init_p=None, interval=100, sweep=False):
        """
        prioritized backward exp. replay
        beta should be changed from 0
        """
        q_table = self.init_q_tbl()

        if init_p is None:
            init_p = eps
        # fast backward decay gives better results
        backward_p = [0.01 ** (len(s_) - loc_ - 1) for s_ in self.seq_rb for loc_, _ in enumerate(s_)]
        p = [init_p * _ for _ in backward_p]  # TODO: increase initial speed
        # p = [init_p] * len(self.seq_trans_idx)
        idx = range(len(p))
        mse_s = []

        num_opt = 0

        for n_ in range(max_iter + 1):

            if n_ % interval == 0:
                mse_s.append(self.q_tbl_mse(q_table))

            # select transitions based on priorities
            # sampling requires re-weight of priorities PBER
            prob = softmax(p, alpha=alpha)
            i = random.choices(idx, weights=prob)[0] if not sweep else p.index(max(p))  # from seq_trans_idx
            s, a, r, ns = self.flat_seq_rb[i]
            next_v = 0 if ns < 0 else max(q_table[ns])

            # td = r + self.gamma * next_v - q_table[s][a]

            # importance sampling weigths
            is_weight = 1 / (self.num_trans * prob[i]) ** beta
            q_table[s][a] += self.lr * (r + self.gamma * next_v - q_table[s][a]) * is_weight

            # update priorities, decay along the sequence
            # PER
            new_td = (abs(r + self.gamma * next_v - q_table[s][a]) + eps)

            # PBER TODO: add backward_p[i] can solely increase the performance
            p[i] = new_td * backward_p[i]  # re-weight with backwards

            # TODO: require this part?
            # PSER: back-propagate priorities to the previous transitions
            seq_, loc_ = self.seq_trans_idx[i]  # (episode_id, trans_id)
            for back_ in range(loc_):
                p[i - 1 - back_] = max(new_td*rho**(back_+1), p[i - 1 - back_])

            if not self.is_greedy_opt(q_table):
                num_opt += 1

        return num_opt, q_table, mse_s

    def prop_sampling(self, alpha=10, eps=0.1, beta=0, greedy=False):
        """
        beta = 0 means gives bias, not IS-weighted
        """
        logger.info('<prop sampling>...')
        s_time = time.time()
        q_table = [[random.normalvariate(0, 0.1), random.normalvariate(0, 0.1)] for _ in range(self.n)]
        num_updates = 0
        p = [100] * len(self.rb_unique)  # all transitions are initialized to have the max priority

        while True:
            (s, a, r, ns) = random.choice(self.rb)  # sample from trajectory, it's experience!
            next_v = 0 if ns < 0 else max(q_table[ns])
            # update priority !
            p[self.rb_unique.index((s, a, r, ns))] = abs(r + self.gamma * next_v - q_table[s][a]) + eps

            norm_p = sum([x ** alpha for x in p])
            prob = [x ** alpha / norm_p for x in p]

            if greedy:
                i = p.index(max(p))
            else:
                # sampling, if greedy, choose the largest
                i = random.choices(range(len(prob)), weights=prob)[0]  # sample from replay buffer

            (s, a, r, ns) = self.rb_unique[i]
            next_v = 0 if ns < 0 else max(q_table[ns])
            # update with IS
            q_table[s][a] += self.lr * (r + self.gamma * next_v - q_table[s][a]) / (len(self.rb) / prob[i]) ** beta
            num_updates += 1
            #
            if self.is_convergent(q_table):
                break
        logger.info(f'<prop sampling finished> n={self.n} time_used={time.time() - s_time:.3f}s '
                    f'num_updates={num_updates}')
        return num_updates

    def oracle_sampling(self, use_td=True):
        logger.info('<oracle sampling>...')
        s_time = time.time()
        q_table = [[random.normalvariate(0, 0.1), random.normalvariate(0, 0.1)] for _ in range(self.n)]
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
        logger.info(
            f'<oracle sampling finished!> global_TD={use_td}, n={self.n}> time_used={time.time() - s_time:.3f}s '
            f'num_updates={num_updates}')

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
            s, a, r, ns = random.choice(self.rb)  # sample
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


# def main():
#     from statistics import median
#     from matplotlib import pyplot as plt
#     x, y_uni, y_uni_fa, y_oracle, y_prop = [], [], [], [], []
#     for n in range(2, 14):
#         print('n=', n)
#         game = BlindCliffWalk(n=n)
#
#         oracle = [game.oracle_sampling() for _ in range(5)]
#
#         # do 5 times to get the average
#         uni = [game.uniform_sampling() for _ in range(5)]
#         # fa = [game.uniform_FA() for _ in range(3)]
#         prop = [game.prop_sampling() for _ in range(5)]
#         x.append(len(game.rb))
#         y_uni.append(median(uni))  # y_uni.append(sum(uni)/len(uni))
#         # y_uni_fa.append(sum(fa)/len(fa))
#         y_oracle.append(median(oracle))
#         y_prop.append(sum(prop) / len(prop))
#
#     plt.plot(x, y_uni, label='uniform')
#     plt.plot(x, y_oracle, label='oracle')
#     # plt.plot(x, y_uni_fa, label='FA')
#     plt.plot(x, y_prop, label='prop')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.grid(which='both')
#     plt.legend()
#     plt.title('BCW')
#     plt.show()


#
# n = 12
# num_trials = 10
# max_num = 20000
# iter_interval = 100
# game = BlindCliffWalk(n=n)
#
# num_iter = list(range(0, max_num + 1, iter_interval))  # 0, 100, 200, ..., 2001
# x, mse, m = [], [], []
#
# for i_ in trange(num_trials):
#     _, uni_mse = game.uniform(max_iter=max_num)
#     _, per_mse = game.per(max_iter=max_num, beta=0, init_p=1)
#     _, pser_mse = game.pser(max_iter=max_num, beta=0, init_p=1)
#     _, pber_mse = game.pber(max_iter=max_num, beta=0, init_p=1)
#
#     x += num_iter
#     mse += uni_mse
#     m += ['Uniform'] * len(num_iter)
#
#     x += num_iter
#     mse += per_mse
#     m += ['PER'] * len(num_iter)
#
#     x += num_iter
#     mse += pser_mse
#     m += ['PSER'] * len(num_iter)
#
#     x += num_iter
#     mse += pber_mse
#     m += ['PBER'] * len(num_iter)
#
# df = pd.DataFrame(columns=['num_iter', 'mse', 'method'])
# df['num_iter'] = x
# df['mse'] = mse
# df['method'] = m
# fig, ax = plt.subplots(1, 1, figsize=(14, 10))
# sns.lineplot(data=df, x="num_iter", y="mse", hue="method", ax=ax)
# plt.show()
#
#


ray.init()

n = 12  # should be odd
num_trials = 20
max_num = 20000
iter_interval = 100


@ray.remote
def run_training(func, desc: str, idx: int, **kwargs):
    game = BlindCliffWalk(n=n)
    s_t = time.time()
    n_opt_, _, mse_ = game.__getattribute__(func)(**kwargs)
    print(f'{desc}-{idx} Done! ({time.time() - s_t:.3f})s  opt={n_opt_}')
    return n_opt_, mse_, desc


funcs = ['uniform', 'per', 'pser', 'pber']
decs = ['Uniform', 'PER', 'PSER', 'PBER']
kwargs = [{'max_iter': max_num},
          {'max_iter': max_num, 'beta': 0, 'init_p': 1},
          {'max_iter': max_num, 'beta': 0, 'init_p': 1},
          {'max_iter': max_num, 'beta': 0, 'init_p': 1}]
futures = [run_training.remote(f, desc, i, **args) for i in range(num_trials) for f, desc, args in
           zip(funcs, decs, kwargs)]
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

fig, ax = plt.subplots(2, 1, figsize=(14, 12))
sns.lineplot(data=df, x="num_iter", y="mse", hue="method", ax=ax[0])
sns.kdeplot(data=opt_df, x="greedy_opt", hue="method", ax=ax[1])
fig.show()

print('time used =', time.time() - s_time)
# print(steps_to_opt)

#
#
# if __name__ == '__main__':
#     self = BlindCliffWalk(n=10)
#     tbl, mse = self.uniform(100)
