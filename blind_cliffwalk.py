
import time
import sys
import random
from loguru import logger

logger.remove()
logger.add(sys.stderr, level='INFO')


class BlindCliffWalk(object):
    """
    0 -> (0) -> 1 -> (1) -> 2 -> (0) ...
    next_v = 0 if ns < 0 else max(q_table[ns]) for terminate !
    odd state + odd action -> larger state, otherwise smaller state
    """

    def __init__(self, n: int, r: int = 1, gamma=None):
        if gamma is None:
            gamma = 1 - 1 / n

        self.n = n  # number of states
        self.r = r  # the reward
        self.gamma = gamma
        self.ground_truth = self.get_opt_ground_truth()
        self.lr = 1 / 4

        self.replay_buffer = []
        for s_ in range(n):
            self.replay_buffer += [(s_, 0) + self.step(s_, 0)] * (2 ** (n - 1 - s_))
            self.replay_buffer += [(s_, 1) + self.step(s_, 1)] * (2 ** (n - 1 - s_))

        self.rb_unique = list(set(self.replay_buffer))  # for oracle only, it removes frequencies

    def next_s(self, s, a):
        if s == self.n - 1 or s % 2 != a:
            return -1  # episode terminates
        return s + 1

    def get_opt_ground_truth(self):
        """
        Q-table for the optimal policy!
        wrong action leads to immediately stop of a episode
        """
        q_tbl = []

        for s in range(self.n):
            val = [0, 0]
            val[s % 2] = self.gamma**(self.n - s - 1)  # 'right' action
            q_tbl.append(val)
        return q_tbl

    def step(self, s, a):
        reward = 0

        if s == self.n - 1:
            next_s = -1
            if s % 2 == a:
                reward = self.r
        else:
            next_s = s + 1 if s % 2 == a else -1
        return reward, next_s

    def global_error(self, q_tbl):
        """
        MSE of all transitions
        """
        error, cnt = 0, 0
        for (s, a, r, ns) in self.replay_buffer:
            error += (r + self.gamma * max(q_tbl[ns]) - q_tbl[s][a]) ** 2
            cnt += 1

        return error

    def is_convergent(self, q_tbl):
        error = 0
        for s in range(self.n):
            error += (self.ground_truth[s][0] - q_tbl[s][0]) ** 2 + (self.ground_truth[s][1] - q_tbl[s][1]) ** 2
        return error/2/self.n < 1e-3

    def uniform_sampling(self):
        logger.info('<uniform sampling>...')
        s_time = time.time()
        q_table = [[random.normalvariate(0, 0.1), random.normalvariate(0, 0.1)] for _ in range(self.n)]
        num_updates = 0
        while True:
            s, a, r, ns = random.choice(self.replay_buffer)  # sample
            next_v = 0 if ns < 0 else max(q_table[ns])
            q_table[s][a] += self.lr * (r + self.gamma*next_v - q_table[s][a])  # update
            num_updates += 1

            if self.is_convergent(q_table):
                break

        logger.info(f'<uniform sampling finished !> n={self.n} time_used={time.time()-s_time:.3f}s num_updates={num_updates}')
        return num_updates

    def prop_sampling(self, alpha=10, eps=0.1, beta=0, greedy=False):
        """
        beta = 0 means gives bias, not IS-weighted
        """
        logger.info('<prop sampling>...')
        s_time = time.time()
        q_table = [[random.normalvariate(0, 0.1), random.normalvariate(0, 0.1)] for _ in range(self.n)]
        num_updates = 0
        p = [100]*len(self.rb_unique) # all transitions are initialized to have the max priority

        while True:
            (s, a, r, ns) = random.choices(self.replay_buffer)  # sample from trajectory, it's experience!
            next_v = 0 if ns < 0 else max(q_table[ns])
            # update priority !
            p[self.rb_unique.index((s, a, r, ns))] = abs(r + self.gamma * next_v - q_table[s][a]) + eps

            norm_p = sum([x ** alpha for x in p])
            prob = [x ** alpha / norm_p for x in p]

            if greedy:
                i = p.index(max(p))
            else:
                # sampling, if greedy, choose the largest
                i = random.choices(range(len(prob)), weights=prob)[0]   # sample from replay buffer

            (s, a, r, ns) = self.rb_unique[i]
            next_v = 0 if ns < 0 else max(q_table[ns])
            # update with IS
            q_table[s][a] += self.lr * (r + self.gamma * next_v - q_table[s][a])/(len(self.replay_buffer)/prob[i])**beta
            num_updates += 1
            #
            if self.is_convergent(q_table):
                break
        logger.info(f'<prop sampling finished> n={self.n} time_used={time.time()-s_time:.3f}s '
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
                    error = abs(self.ground_truth[s][a] - q_table[s][a])
                    if error > best:
                        best_trans = (s, a, r, ns)
                        best = error

            (s, a, r, ns) = best_trans
            next_v = 0 if ns < 0 else max(q_table[ns])
            q_table[s][a] += self.lr * (r + self.gamma * next_v- q_table[s][a])  # update
            num_updates += 1
            if self.is_convergent(q_table):
                break
        logger.info(f'<oracle sampling finished!> global_TD={use_td}, n={self.n}> time_used={time.time()-s_time:.3f}s '
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
            s, a, r, ns = random.choice(self.replay_buffer)  # sample
            next_v = 0 if ns < 0 else theta_s[ns] + max(theta_a) + bias  # max only effect action
            current_v = theta_s[s] + theta_a[a] + bias
            delta = r + self.gamma*next_v - current_v
            theta_s[s] += self.lr * delta
            theta_a[a] += self.lr * delta
            bias += self.lr * delta
            num_updates += 1

            q_tbl = [[theta_s[s_]+theta_a[0]+bias, theta_s[s_]+theta_a[1]+bias] for s_ in range(self.n)]
            if self.is_convergent(q_tbl):  # TODO
                break

        return num_updates


def main():
    from statistics import median
    from matplotlib import pyplot as plt
    x, y_uni, y_uni_fa, y_oracle, y_prop = [], [], [], [], []
    for n in range(2, 14):
        print('n=', n)
        game = BlindCliffWalk(n=n)

        oracle = [game.oracle_sampling() for _ in range(5)]

        # do 5 times to get the average
        uni = [game.uniform_sampling() for _ in range(5)]
        # fa = [game.uniform_FA() for _ in range(3)]
        prop = [game.prop_sampling() for _ in range(5)]
        x.append(len(game.replay_buffer))
        y_uni.append(median(uni))        # y_uni.append(sum(uni)/len(uni))
        # y_uni_fa.append(sum(fa)/len(fa))
        y_oracle.append(median(oracle))
        y_prop.append(sum(prop)/len(prop))

    plt.plot(x, y_uni, label='uniform')
    plt.plot(x, y_oracle, label='oracle')
    # plt.plot(x, y_uni_fa, label='FA')
    plt.plot(x, y_prop, label='prop')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='both')
    plt.legend()
    plt.title('BCW')
    plt.show()


if __name__ == '__main__':
    self = BlindCliffWalk(n=2)
    num_updates = self.uniform_sampling()

