import random
import numpy

"""
causal model
"""

def truth(x, u):
    return 5 * x ** 2 - 7 * x + 10 + 10 * u  # (a, b, c, d)


def gen_data(n=100):
    d = []
    for _ in range(n):
        # u = random.uniform(0, 1)
        u = random.randint(0, 5)
        x = random.normalvariate(0, 1)
        d.append((x, u, truth(x, u)))

    return d


class F(object):
    """
    for training y = f(x, u) with noise=u, it is fine to train with sampling u~Noise_Distri
    """

    def __init__(self):
        self.paras = {'a': random.random(),
                      'b': random.random(),
                      'c': random.random(),
                      'd': random.random(),
                      }

        self.data = gen_data()
        self.lr_e = 0.01
        self.lr_m = 0.01

        self.u = [0] * len(self.data)

    def E_step(self, i=100):
        for _ in range(i):
            for i, (x, _, y) in enumerate(self.data):
                self.u[i] = self.u[i] - self.lr_e * (self.eval(x, self.u[i]) - y) * self.paras['d']

            print("E loss", self.loss())

    def M_step(self, i=100):
        for _ in range(i):
            for i, (x, _, y) in enumerate(self.data):
                u = self.u[i]
                # u = random.uniform(0, 1)
                y_hat = self.eval(x, u)
                self.paras['a'] = self.paras['a'] - self.lr_m * (y_hat - y) * x ** 2
                self.paras['b'] = self.paras['b'] - self.lr_m * (y_hat - y) * x
                self.paras['c'] = self.paras['c'] - self.lr_m * (y_hat - y)
                # self.paras['d'] = self.paras['d'] - self.lr_m * (y_hat - y) * u

            print("M loss", self.loss())

    def eval(self, x, u):
        return self.paras['a'] * x ** 2 + self.paras['b'] * x + self.paras['c'] + self.paras['d'] * u

    def loss(self):
        return sum([(self.eval(x, u) - y) ** 2 for x, u, y in self.data]) / len(self.data)

    def train(self):
        for n in range(100):
            self.E_step()
            self.M_step()
            print(f"Step={n}: loss={self.loss()}")


self = F()
self.train()
