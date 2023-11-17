import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.util.normalization import denormalize

import matplotlib.pyplot as plt

def random_by_bounds(n_var, xl, xu, n_samples=1):
    val = np.random.random((n_samples, n_var))
    return denormalize(val, xl, xu)


def random(problem, n_samples=1):
    return random_by_bounds(problem.n_var, problem.xl, problem.xu, n_samples=n_samples)


class FloatRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        return random(problem, n_samples=n_samples)


class Problem:
    def __init__(self):
        self.n_var = 3
        self.xl = [1,2,3]
        self.xu = [5,10,15]

p = Problem()
s = FloatRandomSampling()
res = s.do(p, 100)

for ind in res:
    print(ind.X)

plt.scatter([ind.X[2] for ind in res], [ind.X[1] for ind in res])
plt.show()