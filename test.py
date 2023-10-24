import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from typing import List
from collections import deque
import signal
import functions
import ndrank
from pymoo.optimize import minimize

CRITERIA = functions.criteriumList[0]

_POINT_MAX = 1e100

_DELAY = 0.01

infp = float('inf')
infn = float('-inf')

class Killer:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, lambda _, __: self.exitt())
        signal.signal(signal.SIGTERM, lambda _, __: self.exitt())
    def exitt(self):
        self.kill_now = True

class DES:
    """
    Parameters
    ----------
    objective_function: str
        Chosen from: quadratic, elliptic, bent, rastrigin, rosenbrock, ackley
    dimensions: int
        Objective function dimensionality.
    lambda_arg: int
        Population count. Must be > 3, if set to None, default value will be computed.
    stop_after: int
        How many iterations are to be run
    visuals: bool
        If True, every algorithm generation will be visualised (only 2 first dimensions)
    """
    def __init__(self, dimensions: int = 2, objective_count: int = 1, lambda_arg: int = None, stop_after: int = 50, visuals: bool = False):
        assert dimensions > 0, "Number of dimensions must be greater than 0"
        self._N = dimensions
        self._M = objective_count
        self._stop_after = stop_after
        self._visuals = visuals and self._N >= 2

        self._F = 1/np.sqrt(2)
        self._C = 4/(self._N + 4)
        self._H = 6 + 3*np.sqrt(self._N)

        self._count_eval = 0
        self._budget = infp

        # Initial point
        self._mean_m = None
        self._mean_s = None

        # Population size
        if lambda_arg == None:
            self._lambda = 4 * self._N # default population size
        else:
            self._lambda = lambda_arg
        assert self._lambda > 1, "Population size must be greater than 1"

        # Number of parents/points to be selected
        self._mu = self._lambda // 2

        # E||N(0, I)||
        self._chi = 1*np.sqrt(self._N) * (1 - 1 / (4 * self._N) + 1 / (21 * self._N ** 2))

        # noise intensity
        self._EPS = 10 ** (-8)/ self._chi
        # self._EPS = 1/ self._chi

        # Evolution paths
        self._path = np.zeros(self._N)
        self._c = 4/(self._N + 4)

        # Store current generation number
        self._generation = 0

        # Store important values at each generation
        self._results = []
        self._mean_history = []
        self._worst_fitness = infp

        self._populations = deque([])
        self._ndpoints = []

        # Run the algorithm
        self._killer = Killer()
        self._visuals_started = False
        self._generation_loop()

    def _generation_loop(self):
        assert self._results == [], "One algorithm instance can only run once."
        self._init_first_population(loc=10*np.ones(self._N), scale=1)
        for _ in range(self._stop_after):
            if self._killer.kill_now:
                exit()
            if self._visuals:
                if not self._visuals_started:
                    self._visuals_started = True
                    
                    plt.rcParams["figure.figsize"] = (16,9)
                    plt.rcParams['font.size'] = '22'
                    # plt.tight_layout()
                    self._fig, (self._ax1, self._ax2) = plt.subplots(1, 2)
                    self._fig.subplots_adjust(top = 0.8, bottom = 0.1, left = 0.1, right = 0.99)

                self._draw_features()
                self._draw_values()
                title = "Iteracja " + str(self._generation) + ", \n"
                title += "Liczebność populacji: " + str(self._lambda) + ", \n"
                title += "Wymiarowość: " + str(self._N) + ", \n"
                self._fig.suptitle(title)

                plt.pause(_DELAY)
            self._update()
            self._new_generation()

    def _update(self) -> None:
        assert len(self._populations) <= self._H, f"There should be no more than H populations saved in history"
        assert len(self._populations[0]) == self._mu, f"There should be exactly mu points in each saved population"
        assert len(self._populations[0][0]) == self._N, f"Each point should have exactly N dimensions"
        for point in self._populations[0]:
            assert np.all(
                np.abs(point) < _POINT_MAX
            ), f"Absolute value of all generated points must be less than {_POINT_MAX} to avoid overflow errors."

    def _init_first_population(self, loc: np.array = 0, scale: float = 1) -> None:
        self._last_population = []
        for _ in range(self._lambda):
            new = np.random.normal(loc = loc, scale = scale, size = self._N)
            values = self._evaluate(new)
            rank = infp
            self._last_population.append([new, values, rank])

        self._ndpoints = ndrank.getNonDominated(self._last_population)
        ndrank.calcRank(self._last_population)
        selected = np.array([x[0] for x in sorted(self._last_population, key=lambda x: x[2])][:self._mu]) ### implement NSGA sorting

        self._mean_m = np.mean(np.array([x[0] for x in self._last_population]), axis=0)
        self._mean_s = np.mean(selected, axis=0)

        self._populations.append(selected)

    def _new_generation(self) -> None:
        self._last_population = []
        for _ in range(self._lambda):
            new = self._sample_solution()
            values = self._evaluate(new)
            rank = infp
            self._last_population.append([new, values, rank])

        ndrank.calcRank(self._last_population)
        self._ndpoints = ndrank.getNonDominated(self._ndpoints.copy() + [point for point in self._last_population if point[2] == 0])
        selected = np.array([x[0] for x in sorted(self._last_population, key=lambda x: x[2])][:self._mu]) ### implement NSGA sorting

        self._mean_m = np.mean(np.array([x[0] for x in self._last_population]), axis=0)
        self._mean_s = np.mean(selected, axis=0)

        self._path = (1-self._c) * self._path + self._c * (self._mean_s - self._mean_m)

        self._populations.appendleft(selected)
        if len(self._populations) > self._H:
            self._populations.pop()

        self._generation += 1

    def _sample_solution(self) -> np.ndarray:
        history_index = np.random.randint(0, min(len(self._populations), self._H))
        x1_index = np.random.randint(0, self._mu)
        x2_index = np.random.randint(0, self._mu)

        diff = self._populations[history_index][x1_index] - self._populations[history_index][x2_index]
        diff *= self._F
        diff += self._path * self._chi * np.random.normal()
        return self._mean_s + diff + self._EPS * np.random.standard_normal(self._N)

    def mean_history(self) -> List[float]:
        return self._mean_history

    def _evaluate(self, x):
        if self._count_eval >= self._budget:
            return [self._worst_fitness for _ in range(len(CRITERIA))]
        self._count_eval += 1
        return [f(x) for f in CRITERIA]