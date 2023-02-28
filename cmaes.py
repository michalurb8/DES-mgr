import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from collections import deque

######################
CRITERIA = [lambda x: np.dot(x,x), lambda x: np.dot(x-2, x+3)]
######################

_MAXIMISE = False
_EPS = 1e-50
_POINT_MAX = 1e100
_SIGMA_MAX = 1e100

_DELAY = 0.2

infp = float('inf')
infn = float('-inf')

class CMAES:
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
        self._visuals = visuals

        self._F = 1/np.sqrt(2)
        self._C = 4/(self._N + 4)
        self._H = 6 + 3*np.sqrt(self._N)

        self._count_eval = 0
        self._budget = 1000

        # Initial point
        self._old_mean = None
        self._new_mean = None
        # Step size
        self._sigma = 1

        # Population size
        if lambda_arg == None:
            self._lambda = 4 * self._N # default population size
        else:
            self._lambda = lambda_arg
        assert self._lambda > 1, "Population size must be greater than 1"

        # Number of parents/points to be selected
        self._mu = self._lambda // 2

        # E||N(0, I)||
        self._chi = np.sqrt(self._N) * (1 - 1 / (4 * self._N) + 1 / (21 * self._N ** 2))

        # Evolution paths
        self._path_c = np.zeros(self._N)

        # Store current generation number
        self._generation = 0

        # Store important values at each generation
        self._results = []
        self._mean_history = []
        self._worst_fitness = infp

        self._populations = deque([])

        # Run the algorithm
        self._generation_loop()

    def _generation_loop(self):
        assert self._results == [], "One algorithm instance can only run once."
        self._init_first_population()
        for _ in range(self._stop_after):
            if self._visuals == True and self._N >= 2:
                self._draw()
            self._update()
            self._new_generation()

    def _update(self) -> None:
        assert len(self._populations) <= self._H, f"There should be no more than H populations saved in history"
        assert len(self._populations[0]) == self._mu, f"There should be exactly mu points in each saved population"
        assert len(self._populations[0][0]) == self._N, f"Each point should have exatcly N dimensions"
        for point in self._populations[0]:
            assert np.all(
                np.abs(point) < _POINT_MAX
            ), f"Absolute value of all generated points must be less than {_POINT_MAX} to avoid overflow errors."

        #### UPDATE TODO
        self._generation += 1


    
    def _get_diffs(self) -> List[np.array]:
        diffs = []
        for _ in range(self._lambda):
            history_index = np.random.randint(0, min(len(self._populations), self._H))
            x1_index = np.random.randint(0, self._lambda)
            x2_index = np.random.randint(0, self._lambda)

            diff = self._populations[history_index][x1_index][0] - self._populations[history_index][x2_index][0]
            diffs.append(diff)

        return diffs
    
    def _init_first_population(self, loc: np.array = 0, scale: float = 1) -> None:
        self._last_population = []
        for _ in range(self._lambda):
            new = np.random.normal(loc = loc, scale = scale, size = self._N)
            value = self._evaluate(new)
            self._last_population.append((new, value))
        selected = np.array([x[0] for x in sorted(self._last_population, key=lambda x: x[1], reverse=_MAXIMISE)][:self._mu]) ###fix sorting
        self._old_mean = self._new_mean
        self._new_mean = np.mean(selected, axis=0)

        self._populations.append(selected)

    def _new_generation(self) -> None:
        self._last_population = []
        for _ in range(self._lambda):
            new = self._sample_solution()
            value = self._evaluate(new)
            self._last_population.append((new, value))

        selected = np.array([x[0] for x in sorted(self._last_population, key=lambda x: x[1], reverse=_MAXIMISE)][:self._mu])

        self._old_mean = self._new_mean
        self._new_mean = np.mean(selected, axis=0)

        self._populations.appendleft(selected)
        if len(self._populations) > self._H:
            self._populations.pop()

    def _sample_solution(self) -> np.ndarray:
        return self._new_mean.copy() + np.random.standard_normal(self._N)

    def mean_history(self) -> List[float]:
        return self._mean_history

    def _evaluate(self, x):
        if self._count_eval < self._budget:
            return CRITERIA[0](x)
        return self._worst_fitness
        # if self._count_eval < self._budget:
        #     self._count_eval += 1
        #     return np.array([f(x) for f in CRITERIA])
        # return np.array([self._worst_fitness for _ in CRITERIA])

    def _draw(self):
        title = "Iteracja " + str(self._generation) + ", \n"
        title += "Liczebność populacji: " + str(self._lambda) + ", \n"
        title += "Wymiarowość: " + str(self._N) + ", \n"
        # title += "Funkcja celu: " + str(self._fitness.__name__)
        plt.rcParams["figure.figsize"] = (8,9)
        plt.rcParams['font.size'] = '22'
        plt.tight_layout()
        plt.subplots_adjust(top = 0.8, bottom = 0.1, left = 0.1, right = 0.99)
        plt.title(title)

        # plt.axis('equal')

        plt.axvline(0, linewidth=4, c='black')
        plt.axhline(0, linewidth=4, c='black')
        x1 = [point[0][-1] for point in self._last_population]
        x2 = [point[0][-2] for point in self._last_population]
        plt.scatter(x1, x2, s=50)
        # x1 = [point[-1] for point in self._populations[0][:self._mu]]
        # x2 = [point[-2] for point in self._populations[0][:self._mu]]
        # plt.scatter(x1, x2, s=15)
        plt.scatter(self._new_mean[-1], self._new_mean[-2], s=100, c='black')
        plt.grid()
        zoom_out = 1.3
        max1 = zoom_out*max([abs(point[0][-1]) for point in self._last_population])
        max2 = zoom_out*max([abs(point[0][-2]) for point in self._last_population])
        plt.xlim(-max1, max1)
        plt.ylim(-max2, max2)
        plt.pause(_DELAY)
        plt.clf()
        plt.cla()

if __name__ == "__main__":
    X, Y = CMAES(None, 2, 2, None, None).init_first_population(0,1)
    plt.scatter(X,Y)
    plt.show()