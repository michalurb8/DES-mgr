import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.core.algorithm import Algorithm
from pymoo.core.repair import NoRepair, Repair
from pymoo.core.initialization import Initialization
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.population import Population

from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort

from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV

from pymoo.termination import get_termination
from collections import deque

_DELAY = 0.05

# =========================================================================================================
# Implementation
# =========================================================================================================

class ReflectionRepair(Repair):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        for Xi in X:
            for j in range(len(Xi)):
                while Xi[j] < problem.xl[j]:
                    Xi[j] = 2*problem.xl[j] - Xi[j]
                while Xi[j] > problem.xu[j]:
                    Xi[j] = 2*problem.xu[j] - Xi[j]
        return X

class DES(Algorithm):

    def __init__(self,
                 sampling=FloatRandomSampling(),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 repair=ReflectionRepair(),
                 visuals=False,
                 pop_size=None,
                 archive_size=None,
                 **kwargs
                 ):

        super().__init__(**kwargs)

        # DES specific parameters (filled during _setup()):
        self._EPS = None
        self._CC = None
        self._CD = None
        self._CE = None
        self._H = None
        self._mu = None
        self._mean_curr = None
        self._mean_next = None
        self._delta = None
        self._path = None
        self.visuals = visuals
        self.visuals_started = False

        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.N = None
        self.pop = None

        self.archive_size = archive_size

        self.param_archive = deque()
        self.point_archive = None

        # Algorithm class parameters (filled during _setup()):
        if pop_size:
            self.pop_size = pop_size
        else:
            self.pop_size = None
        self.n_offsprings = None

        # set the duplicate detection class - a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        # simply set the no repair object if it is None
        self.repair = repair if repair is not None else ReflectionRepair()

        self.initialization = Initialization(sampling,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)

    def _set_optimum(self, **kwargs):
        self.opt = self.point_archive
    
    def _setup(self, problem, **kwargs):
        N = problem.n_var

        if not self.pop_size:
            self.pop_size = 4*N
        self.n_offsprings = self.pop_size
        self.N = N

        self._mu = self.pop_size // 2
        self._EPS = 10**-6
        self._CC = 1/(2*np.sqrt(N))
        self._CD = self._mu/(self._mu + 2)
        self._CE = 2/(N*N)
        self._H = 6 + 3*np.sqrt(N)
        self._mean_curr = None
        self._mean_next = None
        self._delta = None
        self._path = None

        self.param_archive = deque()
        self.point_archive = None
        if not self.archive_size:
            self.archive_size = self.pop_size

        return self

    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self._mean_next = get_mean(self.pop)
        self._delta = None
        self._path = None

    def _survival(self) -> Population:
        # get the objective space values and objects
        F = self.point_archive.get("F").astype(float, copy=False)

        # do the non-dominated sorting until splitting front
        fronts = fast_non_dominated_sort(F)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                self.point_archive[i].set("rank", k)
                self.point_archive[i].set("crowding", crowding_of_front[j])
        
        sort_criteria = [lambda ind: ind.get('rank'), lambda ind: ind.get('crowding')*-1]
        I = np.lexsort([criterium(self.point_archive) for criterium in reversed(sort_criteria)])
        self.point_archive[:] = self.point_archive[I]

        return Population.create(*self.point_archive[:self.archive_size])

    def _selection(self) -> Population:
        # get the objective space values and objects
        F = self.pop.get("F").astype(float, copy=False)

        # do the non-dominated sorting until splitting front
        fronts = fast_non_dominated_sort(F)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                self.pop[i].set("rank", k)
                self.pop[i].set("crowding", crowding_of_front[j])
        
        sort_criteria = [lambda ind: ind.get('rank'), lambda ind: ind.get('crowding')*-1]
        I = np.lexsort([criterium(self.pop) for criterium in reversed(sort_criteria)])
        self.pop[:] = self.pop[I]


        return Population.create(*self.pop[:self._mu])

    def _infill(self):
        assert self._mean_next is not None, f"_mean_next empty during infill: {self._mean_next}"
        self._mean_curr = self._mean_next
        assert self._mean_curr is not None, f"_mean_curr empty during infill: {self._mean_curr}"

        parents = self._selection()

        self._mean_next = get_mean(parents)

        self._delta = self._mean_next - self._mean_curr
        self._path = (1-self._CC) * self._path + np.sqrt(self._mu * self._CC * (2 - self._CC)) * self._delta if self._path is not None else self._delta

        self.param_archive.append((parents, self._delta, self._path))
        horizon = len(self.param_archive)
        if horizon > self._H:
            self.param_archive.popleft()

        off = []
        for _ in range(self.pop_size):
            point_index, delta_index, path_index = np.random.randint(0, horizon, size=3)
            point1, point2 = np.random.randint(0, self._mu, size=2)
            difference = np.sqrt(self._CD/2) * (self.param_archive[-point_index][0][point1].get("X") - self.param_archive[-point_index][0][point2].get("X"))
            difference += np.sqrt(self._CD) * np.random.normal() * self.param_archive[-delta_index][1]
            difference += np.sqrt(1-self._CD) * np.random.normal() * self.param_archive[-path_index][2]
            difference += self._EPS * np.sqrt(1-self._CE)**(self.n_gen/2) * np.random.multivariate_normal(np.zeros(self.N), np.eye(self.N))

            new_position = self._mean_next + difference

            off.append(new_position)

        pop = Population.new("X", off)
        pop = self.repair.do(self.problem, pop)

        if self.visuals and not self.n_gen % 10:
            if not self.visuals_started:
                self.visuals_started = True
                plt.rcParams["figure.figsize"] = (12,7)
                plt.rcParams['font.size'] = '22'
                # plt.tight_layout()
                self._fig, (self._ax1, self._ax2) = plt.subplots(1, 2)
                self._fig.subplots_adjust(top = 0.8, bottom = 0.1, left = 0.1, right = 0.99)
            self._draw_features()
            self._draw_values()
            title = "Iteracja " + str(self.n_gen) + ", \n"
            title += "Liczebność populacji: " + str(self.pop_size) + ", \n"
            title += "Wymiarowość: " + str(self.N) + ", \n"
            self._fig.suptitle(title)

            plt.pause(_DELAY)

        return pop

    def _advance(self, infills=None, **kwargs):
        if self.point_archive is None:
            self.point_archive = Population.merge(Population(), self.pop)
        else:
            self.point_archive = Population.merge(self.point_archive, self.pop)
            self.point_archive = self._survival()
        
        gd = GD(self.problem.pareto_front())(self.point_archive.get('F'))
        igd = IGD(self.problem.pareto_front())(self.point_archive.get('F'))
        gdp = GDPlus(self.problem.pareto_front())(self.point_archive.get('F'))
        igdp = IGDPlus(self.problem.pareto_front())(self.point_archive.get('F'))
        hv = HV([1]*self.problem.n_obj)(self.point_archive.get('F'))
        self.history.append((self.evaluator.n_eval, gd, igd, gdp, igdp, hv))

        self.pop = infills

    def _draw_features(self):
        self._ax1.clear()
        self._ax1.grid(zorder = 1)

        self._ax1.axvline(0, linewidth=4, c='black', zorder = 2)
        self._ax1.axhline(0, linewidth=4, c='black', zorder = 2)

        for i in self.pop:
            r = i.get('rank')
            if not r: r = 1
            self._ax1.scatter(i.X[0], i.X[1], c='black', s=r*20, zorder = 3)

        self._ax1.scatter(self._mean_curr[0], self._mean_curr[1], s=50, c='yellow', zorder = 4)
        self._ax1.scatter(self._mean_next[0], self._mean_next[1], s=20, c='red', zorder = 5)
        self._ax1.plot([self._mean_curr[0], self._mean_next[0]], [self._mean_curr[1], self._mean_next[1]], zorder = 5)

        axis_equal = True
        treshold = 1.1
        max1 = max2 = float('inf')

        # if axis_equal:
        #     max1 = max([abs(i.X[0]) for i in self.pop])
        #     max1 = np.exp(np.ceil(np.log(max1)/treshold)*treshold)
        #     max2 = max([abs(i.X[1]) for i in self.pop])
        #     max2 = np.exp(np.ceil(np.log(max2)/treshold)*treshold)
        #     self._ax1.axis('equal')
        #     max1 = max2 = max(max1,max2)
        # else:
        #     max1 = 1.2*max([abs(i.X[0]) for i in self.pop])
        #     max2 = 1.2*max([abs(i.X[1]) for i in self.pop])
        #     self._ax1.axis('auto')

        max1 = min(max1, 1.2 * max(self.problem.xl[0], self.problem.xu[0]))
        max2 = min(max2, 1.2 * max(self.problem.xl[1], self.problem.xu[1]))


    def _draw_values(self):
        self._ax2.clear()
        self._ax2.grid()
        x1 = [i.F[0] for i in self.pop]
        x2 = [i.F[1] for i in self.pop]

        for i in self.point_archive:
            self._ax2.scatter(i.F[0], i.F[1], c='black', s=20, zorder = 3, alpha=0.1)

        for i in self.pop:
            r = i.get('rank')
            if not r: r = 0
            self._ax2.scatter(i.F[0], i.F[1], c=['purple', 'blue', 'green', 'yellow', 'orange', 'red'][r%6], s=20, zorder = 3)

        rang = max(max(x1) - min(x1), max(x2) - min(x2))
        if rang > self._EPS:
            scale = np.exp(1.3 + np.floor(np.log(rang)))

            min1 = np.floor(min(x1) / scale) - 0.2
            min2 = np.floor(min(x2) / scale) - 0.2

            max1 = np.ceil(max(x1) / scale) + 0.2
            max2 = np.ceil(max(x2) / scale) + 0.2

            self._ax2.axis('auto')
        else:
            self._ax2.axis('auto')

def get_mean(pop: Population):
    return np.mean([ind.get('X') for ind in pop], axis=0)

def calc_crowding_distance(F, filter_out_duplicates=True):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]

        # sort each column and get index
        I = np.argsort(_F, axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _F = _F[I, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan

        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divide by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    # crowding[np.isinf(crowding)] = 1e+14
    return crowding
