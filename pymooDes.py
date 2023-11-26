import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.core.algorithm import Algorithm
from pymoo.core.repair import NoRepair
from pymoo.core.initialization import Initialization
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.population import Population
from pymoo.core.individual import Individual

from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
from pymoo.termination.default import DefaultMultiObjectiveTermination

from pymoo.termination import get_termination

# =========================================================================================================
# Implementation
# =========================================================================================================

class DES(Algorithm):

    def __init__(self,
                 sampling=FloatRandomSampling(),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 repair=None,
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
        self.shown = False #TODO

        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.N = None
        self.pop = None
        self.off = None

        self.param_archive = []

        # Algorithm class parameters (filled during _setup()):
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
        self.repair = repair if repair is not None else NoRepair()

        self.initialization = Initialization(sampling,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)


        # self.termination = DefaultMultiObjectiveTermination()
        self.termination = get_termination("n_eval", 1000)

    # def _set_optimum(self, **kwargs):
    #     if not has_feasible(self.pop):
    #         self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
    #     else:
    #         self.opt = self.pop[self.pop.get("rank") == 0]
    
    def _setup(self, problem, **kwargs):
        N = problem.n_var

        self.pop_size = 4*N
        self.n_offsprings = self.pop_size
        self.N = N

        self._mu = self.pop_size // 2
        self._EPS = 10**-6
        self._CC = np.power(N, -0.5)
        self._CD = self._mu/(self._mu + 2)
        self._CE = 2/(N*N)
        self._H = 6 + 3*np.sqrt(N)
        self._mean_curr = None
        self._mean_next = None
        self._delta = None
        self._path = None

        self.param_archive = []

        return self

    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self._mean_next = get_mean(infills)
        self._delta = None
        self._path = None

    def _selection(self):
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
        parents = self._selection()

        assert self._mean_next is not None, f"_mean_next empty during infill: {self._mean_next}" #TODO
        self._mean_curr = self._mean_next
        assert self._mean_curr is not None, f"_mean_curr empty during infill: {self._mean_curr}" #TODO
        self._mean_next = get_mean(parents)

        self._delta = self._mean_next - self._mean_curr
        self._path = (1-self._CC) * self._path + np.sqrt(self._mu * self._CC * (2 - self._CC)) if self._path is not None else self._delta

        self.param_archive.append((self.pop, self._delta, self._path))

        off = []
        for _ in range(self.pop_size):
            horizon = min(len(self.param_archive), self._H)
            point_index, delta_index, path_index = np.random.randint(0, horizon, size=3)
            point1, point2 = np.random.randint(0, self.pop_size-1, size=2)
            difference = np.sqrt(self._CD/2) * (self.param_archive[-point_index][0][point1].get("X") - self.param_archive[-point_index][0][point2].get("X"))
            difference += np.sqrt(self._CD) * np.random.normal() * self.param_archive[-delta_index][1]
            difference += np.sqrt(1-self._CD) * np.random.normal() * self.param_archive[-path_index][2]
            difference += self._EPS * np.sqrt(1-self._CE)**(self.n_gen/2) * np.random.multivariate_normal(np.zeros(self.N), np.eye(self.N))

            new_position = self._mean_next + difference        

            off.append(new_position)

        if not self.shown:
            plt.show()
            plt.ioff()
            self.shown = True
        plt.cla()
        plt.clf()
        plt.axvline(0)
        plt.axhline(0)
        for i in off:
            plt.scatter(i[0], i[1])
        plt.pause(0.2)

        pop = Population.new("X", off)
        # print(pop)
        # for p in pop:
        #     print(p.get('X'))
        # print(pop.get('X'))
        # exit()
        return pop

    def _advance(self, infills=None, **kwargs):
        # just set the new population as the newly generated individuals
        self.pop = infills

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
