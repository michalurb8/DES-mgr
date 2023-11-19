import numpy as np

from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.core.algorithm import Algorithm
from pymoo.core.repair import NoRepair
from pymoo.core.initialization import Initialization
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.operators.sampling.rnd import FloatRandomSampling

from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds import fast_non_dominated_sort

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

        super().__init__(save_history=True, **kwargs)

        # DES specific parameters (filled during _setup()):
        self._F = None
        self._C = None
        self._H = None

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


        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.pop = None
        self.off = None

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_rank_and_crowding'

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]
    
    def _setup(self, problem, **kwargs):
        N = problem.n_var

        self._F = 1/np.sqrt(2)
        self._C = 4/(N + 4)
        self._H = 6 + 3*np.sqrt(N)

        self.pop_size = 4 + np.floor(3 * np.log10(N))
        self.n_offsprings = self.pop_size

        return self

    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        # not necessary - first iteration only generates randomly and evaluates
        pass

    def _sorting(self):
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

    def _infill(self):
        _sorting()


        return off

    def _advance(self, infills=None, **kwargs):
        # just set the new population as the newly generated individuals
        self.pop = infills

def _sorting(pop):
    # get the objective space values and objects
    F = pop.get("F").astype(float, copy=False)

    # do the non-dominated sorting until splitting front
    fronts = fast_non_dominated_sort(F)
    print(f"fronts: {fronts}\n")
    for k, front in enumerate(fronts):

        # calculate the crowding distance of the front
        crowding_of_front = calc_crowding_distance(F[front, :])

        # save rank and crowding in the individual class
        for j, i in enumerate(front):
            pop[i].set("rank", k)
            pop[i].set("crowding", crowding_of_front[j])
    
    sort_criteria = [lambda ind: ind.get('rank'), lambda ind: ind.get('crowding')*-1]
    I = np.lexsort([criterium(pop) for criterium in reversed(sort_criteria)])
    sorted = pop[I]
    return sorted


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
