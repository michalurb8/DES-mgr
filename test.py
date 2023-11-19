from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.core.population import Population
from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.rnd import FloatRandomSampling

import numpy as np
from pymoo.problems import get_problem
from pymooDes import DES
from pymoo.core.evaluator import Evaluator


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

p = get_problem("dtlz1", n_var=3, n_obj = 2)
a = DES()
i =  Initialization(FloatRandomSampling())

def showpop(pop):
    for ind in pop:
        print(ind.F, ind.get('rank'), ind.get('crowding'))
    print()

pop = i.do(p, 10, algorithm=a)
Evaluator().eval(p, pop, algorithm=a)
showpop(pop)
showpop(_sorting(pop))
# print("SORTED!")
# showpop(pop)