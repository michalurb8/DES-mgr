from pymoo.optimize import minimize
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination

from sys import stdout
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from pymooDes import DES

def evaluate(algorithm: Algorithm, problem: Problem, iterations: int, lambda_arg: int, stop_after: int):
    """
    evaluate() runs the algorithm on a given problem for multiple iterations
    Currently GD, IGD, GD+ and IGD+ are calculated
    Parameters:
    ----------
    algorithm : pymoo.Algorithm
    problem: pymoo.Problem
    dimensions : int
        Objective function dimensionality
    iterations : int
        Number of different algorithm runs in a single experiment
    lambda_arg : int
        Population count. Must be > 3, if set to None, default value will be computed
    stop_after: int
        For how many generations the algorithm should run each iteration
    ----------
    """
    history = []
    termination = get_termination("n_eval", stop_after)
    print("Starting evaluation...")
    lambda_prompt = str(lambda_arg) if lambda_arg is not None else f"default: {problem.n_var}"
    print(f"dimensions: {problem.n_var}; iterations: {iterations}; population: {lambda_prompt}")
    for iteration in range(iterations):
        stdout.write(f"\rIteration: {1+iteration} / {iterations}")
        stdout.flush()
        result = minimize(problem, algorithm, termination)
        history.append(np.array(result.history))
    print() # flush buffer
    averaged = np.mean(np.array(history), axis=0)
    return averaged

def dimensionality_test(problem, iterations, lambda_arg, stop_after, dimensions: List[int]):
    des = DES()
    nsga = NSGA2()
    values_des = evaluate(des, problem, iterations, lambda_arg, stop_after)
    values_nsga = evaluate(nsga, problem, iterations, lambda_arg, stop_after)

    fig, axes = plt.subplots(2, 2)

    for ix, iy, index, metric in [(0, 0, 1, 'GD'), (1, 0, 2, 'IGD'), (0, 1, 3, 'GD+'), (1, 1, 4, 'IGD+')]:
        ax = axes[iy][ix]
        ax.set_title(f"Porównanie algorytmów względem metryki {metric}")
        ax.plot([i[0] for i in values_des], [i[index] for i in values_des], c='green', label='DES')
        ax.plot([i[0] for i in values_nsga], [i[index] for i in values_nsga], c='blue', label='NSGA2')
        ax.set_xlabel("Liczba ocenionych punktów")
        ax.set_ylabel(f"Wartość metryki {metric}")
        ax.set_yscale('log')
        ax.grid()

    fig.suptitle(f"Uśrednione wartości metryk dla populacji zwracanej przez algorytmy")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'upper right')
    plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

if __name__ == "__main__":
    # evaluate(DES(), OmniTest(), 10, 20, 100)
    dimensionality_test(problem = OmniTest(n_var=3), iterations = 10, lambda_arg = None, stop_after = 10000, dimensions = [])