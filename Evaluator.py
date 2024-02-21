from pymoo.optimize import minimize
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination

from pymoo.problems.multi.omnitest import OmniTest
from pymoo.problems.multi.zdt import ZDT1
from pymoo.problems.many.dtlz import DTLZ1

from sys import stdout
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from pymooDes import DES

def evaluate(algorithm: Algorithm, problem: Problem, iterations: int, stop_after: int):
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
    print(f"dimensions: {problem.n_var}; iterations: {iterations}; population: {algorithm.pop_size if algorithm.pop_size else problem.n_var * 4}")
    for iteration in range(iterations):
        stdout.write(f"\rIteration: {1+iteration} / {iterations}")
        stdout.flush()
        result = minimize(problem, algorithm, termination)
        history.append(np.array(result.history))
    print() # flush buffer
    averaged = np.mean(np.array(history), axis=0)
    return averaged

def TEST_algo(problem, iterations, stop_after):
    des = DES(archive_size=200, pop_size = 20)
    nsga = NSGA2()

    values_des = evaluate(des, problem, iterations, stop_after)
    values_nsga = evaluate(nsga, problem, iterations, stop_after)

    fig, axes = plt.subplot_mosaic("AB;CD;EE", constrained_layout=True)

    for axIndex, metricIndex, metric in [('A', 1, 'GD'), ('B', 2, 'IGD'), ('C', 3, 'GD+'), ('D', 4, 'IGD+'), ('E', 5, 'HV')]:
        ax = axes[axIndex]
        ax.set_title(f"Porównanie algorytmów względem metryki {metric}")
        ax.plot([i[0] for i in values_des], [i[metricIndex] for i in values_des], c='red', label='DES')
        ax.plot([i[0] for i in values_nsga], [i[metricIndex] for i in values_nsga], c='blue', label='NSGA2')
        ax.set_xlabel("Liczba ocenionych punktów")
        ax.set_ylabel(f"Wartość metryki {metric}")
        if metric != 'HV': ax.set_yscale('log')
        ax.grid()

    fig.suptitle(f"Uśrednione wartości metryk dla populacji zwracanej przez algorytmy", fontsize="20")
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc = 'upper right', title='Algorytm', fontsize="16")
    plt.setp(legend.get_title(),fontsize='24')
    plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

def archive_test(problem, iterations, stop_after, archive = [10, 50, 100, 200, 400]):

    values = []
    for a in archive:
        des = DES(pop_size = 20, archive_size=a)
        v = evaluate(des, problem, iterations, stop_after)
        values.append(v)

    fig, axes = plt.subplot_mosaic("AB;CD;EE", constrained_layout=True)

    for axIndex, metricIndex, metric in [('A', 1, 'GD'), ('B', 2, 'IGD'), ('C', 3, 'GD+'), ('D', 4, 'IGD+'), ('E', 5, 'HV')]:
        ax = axes[axIndex]
        ax.set_title(f"Porównanie algorytmów względem metryki {metric}")
        for archiveIndex, value in enumerate(values):
            ax.plot([i[0] for i in value], [i[metricIndex] for i in value], label=str(archive[archiveIndex]))
        ax.set_xlabel("Liczba ocenionych punktów")
        ax.set_ylabel(f"Wartość metryki {metric}")
        if metric != 'HV': ax.set_yscale('log')
        ax.grid()

    fig.suptitle(f"Uśrednione wartości metryk dla populacji zwracanej przez algorytmy", fontsize='20')
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc = 'upper right', title='Maksymalna liczba punktów w archiwum', fontsize='16')
    plt.setp(legend.get_title(),fontsize='24')
    plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

def lambda_test(problem, iterations, stop_after, lambdas = [12, 20]):

    values = []
    for l in lambdas:
        des = DES(pop_size = l, archive_size = 300)
        v = evaluate(des, problem, iterations, stop_after)
        values.append(v)

    fig, axes = plt.subplot_mosaic("AB;CD;EE", constrained_layout=True)

    for axIndex, metricIndex, metric in [('A', 1, 'GD'), ('B', 2, 'IGD'), ('C', 3, 'GD+'), ('D', 4, 'IGD+'), ('E', 5, 'HV')]:
        ax = axes[axIndex]
        ax.set_title(f"Porównanie algorytmów względem metryki {metric}")
        for lambdaIndex, value in enumerate(values):
            ax.plot([i[0] for i in value], [i[metricIndex] for i in value], label=str(lambdas[lambdaIndex]))
        ax.set_xlabel("Liczba ocenionych punktów")
        ax.set_ylabel(f"Wartość metryki {metric}")
        if metric != 'HV': ax.set_yscale('log')
        ax.grid()

    fig.suptitle(f"Uśrednione wartości metryk dla populacji zwracanej przez algorytmy", fontsize="16")
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc = 'upper right', title='Liczebność populacji', fontsize="20")
    plt.setp(legend.get_title(), fontsize='24')
    plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

if __name__ == "__main__":
    problem = ZDT1()
    problem = DTLZ1()
    problem = OmniTest(n_var=3)
    TEST_algo(problem = problem, iterations = 2, stop_after = 7000)
    # lambda_test(problem, 20, 6000, [10, 23, 36, 50])
    # archive_test(problem, 10, 2000, [10, 30, 60, 100])