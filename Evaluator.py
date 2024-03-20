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
    termination = get_termination("n_iter", stop_after)
    print("Starting evaluation...")
    print(f"dimensions: {problem.n_var}; iterations: {iterations}; population: {algorithm.pop_size if algorithm.pop_size else 'default'}")
    for iteration in range(iterations):
        stdout.write(f"\rIteration: {1+iteration} / {iterations}")
        stdout.flush()
        result = minimize(problem, algorithm, termination)
        history.append(np.array(result.history))
    print() # flush buffer
    averaged = np.mean(np.array(history), axis=0)
    return averaged

def TEST_algo(problem, iterations, stop_after):
    # des = DES(pop_size=20, archive_size=100)
    des = DES()
    nsga = NSGA2()

    values_des = evaluate(des, problem, iterations, stop_after)
    values_nsga = evaluate(nsga, problem, iterations, stop_after)

    fig, axes = plt.subplot_mosaic("AB;CD;EE", constrained_layout=True)

    for axIndex, metricIndex, metric in [('A', 1, 'GD'), ('B', 2, 'IGD'), ('C', 3, 'GD+'), ('D', 4, 'IGD+'), ('E', 5, 'HV')]:
        ax = axes[axIndex]
        ax.set_title(f"Porównanie algorytmów względem miary {metric}", fontsize='14')
        ax.plot([i[0] for i in values_des], [i[metricIndex] for i in values_des], c='red', label='DES')
        ax.plot([i[0] for i in values_nsga], [i[metricIndex] for i in values_nsga], c='blue', label='NSGA2')
        ax.set_xlabel("Liczba ocenionych punktów", fontsize='14')
        ax.set_ylabel(f"Wartość miary {metric}", fontsize='14')
        ax.tick_params(axis='both', labelsize=15)
        if metric != 'HV': ax.set_yscale('log')
        ax.grid()

    fig.suptitle(f"Uśrednione wartości miar dla populacji zwracanej przez algorytmy", fontsize="20")
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc = 'upper right', title='Algorytm', fontsize="16")
    plt.setp(legend.get_title(),fontsize='24')
    plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

def time_test(problem, iterations, stop_after):
    des = DES(archive_size=12)
    nsga = NSGA2()

    values_des = evaluate(des, problem, iterations, stop_after)
    values_nsga = evaluate(nsga, problem, iterations, stop_after)

    fig, ax = plt.subplots()

    ax.set_title(f"Porównanie czasu działania algorytmów", fontsize='14')
    ax.plot([i[0] for i in values_des], [i[6] for i in values_des], c='red', label='DES')
    ax.plot([i[0] for i in values_nsga], [i[6] for i in values_nsga], c='blue', label='NSGA2')
    ax.set_xlabel("Liczba ocenionych punktów", fontsize='14')
    ax.set_ylabel(f"Czas [ms]", fontsize='14')
    ax.tick_params(axis='both', labelsize=15)
    ax.grid()

    fig.suptitle(f"Uśrednione wartości miar dla populacji zwracanej przez algorytmy", fontsize="20")
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc = 'upper right', title='Algorytm', fontsize="16")
    plt.setp(legend.get_title(),fontsize='24')
    plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

def single_run(problem, algorithm, iterations, stop_after):
    v = evaluate(algorithm, problem, iterations, stop_after)

    fig, axes = plt.subplot_mosaic("AB;CD;EE", constrained_layout=True)

    for axIndex, metricIndex, metric in [('A', 1, 'GD'), ('B', 2, 'IGD'), ('C', 3, 'GD+'), ('D', 4, 'IGD+'), ('E', 5, 'HV')]:
        ax = axes[axIndex]
        ax.set_title(f"Porównanie względem miary {metric}", fontsize=20)
        ax.plot([i[0] for i in v], [i[metricIndex] for i in v])
        ax.set_xlabel("Liczba ocenionych punktów", fontsize=16)
        ax.set_ylabel(f"Wartość miary {metric}", fontsize=16)
        if metric != 'HV': ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=15)
        ax.grid()

    fig.suptitle(f"Wartości miar uśrednione z {iterations} przebiegów algorytmu DES", fontsize='24')
    # plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

def archive_time(problem, iterations, stop_after, archive = [50, 100, 200, 400]):
    values = []
    for a in archive:
        des = DES(archive_size=a)
        v = evaluate(des, problem, iterations, stop_after)
        values.append(v)

    fig, ax = plt.subplots()
    for archiveIndex, value in enumerate(values):
        ax.plot([i[0] for i in value], [i[6] for i in value], label=str(archive[archiveIndex]))
    ax.set_xlabel("Liczba ocenionych punktów", fontsize='14')
    ax.set_ylabel(f"Czas [ms]", fontsize='14')
    ax.tick_params(axis='both', labelsize=15)
    ax.grid()

    plt.title(f"Porównanie czasu działania jednego pokolenia\nw zależności od liczby obliczeń wartości kryteriów", fontsize='20')
    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(handles, labels, loc = 'upper left', title='Maksymalna liczba\npunktów w archiwum', fontsize='16')
    plt.setp(legend.get_title(),fontsize='16')
    plt.ylim(bottom=0)
    plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

def nvar_test(iterations, stop_after, nvars = [2, 3, 4, 5]):
    values = []
    for nvar in nvars:
        des = DES()
        problem = OmniTest(n_var=nvar)
        v = evaluate(des, problem, iterations, stop_after)
        values.append(v)

    fig, axes = plt.subplot_mosaic("AB;CD;EE", constrained_layout=True)

    for axIndex, metricIndex, metric in [('A', 1, 'GD'), ('B', 2, 'IGD'), ('C', 3, 'GD+'), ('D', 4, 'IGD+'), ('E', 5, 'HV')]:
        ax = axes[axIndex]
        ax.set_title(f"Porównanie względem miary {metric}")
        for nvarIndex, value in enumerate(values):
            ax.plot([i[0] for i in value], [i[metricIndex] for i in value], label=str(nvars[nvarIndex]))
        ax.set_xlabel("Liczba ocenionych punktów", fontsize=12)
        ax.set_ylabel(f"Wartość miary {metric}", fontsize=12)
        if metric != 'HV': ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=15)
        ax.grid()

    fig.suptitle(f"Zależność między wartościami miar a liczbą zmiennych decyzyjnych\ndla algorytmu DES, zadania optymalizacji OmniTest", fontsize='20')
    handles, labels = ax.get_legend_handles_labels()
    legend = axes['B'].legend(handles, labels, loc = 'upper right', title='Liczba\nzmiennych\ndecyzyjnych', fontsize='14')
    plt.setp(legend.get_title(),fontsize='16')
    plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

def add_mean_test(problem, iterations, stop_after):
    values = []
    for add_mean in [False, True]:
        des = DES(add_mean=add_mean)
        v = evaluate(des, problem, iterations, stop_after)
        values.append(v)

    fig, axes = plt.subplot_mosaic("AB;CD;EE", constrained_layout=True)

    for axIndex, metricIndex, metric in [('A', 1, 'GD'), ('B', 2, 'IGD'), ('C', 3, 'GD+'), ('D', 4, 'IGD+'), ('E', 5, 'HV')]:
        ax = axes[axIndex]
        ax.set_title(f"Porównanie względem miary {metric}")
        for addIndex, value in enumerate(values):
            ax.plot([i[0] for i in value], [i[metricIndex] for i in value], label=("Wyłączone" if addIndex else "Włączone"))
        ax.set_xlabel("Liczba ocenionych punktów", fontsize=12)
        ax.set_ylabel(f"Wartość miary {metric}", fontsize=12)
        if metric != 'HV': ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=15)
        ax.grid()

    fig.suptitle(f"Wpływ oceny punktu środkowego na działanie algorytmu DES", fontsize='20')
    handles, labels = ax.get_legend_handles_labels()
    legend = axes['E'].legend(handles, labels, loc = 'lower right', title='Uwzględnianie punktu\nśrodkowego populacji', fontsize='14')
    plt.setp(legend.get_title(),fontsize='16')
    plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

def crowding_test(problem, iterations, stop_after):
    values = []
    for crowding in [False, True]:
        des = DES(ignore_crowding=crowding)
        v = evaluate(des, problem, iterations, stop_after)
        values.append(v)

    fig, axes = plt.subplot_mosaic("AB;CD;EE", constrained_layout=True)

    for axIndex, metricIndex, metric in [('A', 1, 'GD'), ('B', 2, 'IGD'), ('C', 3, 'GD+'), ('D', 4, 'IGD+'), ('E', 5, 'HV')]:
        ax = axes[axIndex]
        ax.set_title(f"Porównanie względem miary {metric}")
        for ignoreIndex, value in enumerate(values):
            ax.plot([i[0] for i in value], [i[metricIndex] for i in value], label=("Wyłączone" if ignoreIndex else "Włączone"))
        ax.set_xlabel("Liczba ocenionych punktów", fontsize=12)
        ax.set_ylabel(f"Wartość miary {metric}", fontsize=12)
        if metric != 'HV': ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=15)
        ax.grid()

    fig.suptitle(f"Wpływ współczynnika grupowania na działanie algorytmu DES", fontsize='20')
    handles, labels = ax.get_legend_handles_labels()
    legend = axes['E'].legend(handles, labels, loc = 'lower right', title='Sortowanie populacji po wartości\nwspółczynnika grupowania', fontsize='14')
    plt.setp(legend.get_title(),fontsize='16')
    plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

def archive_test(iterations, stop_after, archive = [10, 50, 100, 200, 400]):
    problem = OmniTest(n_var=3)
    values = []
    for a in archive:
        des = DES(archive_size = a)
        v = evaluate(des, problem, iterations, stop_after)
        values.append(v)

    fig, axes = plt.subplot_mosaic("AB;CD;EE", constrained_layout=True)

    for axIndex, metricIndex, metric in [('A', 1, 'GD'), ('B', 2, 'IGD'), ('C', 3, 'GD+'), ('D', 4, 'IGD+'), ('E', 5, 'HV')]:
        ax = axes[axIndex]
        ax.set_title(f"Porównanie względem miary {metric}")
        for archiveIndex, value in enumerate(values):
            ax.plot([i[0] for i in value], [i[metricIndex] for i in value], label=str(archive[archiveIndex]))
        ax.set_xlabel("Liczba ocenionych punktów", fontsize=12)
        ax.set_ylabel(f"Wartość miary {metric}", fontsize=12)
        if metric != 'HV': ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=15)
        ax.grid()

    fig.suptitle(f"Zależność między wartościami miar a maksymalną wielkością archiwum\nalgorytmu DES dla zadania OmniTest w wersji z trzema zmiennymi decyzyjnymi", fontsize='20')
    handles, labels = ax.get_legend_handles_labels()
    legend = axes['E'].legend(handles, labels, loc = 'lower right', title='Maksymalna liczba\npunktów w archiwum', fontsize='14')
    plt.setp(legend.get_title(),fontsize='16')
    plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

def lambda_test(problem, iterations, stop_after, lambdas = [12, 20]):

    values = []
    for l in lambdas:
        des = DES(pop_size = l, archive_size = 100)
        v = evaluate(des, problem, iterations, stop_after)
        values.append(v)

    fig, axes = plt.subplot_mosaic("AB;CD;EE", constrained_layout=True)

    for axIndex, metricIndex, metric in [('A', 1, 'GD'), ('B', 2, 'IGD'), ('C', 3, 'GD+'), ('D', 4, 'IGD+'), ('E', 5, 'HV')]:
        ax = axes[axIndex]
        ax.set_title(f"Porównanie względem miary {metric}", fontsize=14)
        for lambdaIndex, value in enumerate(values):
            ax.plot([i[0] for i in value], [i[metricIndex] for i in value], label=str(lambdas[lambdaIndex]))
        ax.set_xlabel("Liczba ocenionych punktów", fontsize=12)
        ax.set_ylabel(f"Wartość miary {metric}", fontsize=12)
        if metric != 'HV': ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=15)
        ax.grid()

    fig.suptitle(f"Zależność między wartościami miar\na wielkością populacji algorytmu DES dla zadania ZDT1", fontsize="20")
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc = 'upper right', title='Liczebność\npopulacji', fontsize="12")
    plt.setp(legend.get_title(), fontsize='20')
    plt.rc('font', family='normal', weight = 'bold', size = 22)
    plt.show()

if __name__ == "__main__":
    problem = ZDT1()
    # problem = DTLZ1()
    # problem = OmniTest(n_var=3)

    # archive_time(problem=problem, iterations = 100, stop_after=60)
    # time_test(problem = problem, iterations = 8, stop_after = 100)
    # TEST_algo(problem = problem, iterations = 3, stop_after = 500)
    # lambda_test(problem, 20, 12000, [60, 120, 240, 360])
    # archive_test(20, 500, [12, 36, 100, 200])
    # nvar_test(iterations=1, stop_after=200)
    # crowding_test(problem, 10, 9000)
    add_mean_test(problem, 5, 5000)

    # minimize(problem, DES(visuals=True, pop_size = 30, archive_size= 100), get_termination('n_eval', 50000))
    # minimize(problem, NSGA2(visuals=True), get_termination('n_eval', 50000))

    # single_run(problem, DES(), 10, 250)