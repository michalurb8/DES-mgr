from des import DES
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout

def evaluate(dimensions: int, iterations: int, lambda_arg: int, stop_after: int, visual: bool):
    """
    evaluate() runs the algorithm for multiple iterations
    Four different MOO metrics are calculated
    Parameters:
    ----------
    dimensions : int
        Objective function dimensionality
    iterations : int
        Number of different algorithm runs in a single experiment
    lambda_arg : int
        Population count. Must be > 3, if set to None, default value will be computed
    stop_after: int
        For how many generations the algorithm should run each iteration
    visual: bool
        If True, every algorithm generation will be plotted (only 2 first dimensions)
    ----------
    """
    history = []
    print("Starting evaluation...")
    lambda_prompt = str(lambda_arg) if lambda_arg is not None else "default"
    print(f"dimensions: {dimensions}; iterations: {iterations}; population: {lambda_prompt}")
    for iteration in range(iterations):
        stdout.write(f"\rIteration: {1+iteration} / {iterations}")
        stdout.flush()
        algo = DES(dimensions, lambda_arg, stop_after, visual)
        history.append(algo.get_metrics())
    print() #necessary
    values1 = [list(zip(*it_history))[0] for it_history in history]
    values2 = [list(zip(*it_history))[1] for it_history in history]
    for v in values1:
        plt.plot(v)
    plt.ion()
    plt.pause(1)
    plt.show()
    plt.close()
    for v in values2:
        plt.plot(v)
    plt.ion()
    plt.pause(1)
    plt.show()
    return

if __name__ == "__main__":
    evaluate(3, 3, 20, 40, False)