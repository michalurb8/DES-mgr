from pymoo.optimize import minimize
from pymooDes import DES
import matplotlib.pyplot as plt
from pymoo.problems.multi.omnitest import OmniTest
from sys import stdout
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination

def evaluate(dimensions: int, iterations: int, lambda_arg: int, stop_after: int, visual: bool):
    """
    evaluate() runs the algorithm for multiple iterations
    MOO metrics are calculated for DES and NSGA2
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
    p = OmniTest(n_var = dimensions)
    print("Starting evaluation...")
    lambda_prompt = str(lambda_arg) if lambda_arg is not None else "default"
    print(f"dimensions: {dimensions}; iterations: {iterations}; population: {lambda_prompt}")
    des = DES()
    nsga = NSGA2()
    for iteration in range(iterations):
        stdout.write(f"\rIteration: {1+iteration} / {iterations}")
        stdout.flush()
        res = minimize(p, des, termination=get_termination("n_iter", stop_after))
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