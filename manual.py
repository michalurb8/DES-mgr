import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymooDes import DES
from pymoo.optimize import minimize
from pymoo.core.repair import Repair
from pymoo.core.repair import NoRepair
import argparse
from pymoo.termination import get_termination

parser = argparse.ArgumentParser(prog="DES",
                                 description='This program allows you to run DES for multi-objective optimization')

parser.add_argument('-s', '--stop', type=int, default=150,
                    help='Termination criterium.')

parser.add_argument('-d', '--dimensions', type=int, default=2,
                    help='Number of dimensions.')

parser.add_argument('-v', '--vis', default=True,
                    help='Turn off visualisation.', action='store_false')



if __name__ == "__main__":
    args = parser.parse_args()

    class MyProblem(ElementwiseProblem):

        def __init__(self):
            super().__init__(n_var=args.dimensions,
                            n_obj=2,
                            n_ieq_constr=0,
                            xl=-10,
                            xu=10)

        def _evaluate(self, x, out, *args, **kwargs):
            f1 = sum([xi**2 for xi in x[1:]]) + (x[0] - 1) ** 2
            f2 = sum([xi**2 for xi in x[1:]]) + (x[0] + 1) ** 2

            out["F"] = [f1, f2]
            out["G"] = []

    problem = MyProblem()
    alg = DES(visuals=args.vis)

    res = minimize(problem, alg, termination=get_termination("n_iter", args.stop), save_history=False)

    X = res.X
    print(X)
    F = res.F