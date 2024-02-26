import argparse

from pymooDes import DES
from Evaluator import single_run

from pymoo.optimize import minimize
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.termination import get_termination

parser = argparse.ArgumentParser(prog="Multi-DES",
                                 description='This program allows you to run DES')

parser.add_argument('-i', '--iterations', type=int, default=10,
                    help='How many separate algorithm runs')

parser.add_argument('-d', '--dimensions', type=int, default=10,
                    help='Number of dimensions')

parser.add_argument('-l', '--lbd', type=int, default=None,
                    help='Population size')

parser.add_argument('-s', '--stop', type=int, default=10000,
                    help='Stop after s generations')

parser.add_argument('-a', '--arch', default=None,
                    help='Size of the archive')

parser.add_argument('-v', '--vis', default=False,
                    help='Turn on visualisation', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    alg = DES(pop_size=args.lbd, archive_size=args.arch, visuals=args.vis)
    single_run(OmniTest(), alg, args.iterations, args.stop)