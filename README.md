# DES-mgr
Implementation of multi-objective version of DES - Differential Evolution Strategy algorithm using pymoo.

# Requirements
 - Python 3.10.2

   
Python libraries:
 - pymoo 0.6.0.1
 - matplotlib 3.5.1
 - numpy 1.23.5

# Use cases
> ./main.py [-i] [-l] [-s] [-a] [-v]

- -i --iterations: An integer, number of independent algorithm runs. 10 by default
- -l --lbd: An integer, size of algorithm's population. 4*N by default, where N is the dimensionality of the feature space
- -s --stop: An integer, number of generations after which an algorithm run stops. 150 by default
- -a --arch: An integer, the maximum size of the point archive. 200 by default
- -v: A flag that turns on visual mode. Off by default
- -h --help: Show a help message with all parameters' descriptions

Example:
> ./main.py -a 20 -i 30 -s 100
