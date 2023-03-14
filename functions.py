import numpy as np
from typing import Callable

def quadratic(x: np.ndarray) -> float:
    return float(np.dot(x, x))




def bent_cigar(x: np.ndarray) -> float:
    return x[0]**2 + 1e6 * np.sum(x[1:]**2)

def rastrigin(x: np.ndarray) -> float:
    return float(np.sum(x ** 2 + -10 * (np.cos(2 * np.pi * x)) + 10))

def elliptic(x: np.ndarray) -> float:
    dim = x.shape[0]
    if dim == 1:
        return float(np.dot(x, x))
    arr = [np.power(1e6, p) for p in np.arange(0, dim) / (dim - 1)]
    return float(np.matmul(arr, x ** 2))

def hgbat(x: np.ndarray) -> float:
    summ1 = np.sqrt(np.abs(np.dot(x,x) - sum(x)))
    summ2 = (0.5 * np.dot(x,x) + sum(x)) / x.shape[0]
    return summ1 + summ2

def rosenbrock(x: np.ndarray) -> float:
    return sum([x[i]**2 + 100*(x[i+1] - x[i]**2 - 2*x[i])**2 for i in range(x.shape[0] - 1)])

def griewank(x: np.ndarray) -> float:
    product = np.product(np.array([np.cos(x[i-1]/i) for i in range(1, x.shape[0] + 1)]))
    return np.dot(x,x)/4000 - product + 1

def ackley(x: np.ndarray) -> float:
    exp1 = -20 * np.exp(-0.2*np.sqrt(np.sum(x**2)/len(x))) + 20
    exp2 = -1  * np.exp(np.sum(np.cos(2*np.pi*x)/len(x))) + np.e
    return exp1 + exp2

def discus(x: np.ndarray) -> float:
    return 1e6*x[0]**2 + np.sum(x[1:]**2)



# def func(x: np.ndarray) -> float:

def Get_by_name(name: str) -> Callable:
    return globals()[name]



gaussian2 = lambda p, xmu=0, ymu=0 : 1-np.exp(-((p[0]-xmu)**2+(p[1]-ymu)**2)/4)
gaussian3 = lambda p, xmu=0, ymu=0, zmu=0 : 1-np.exp(-((p[0]-xmu)**2+(p[1]-ymu)**2+(p[2]-zmu)**2)/4)
gaussiann = lambda p, mean: 1-np.exp(-np.dot(p-mean, p-mean)/4)

criteriumList = [


    [ # gwiazda
        lambda p : gaussian2(p, -2, 2) + 1.01*gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 2, 2.2) + gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, 0)
    ],
    [ # gwiazda
        lambda p : gaussian2(p, -2, 2) + 1.01*gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, -2, -2)
    ],

 [ # rosenbrock i punkt
        lambda p : rosenbrock(p),
        lambda p : gaussiann(p, 2*np.ones(len(p)))
    ],
    [ # równe hantle plus nierówny punkt
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0.1),
    ],

    [ # dwa hantle równowaga plus punkt
        lambda p : 0.6*gaussian2(p, 0, 3) + 0.4*gaussian2(p, 0, -2),
        lambda p : 0.6*gaussian2(p, 3, 0) + 0.4*gaussian2(p, -2, 0),
        lambda p : gaussian2(p)
    ],
    [ # skrzyżowane nierówne hantle
        lambda p : gaussian2(p, 0, 2) + 0.8*gaussian2(p, 0, -2),
        lambda p : gaussian2(p, 2, 0) + 0.8*gaussian2(p, -2, 0),
    ],
    [ # trójkąt równoramienny
        lambda p : gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 0, 3),
    ],
    [ # dwa hantle równowaga
        lambda p : 0.6*gaussian2(p, 0, 3) + 0.4*gaussian2(p, 0, -2),
        lambda p : 0.6*gaussian2(p, 3, 0) + 0.4*gaussian2(p, -2, 0),
    ],

    [ # dwa punkty
        lambda p : gaussian2(p, 4, 2),
        lambda p : gaussian2(p, -3, -1),
    ],
 [ # równe hantle plus równy punkt
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0),
    ],

    [ # trzy współniniowe
        lambda p : gaussian2(p, 2, 0),
        lambda p : gaussian2(p, -2, 0),
        lambda p : gaussian2(p, 0, 0),
    ],


    [ # nierówno skrzyżowane hantle
        lambda p : gaussian2(p, 0, 3) + gaussian2(p, 0, -2),
        lambda p : gaussian2(p, 3, 0) + gaussian2(p, -2, 0),
    ],

    [ # linia plus punkt
        lambda p : gaussian2(p, 0, 0),
        lambda p : (p[0]+p[1])/10
    ],

    [ # linia plus równe hantle plus punkt
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0),
        lambda p : (p[0]+p[1])/10
    ],

    [ # kwadrat
        lambda p : gaussian2(p, 2, 2),
        lambda p : gaussian2(p, -2, 2),
        lambda p : gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, -2),
    ],


]

criteria = criteriumList[0]