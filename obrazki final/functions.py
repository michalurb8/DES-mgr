import numpy as np

gaussian = lambda x, y, xmu=0, ymu=0 : -np.exp(-((x-xmu)**2+(y-ymu)**2)/4)

criteriumDict = {
    "dwa hantle równowaga" : [
        lambda x,y : 0.6*gaussian(x,y, 0,3) + 0.4*gaussian(x,y, 0,-2),
        lambda x,y : 0.6*gaussian(x,y, 3,0) + 0.4*gaussian(x,y, -2,0),
    ],

    "nierówno skrzyżowane hantle" : [
        lambda x,y : gaussian(x,y, 0,3) + gaussian(x,y, 0,-2),
        lambda x,y : gaussian(x,y, 3,0) + gaussian(x,y, -2,0),
    ],

    "dwa hantle równowaga plus punkt" : [
        lambda x,y : 0.6*gaussian(x,y, 0,3) + 0.4*gaussian(x,y, 0,-2),
        lambda x,y : 0.6*gaussian(x,y, 3,0) + 0.4*gaussian(x,y, -2,0),
        lambda x,y : gaussian(x,y),
    ],

    "rosenbrock" : [
        lambda x,y : 0.01*(x-1)**2 + (y-x**2)**2,
        lambda x,y : gaussian(x,y)
    ],

    "sinusy" : [
        lambda x,y : np.sin(0.2*x)*(0.2*x-1)*(0.2*y-2),
        lambda x,y : np.cos(0.2*y)*0.2*x
    ],

    "równe hantle plus punkt" : [
        lambda x,y : gaussian(x,y, 2,2) + gaussian(x,y, 2,-2),
        lambda x,y : gaussian(x,y, -2,0.1),
    ],

    "linia plus punkt" : [
        lambda x,y : gaussian(x,y, 0,0),
        lambda x,y : (x+y)/10
    ],

    "linia plus równe hantle plus punkt" : [
        lambda x,y : gaussian(x,y, 2,2) + gaussian(x,y, 2,-2),
        lambda x,y : gaussian(x,y, -2,0),
        lambda x,y : (x+y)/10
    ],

    "kwadrat" : [
        lambda x,y : gaussian(x,y,2,2),
        lambda x,y : gaussian(x,y,-2,2),
        lambda x,y : gaussian(x,y,-2,-2),
        lambda x,y : gaussian(x,y,2,-2),
    ],

    "trójkąt równoramienny" : [
        lambda x,y : gaussian(x,y,-2,-2),
        lambda x,y : gaussian(x,y,2,-2),
        lambda x,y : gaussian(x,y,0,3),
    ],

    "trzy współniniowe" : [
        lambda x,y : gaussian(x,y,2,0),
        lambda x,y : gaussian(x,y,-2,0),
        lambda x,y : gaussian(x,y,0,0),
    ],

    "skrzyżowane nierówne hantle" : [
        lambda x,y : 0.501*gaussian(x,y, 0,2) + 0.499*gaussian(x,y, 0,-2),
        lambda x,y : 0.501*gaussian(x,y, 2,0) + 0.499*gaussian(x,y, -2,0),
    ],
}

criteria = criteriumDict[list(criteriumDict.keys())[0]]