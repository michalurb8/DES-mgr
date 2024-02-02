import numpy as np
import matplotlib.pyplot as plt

from functions import criteria

INF = float('inf')

xmin = -4
xmax = 4

ymin = -4
ymax = 4

points = []
candidate = None

fig, (axCoord, axVal) = plt.subplots(1, 2)
# fig.set_figheight(9)
# fig.set_figwidth(18)

res = 101
scale = 0.9
ys = np.linspace(scale*ymin, scale*ymax, res)
xs = np.linspace(scale*xmin, scale*xmax, res)
f1s = np.full_like(ys, INF)
f2s = np.zeros_like(ys, INF)
for y in range(res):
    for x in range(res):
        xx = xs[y][x]
        yy = ys[y][x]
        f1 = criteria[0](xx, yy)
        f2 = criteria[1](xx, yy)
        f1s[y][x] = f1
        f2s[y][x] = f2
rank = np.full_like(ys, INF)
for y in range(res):
    for x in range(res):
        newX = xs[y][x]
        newY = ys[y][x]

    def calcRank() -> None:
        currentRank = 0
        i = 0
        while INF in [point[2] for point in points]:
            i += 1
            for currentPoint in points:
                if currentPoint[2] < currentRank: continue
                for otherPoint in points:
                    if otherPoint[2] >= currentRank and isDominated(currentPoint, otherPoint):
                        break
                else:
                    currentPoint[2]=currentRank
            currentRank += 1

def isDominated(p1, p2):
    for v1, v2 in zip(p1[1], p2[1]):
        if v1 <= v2:
            return False
    return True
