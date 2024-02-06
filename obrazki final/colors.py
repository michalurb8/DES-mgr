import numpy as np
import matplotlib.pyplot as plt

from random import shuffle
from functions import criteria

COLORS = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta']
COLOR_NUM = len(COLORS)
def get_CI(index):
    if index:
        return (index-1)%(COLOR_NUM - 1) + 1
    return 0

INF = float('inf')

class Population:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        self.points = []
        self.candidate = None

        self.fig, (self.axCoord, self.axVal) = plt.subplots(1, 2)
        self.fig.set_figheight(9)
        self.fig.set_figwidth(18)
        self._drawBG()
        plt.ion()
        plt.show()

        res = 101
        res = 41
        scale = 0.9
        for y in np.linspace(scale*self.ymin, scale*self.ymax, res):
            for x in np.linspace(scale*self.xmin, scale*self.xmax, res):
                newX = x
                newY = y
                # newX = np.random.normal()/2
                # newY = np.random.normal()/2 + 1
                crit = np.array([criterium(newX, newY) for criterium in criteria])
                newPoint = [np.array([newX, newY]), crit, INF]
                self.points.append(newPoint)
        print("Points generated")
        self.calcRank()
        print("Ranks calculated")
        self.survival()
        print("Points removed")
        for i in range(5):
            self.reproduce()
            print(f"Points duplicated {i}")
        self.refresh()
        print("Points drawn")

        plt.ioff()
        plt.show()

    def refresh(self) -> None:
        self.axCoord.cla()
        self.axVal.cla()
        self._drawBG()
        self.axCoord.set_xlim([self.xmin, self.xmax])
        self.axCoord.set_ylim([self.ymin, self.ymax])
        maxc = max([point[2] for point in self.points])
        buckets = 2*COLOR_NUM
        divisor = maxc // buckets
        if divisor == 0: divisor = 1
        for point in self.points:
            index = (point[2]-1)//divisor + 1
            color = COLORS[get_CI(index)]
            color = np.random.rand(3,)
            color = 'black'
            self.axCoord.scatter(point[0][0], point[0][1], color=color, s=10)
            self.axVal.scatter(point[1][0], point[1][1], color=color, s=10, zorder=100-2*point[2])
        
        if True:
            for i, (x, y) in enumerate([(0, 3),(3, 0),(-0.45, 2),(2, -0.45),(-1.3,-0.4),(-0.4, -1.3)]):
                color = np.random.rand(3,)
                f1 = criteria[0](x, y)
                f2 = criteria[1](x, y)
                self.axCoord.scatter(x, y, color=COLORS[i], s=200, zorder = 1000)
                self.axVal.scatter(f1, f2, color=COLORS[i], s=200, zorder = 1000)

        
        self.axCoord.set_title('Przestrzeń decyzyjna', fontsize=22)
        self.axCoord.set_xlabel('$x_1$', fontsize = 22, rotation = 0, labelpad = 15)
        self.axCoord.set_ylabel('$x_2$', fontsize = 22, rotation = 0, labelpad = 15)
        self.axCoord.tick_params(labelsize=16)

        self.axVal.set_title('Przestrzeń kryteriów', fontsize=22)
        self.axVal.set_xlabel('$f_1$', fontsize = 22, rotation = 0, labelpad = 15, color='blue')
        self.axVal.set_ylabel('$f_2$', fontsize = 22, rotation = 0, labelpad = 15, color='red')
        self.axVal.tick_params(labelsize=16)

    def calcRank(self) -> None:
        currentRank = 0
        i = 0
        while INF in [point[2] for point in self.points]:
            i += 1
            for currentPoint in self.points:
                if currentPoint[2] < currentRank: continue
                for otherPoint in self.points:
                    if otherPoint[2] >= currentRank and isDominated(currentPoint, otherPoint):
                        break
                else:
                    currentPoint[2]=currentRank
            currentRank += 1

    def survival(self) -> None:
        self.points = [point for point in self.points if point[2] == 0]
    
    def reproduce(self) -> None:
        newPoints = []
        for point in self.points:
            newX = point[0][0] + 0.1 * np.random.normal()
            newY = point[0][1] + 0.1 * np.random.normal()
            crit = np.array([criterium(newX, newY) for criterium in criteria])
            newPoint = [np.array([newX, newY]), crit, INF]
            newPoints.append(newPoint)
        for point in self.points:
            point[2] = INF
        self.points.extend(newPoints)
        self.calcRank()
        self.survival()
    
    def addPoint(self, newX, newY) -> None:
        crit = np.array([criterium(newX, newY) for criterium in criteria])
        newPoint = [np.array([newX, newY]), crit, INF]
        self.points.append(newPoint)

    def _drawBG(self, res = 101):
        x_ax = np.linspace(self.xmin, self.xmax, res)
        y_ax = np.linspace(self.ymin, self.ymax, res)

        xGrid, yGrid = np.meshgrid(x_ax, y_ax)

        for index, criterium in enumerate(criteria):
            z = criterium(xGrid, yGrid)
            z = z - np.amin(z)
            z = z / np.amax(z)
            z = 10 * z
            self.axCoord.contour(xGrid, yGrid, z, levels=list(np.array(list(range(30)))/3), colors=['blue', 'red'][index%COLOR_NUM], alpha=0.2, zorder = -1000)

def isDominated(p1, p2):
    for v1, v2 in zip(p1[1], p2[1]):
        if v1 <= v2:
            return False
    return True

if __name__ == "__main__":
    bounds = [-4, 4, -4, 4]
    bounds = [-2, 3.5, -2, 3.5]
    pop = Population(*bounds)