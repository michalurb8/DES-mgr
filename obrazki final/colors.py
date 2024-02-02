import numpy as np
import matplotlib.pyplot as plt

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
        # res = 5
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
            # color = np.random.rand(3,)
            self.axCoord.scatter(point[0][0], point[0][1], color=color, s=14)
            self.axVal.scatter(point[1][0], point[1][1], color=color, s=15, zorder=100-2*point[2])
        
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

    def _drawBG(self, res = 101):
        x_ax = np.linspace(self.xmin, self.xmax, res)
        y_ax = np.linspace(self.ymin, self.ymax, res)

        xGrid, yGrid = np.meshgrid(x_ax, y_ax)

        for index, criterium in enumerate(criteria):
            z = criterium(xGrid, yGrid)
            z = z - np.amin(z)
            z = z / np.amax(z)
            z = 10 * z
            self.axCoord.contour(xGrid, yGrid, z, levels=list(np.array(list(range(30)))/3), colors=['blue', 'red'][index%COLOR_NUM], alpha=0.4)

def isDominated(p1, p2):
    for v1, v2 in zip(p1[1], p2[1]):
        if v1 <= v2:
            return False
    return True

if __name__ == "__main__":
    bounds = [-4, 4, -4, 4]
    pop = Population(*bounds)