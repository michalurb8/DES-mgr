import numpy as np
import matplotlib.pyplot as plt

from functions import criteria

class Population:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        self.points = []
        self.candidate = None

        self.fig, (self.axCoord, self.axVal) = plt.subplots(1, 2)
        self._drawBG()
        plt.autoscale(enable=True, axis='both')
        plt.ion()
        plt.show()

        point_limit = 2000
        iteration_limit = 20

        for i in range(10000):
            self.randomCandidate()
        print(len(self.points))
        while len(self.points) < point_limit and iteration_limit > 0:
            for _ in range(1000):
                self.spread()
            print(f"Points: {len(self.points)}, Iterations left: {iteration_limit}.")
            self.refresh()
            iteration_limit -= 1
            plt.pause(0.05)

        self.refresh()
        each = len(self.points) // 20
        for i in range(len(self.points)//each):
            point = self.points[i * each]
            self.axCoord.scatter(point[0][0], point[0][1], s=100)
            self.axVal.scatter(point[1][0], point[1][1], s=100)

        print("DONE!")
        plt.ioff()
        plt.show()

    
    def spread(self) -> None:
        parent = self.points[len(self.points)//2]
        var = 0.05
        newX = parent[0][0] + np.random.normal(0, var)
        newY = parent[0][1] + np.random.normal(0, var)
        self.candidate = (np.array([newX, newY]), np.array([criterium(newX, newY) for criterium in criteria]))
        self.removeDominated()

    def refresh(self) -> None:
        self.axCoord.cla()
        self.axVal.cla()
        self._drawBG()
        for point in self.points:
            self.axCoord.scatter(point[0][0], point[0][1], color='green', s=10)
            self.axVal.scatter(point[1][0], point[1][1], color='green', s=10)

    def removeDominated(self) -> None:
        assert self.candidate is not None, "Can't remove dominated points if there are no points"

        for point in self.points:
            if np.all(self.candidate[1] > point[1]):
                self.candidate = None
                return

        nonDominated = []
        for point in self.points:
            if np.any(self.candidate[1] > point[1]):
                nonDominated.append(point)
        self.points = nonDominated

        self.points.append(self.candidate)
        self.candidate = None


    def randomCandidate(self) -> None:
        newX = np.random.uniform(self.xmin, self.xmax)
        newY = np.random.uniform(self.ymin, self.ymax)
        self.candidate = (np.array([newX, newY]), np.array([criterium(newX, newY) for criterium in criteria]))
        self.removeDominated()
    
    def _drawBG(self, res = 101):
        COLORS = ['red', 'black', 'blue', 'green', 'orange']
        COLOR_NUM = len(COLORS)

        x_ax = np.linspace(self.xmin, self.xmax, res)
        y_ax = np.linspace(self.ymin, self.ymax, res)

        xGrid, yGrid = np.meshgrid(x_ax, y_ax)

        for index, criterium in enumerate(criteria):
            z = criterium(xGrid, yGrid)
            z = z - np.amin(z)
            z = z / np.amax(z)
            z = 10 * z
            self.axCoord.contour(xGrid, yGrid, z, levels=list(np.array(list(range(30)))/3), colors=COLORS[index%COLOR_NUM], alpha=0.4)
    
if __name__ == "__main__":
    bounds = [-4, 4, -4, 4]
    pop = Population(*bounds)