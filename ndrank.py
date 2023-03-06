from typing import List
import numpy as np
INF = float('inf')

def _isDominated(p1: np.array, p2: np.array):
    for v1, v2 in zip(p1[1], p2[1]):
        if v1 <= v2:
            return False
    return True

def calcRank(points: List) -> None:

    # points is a list of (points) lists of (list of coordinates, list of objectives and rank):
    # [[[c1, c2], [o1, o2, o3], rank], ...]

    if len(points[0][1]) == 1:
        for i in range(len(points)):
            points[i][2] = points[i][1][0]
        return
    currentRank = 0
    i = 0
    while i<len(points):
        for currentPoint in points:
            if currentPoint[2] < currentRank: continue
            for otherPoint in points:

                if otherPoint[2] >= currentRank and _isDominated(currentPoint, otherPoint):
                    break
            else:
                currentPoint[2]=currentRank
                i+=1
        currentRank += 1

def getNonDominated(points: List) -> List:
    pass

def removeDominated(points: List) -> List:
    pass

if __name__ == '__main__':
    points = [[[],[1,2,3], INF], [[], [2,3,4], INF], [[], [1,1,1], INF]]
    calcRank(points)
    print(points)
