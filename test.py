import numpy as np
import time
from collections import deque

n = 1000000


startx = time.perf_counter()
arr = [5]*30
l = len(arr)
head = 0
for _ in range(n):
    arr[head] = np.random.normal()
    head = (head + 1) % l
stopx = time.perf_counter()


starty = time.perf_counter()
arr = deque([5]*30)
for _ in range(n):
    arr.append(np.random.normal())
    arr.popleft()
stopy = time.perf_counter()

print(10**6*(stopx-startx))
print()

print(10**6*(stopy-starty))
print()