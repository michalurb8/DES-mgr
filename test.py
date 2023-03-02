import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

def f(_):
    lbd = 10
    x = np.random.normal(size=lbd)
    y = np.random.normal(size=lbd)
    ax.cla()
    plt.grid(zorder=1)
    plt.scatter(x,y, zorder=5, c='red', s=50)
    plt.vlines(0, -6, 6, linewidth = 10, colors='black', zorder = 2)
    plt.hlines(0, -6, 6, linewidth = 10, colors='black', zorder = 2)


ani = animation.FuncAnimation(fig, f, interval= 0)
plt.show()