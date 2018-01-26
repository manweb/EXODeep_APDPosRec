import numpy as np
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt

data = genfromtxt('loss.txt')

x = data[:,0]
y1 = data[:,1:4]
y2 = data[:,4:8]

fig = plt.figure(figsize=(6,6))

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.plot(x,y1, linewidth=1)
ax2.plot(x,y2, linewidth=1)

ax1.set_title('position training loss')
ax2.set_title('energy training loss')

ax1.set_xlabel('training step')
ax1.set_ylabel('loss')

ax2.set_xlabel('training step')
ax2.set_ylabel('loss')

plt.show()

