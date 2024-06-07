import numpy as np
import matplotlib.pyplot as plt

data = np.load("sample_x.npy")

fig, ax = plt.subplots()

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_aspect('equal')

ax.scatter(data[:,0], data[:,1], s=0.5)
plt.savefig('sample_in_range.png')

plt.close()
