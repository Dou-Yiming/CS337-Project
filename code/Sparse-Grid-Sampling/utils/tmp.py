import numpy as np

# points = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.9, 1], [1, 0.2]])
corner=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
points = np.random.random((100, 2))

points=np.concatenate((corner,points))
points*=[80,60]
from scipy.spatial import Delaunay
tri = Delaunay(points)

import matplotlib.pyplot as plt
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()