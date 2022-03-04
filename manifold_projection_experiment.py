import numpy as np

norm = np.array([1,1,1]) / (3**(.5))

v = np.array([2,1,2])

projected = v - v.dot(norm)*norm

print(projected)

