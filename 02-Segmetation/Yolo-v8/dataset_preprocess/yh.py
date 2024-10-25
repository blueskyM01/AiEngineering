import numpy as np

a = np.array([[1,3]])
print(a.shape)
ax = np.argmax(a, axis=0)
print(ax.shape)