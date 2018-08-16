import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt

a1 = [-0.5, -1, -1, -0.5]
a2 = [-0.3, 0.2, -0.2, 0.3]
a3 = [-1, -0.5, 0.5, 0.5]
a4 = [-0.8, 0.5, 0.5, -0.6]

v = []
for i in [a1, a2, a3, a4]:
    v.append(np.array(i).reshape((2, 2)))

signal = [1, 3, 4, 2]
vec = np.array([1, 1])
verts = [vec]

for i in range(2000):
    if i % 4 == 0:
        print('\nThe {} round:'.format(i // 4 + 1))
    vec = v[signal[i % 4] - 1] @ vec
    verts.append(vec)
    print('The {} step is {}'.format(i % 4 + 1, vec))

ig, ax = plt.subplots()

Path = mpath.Path
codes = [Path.CURVE4] * (len(verts) - 2)
codes.append(Path.CLOSEPOLY)
codes.insert(0, Path.MOVETO)
path = mpath.Path(verts, codes)

# plot control points and connecting lines
x, y = zip(*path.vertices)
line, = ax.plot(x, y, 'go-')

ax.grid()
ax.axis('equal')
plt.show()
