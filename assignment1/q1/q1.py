import numpy as np
import matplotlib.pyplot as plt

D = 11
dimensions = np.array([2 ** i for i in range(D)])


def l1_norm(x, y):
    return np.sum(np.abs(x - y))


def l2_norm_naive(x, y):
    s = 0
    for i in range(len(x)):
        s += (x[i] - y[i]) ** 2
    return np.sqrt(s)


def squared_l2_norm(x, y):
    return np.sum(np.square(x - y))


l1_norm_avgs = np.zeros(D)
l1_norm_stds = np.zeros(D)

l2_norm_squared_avgs = np.zeros(D)
l2_norm_squared_stds = np.zeros(D)

num_points = 200

for i, d in enumerate(dimensions):
    points = np.random.rand(num_points, d)

    l1_distances = np.array(
        [l1_norm(points[i], points[j]) for i in range(num_points) for j in range(i + 1, num_points)])

    l2_distances = np.array(
        [squared_l2_norm(points[i], points[j]) for i in range(num_points) for j in range(i + 1, num_points)])

    l1_norm_avgs[i] = np.mean(l1_distances)
    l1_norm_stds[i] = np.std(l1_distances)

    l2_norm_squared_avgs[i] = np.mean(l2_distances)
    l2_norm_squared_stds[i] = np.std(l2_distances)

fig = plt.figure()

ax = fig.add_subplot(3, 1, 1)

ax.scatter(dimensions, l1_norm_avgs, marker='o', s=6, label='$\ell_1$ norm')
ax.plot(dimensions, l1_norm_avgs, alpha=0.6)
ax.scatter(dimensions, l2_norm_squared_avgs, marker='o', s=6, label='$\ell_2$ norm squared')
ax.plot(dimensions, l2_norm_squared_avgs, alpha=0.6)
ax.set_title('Average norms')
ax.set_xlabel('Dimension (log)')
ax.set_ylabel('Distance')

ax.set_xscale('log', base=2)
ax.legend()

ax = fig.add_subplot(2, 1, 2)

ax.scatter(dimensions, l1_norm_stds, marker='o', s=6, label='$\ell_1$ norm')
ax.plot(dimensions, l1_norm_stds, alpha=0.6)
ax.scatter(dimensions, l2_norm_squared_stds, marker='o', s=6, label='$\ell_2$ norm squared')
ax.plot(dimensions, l2_norm_squared_stds, alpha=0.6)
ax.set_title('Standard deviations of norms')
ax.set_xlabel('Dimension (log)')
ax.set_ylabel('Distance')

ax.set_xscale('log', base=2)
ax.legend()

fig.subplots_adjust(hspace=0.3)

plt.savefig("figures/output.png")
