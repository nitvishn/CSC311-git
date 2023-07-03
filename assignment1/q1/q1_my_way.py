import numpy as np

d = 2**10
num_points = 1000

# Generate random points of dimension d
points = np.random.rand(num_points, d)

# Compute the distance matrix
distances = []
for i in range(num_points):
    for j in range(i + 1, num_points):
        distances.append(np.sum(np.square(points[i] - points[j])))

# Compute the mean distance
mean_distance = np.mean(distances)

# Compute the variance of the distances
variance = np.var(distances)

print(f"Mean distance (calc): {d/6}")
print(f"Mean distance (real): {mean_distance}")
print()
print(f"Var (calc): {d * 7 / 180}")
print(f"Var (real): {variance}")