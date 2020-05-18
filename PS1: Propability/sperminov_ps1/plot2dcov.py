import numpy as np
from scipy import linalg, dot
import matplotlib.pyplot as plt


# Task2_A
def plot2dcov(name, mean, covariance, k, ax=None, label='', n_points=30):
    if ax is None:
        fig, ax = plt.subplots()
    L = linalg.cholesky(covariance, lower=True)
    A = L * k
    points = [i/(n_points-1) * 2 * np.pi for i in range(n_points)]
    points_coords = np.array([np.sin(points), np.cos(points)])
    Ax = np.array(A).dot(points_coords)
    Ax_b = Ax + mean
    label = 'iso-contour' + label + ': k=' + str(k)
    ax.plot(Ax_b[0, :], Ax_b[1, :], label=label)
    ax.legend()
    ax.set_title(name)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')