import numpy as np
from scipy import linalg, dot
import matplotlib.pyplot as plt
from plot2dcov import plot2dcov

# Task3_A
del_t = 0.5
t_calc = np.arange(0.5, 3.0, 0.5)
A = np.array([[1 + del_t, 0], [0, 1]])
B = np.array([[2 * del_t], [0]])
name = 'Task3_A. Propagation state pdf'

mean = np.array([0, 0])
covariance = np.array([[0.1, 0], [0, 0.1]])
particles = np.random.multivariate_normal(mean, covariance, 500).T
mean = np.array([[mean[0]], [mean[1]]])


fig, ax = plt.subplots()
ax.set_ylim(-7, 7)
ax.set_xlim(-1, 17)
ax.scatter(particles[0, :], particles[1, :], s=0.05, c='gray', marker='o', label='particles')
plot2dcov(name, mean, covariance, 1, ax, ' (t=0)')


for i in t_calc:
    mean = A.dot(mean) + B
    covariance = A.dot(covariance.dot(A.T))
    plot2dcov(name, mean, covariance, 1, ax, ' (t=' + str(i) + ')')
    mean_ = mean.T[0]
    particles = np.random.multivariate_normal(mean_, covariance, 500).T
    ax.scatter(particles[0, :], particles[1, :], s=0.05, c='gray', marker='o')
plt.show()

# Task3_B (see comments on scanned notes)

# Task3_C
name = 'Task3_C. Propagation state pdf with control'
mean = np.array([0, 0])  # for the initial state
covariance = np.array([[0.1, 0], [0, 0.1]])  # for the initial state
A = np.eye(2)
B = np.array([[del_t, 0], [0, del_t]])
u = np.array([[3, 0], [0, 3], [3, 0], [0, -3], [3, 0]])  # controls
u = u.T
particles = np.random.multivariate_normal(mean, covariance, 500).T
mean = np.array([[mean[0]], [mean[1]]])

uncertainty_mean = np.array([[0], [0]])
uncertainty_covariance = np.array([[0.1, 0], [0, 0.1]])

fig1, ax1 = plt.subplots()
ax1.set_xlim(-1, 9)
ax1.scatter(particles[0, :], particles[1, :], s=0.05, c='gray', marker='o', label='particles')
plot2dcov(name, mean, covariance, 1, ax1, ' (t=0)')

k = -1
for i in t_calc:
    k += 1
    mean = mean + B.dot(u[:, k].reshape(2, 1)) + uncertainty_mean
    covariance = covariance + uncertainty_covariance
    plot2dcov(name, mean, covariance, 1, ax1, ' (t=' + str(i) + ')')
    mean_ = mean.T[0]
    particles = np.random.multivariate_normal(mean_, covariance, 500).T
    ax1.scatter(particles[0, :], particles[1, :], s=0.05, c='gray', marker='o')
plt.show()


# Task3_D
name = 'Task3_D. Propagation state pdf with control'
mean = np.array([0, 0, 0])  # for the initial state
covariance = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.5]])  # for the initial state
u = np.array([[3], [2]])  # controls
particles = np.random.multivariate_normal(mean[:2], covariance[:2, :2], 500).T
mean = np.array([[mean[0]], [mean[1]], [mean[2]]])

uncertainty_mean = np.array([[0], [0], [0]])
uncertainty_covariance = np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.1]])

fig2, ax2 = plt.subplots()
ax2.scatter(particles[0, :], particles[1, :], s=0.05, c='gray', marker='o', label='particles')
plot2dcov(name, mean[:2], covariance[:2, :2], 1, ax2, ' (t=0)')

k = -1
for i in t_calc:
    k += 1
    tet_mean = mean[2][0]
    B = np.array([[np.cos(tet_mean) * del_t, 0], [np.sin(tet_mean) * del_t, 0], [0, del_t]])

    mean = mean + B.dot(u) + uncertainty_mean

    J = np.array([[1, 0, -np.sin(tet_mean) * u[0] * del_t], [0, 1, np.cos(tet_mean) * u[0] * del_t], [0, 0, 1]])

    covariance = J.dot(covariance.dot(J.T)) + uncertainty_covariance
    plot2dcov(name, mean[:2], covariance[:2, :2], 1, ax2, ' (t=' + str(i) + ')')
    mean_ = mean.T[0]
    particles = np.random.multivariate_normal(mean_[:2], covariance[:2, :2], 500).T
    ax2.scatter(particles[0, :], particles[1, :], s=0.05, c='gray', marker='o')
plt.show()

# Task3_E (see comments on scanned notes)
name = 'Task3_E. Propagation state pdf with noise in the action space'
mean = np.array([0, 0, 0])  # for the initial state
covariance = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.5]])  # for the initial state
u = np.array([[3], [2]])  # controls
particles = np.random.multivariate_normal(mean[:2], covariance[:2, :2], 500).T
mean = np.array([[mean[0]], [mean[1]], [mean[2]]])

uncertainty_mean = np.array([[0], [0]])
uncertainty_covariance = np.array([[2, 0], [0, 0.1]])

fig3, ax3 = plt.subplots()
ax3.scatter(particles[0, :], particles[1, :], s=0.05, c='gray', marker='o', label='particles')
plot2dcov(name, mean[:2], covariance[:2, :2], 1, ax3, ' (t=0)')

k = -1
for i in t_calc:
    k += 1
    tet_mean = mean[2][0]
    B = np.array([[np.cos(tet_mean) * del_t, 0], [np.sin(tet_mean) * del_t, 0], [0, del_t]])

    mean = mean + B.dot(u + uncertainty_mean)

    J = np.array([[1, 0, -np.sin(tet_mean) * u[0] * del_t], [0, 1, np.cos(tet_mean) * u[0] * del_t], [0, 0, 1]])

    covariance = J.dot(covariance.dot(J.T)) + B.dot(uncertainty_covariance.dot(B.T))
    plot2dcov(name, mean[:2], covariance[:2, :2], 1, ax3, ' (t=' + str(i) + ')')
    mean_ = mean.T[0]
    particles = np.random.multivariate_normal(mean_[:2], covariance[:2, :2], 500).T
    ax3.scatter(particles[0, :], particles[1, :], s=0.05, c='gray', marker='o')
plt.show()

