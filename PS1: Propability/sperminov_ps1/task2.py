import numpy as np
from scipy import linalg, dot
import matplotlib.pyplot as plt
from plot2dcov import plot2dcov


# Task2_A
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

# Distribution 1
mean_1 = np.array([[0], [0]])
covariance_1 = np.array([[1, 0], [0, 2]])

plot2dcov('Task2_A. Distribution 1', mean_1, covariance_1, 1, ax1)
plot2dcov('Task2_A. Distribution 1', mean_1, covariance_1, 2, ax1)
plot2dcov('Task2_A. Distribution 1', mean_1, covariance_1, 3, ax1)

# Distribution 2
mean_2 = np.array([[5], [0]])
covariance_2 = np.array([[3, -0.4], [-0.4, 2]])

plot2dcov('Task2_A. Distribution 2', mean_2, covariance_2, 1, ax2)
plot2dcov('Task2_A. Distribution 2', mean_2, covariance_2, 2, ax2)
plot2dcov('Task2_A. Distribution 2', mean_2, covariance_2, 3, ax2)

# Distribution 3
mean_3 = np.array([[2], [2]])
covariance_3 = np.array([[9.1, 6], [6, 4]])

plot2dcov('Task2_A. Distribution 3', mean_3, covariance_3, 1, ax3)
plot2dcov('Task2_A. Distribution 3', mean_3, covariance_3, 2, ax3)
plot2dcov('Task2_A. Distribution 3', mean_3, covariance_3, 3, ax3)

# plt.show()


# Task2_B
def sample_mean(set_of_points):
    return np.sum(set_of_points, axis=1)/len(set_of_points[0])


def sample_covariance(set_of_points):
    mean = sample_mean(set_of_points)
    result = np.array([[0., 0.], [0., 0.]])
    for i in range(len(set_of_points[0])):
        set = np.array([set_of_points[:, i] - mean])
        result += set * set.T
    result = result/(len(set_of_points[0])-1)
    return result


# Task2_C
mean = np.array([2, 2])
covariance = np.array([[1, 1.3], [1.3, 3]])
random_samples = np.random.multivariate_normal(mean, covariance, 200).T
mean = np.array([[2], [2]])

fig4, ax4 = plt.subplots()
plot2dcov('Task2_C', mean, covariance, 1, ax4, ' (initial)')
ax4.plot(random_samples[0, :], random_samples[1, :], '.', label='random_samples')
sample_mean_calculated = np.array([sample_mean(random_samples)]).T
sample_covariance_calculated = sample_covariance(random_samples)
plot2dcov('Task2_C', sample_mean_calculated, sample_covariance_calculated, 1, ax4, ' for estimated parameters')

plt.show()





