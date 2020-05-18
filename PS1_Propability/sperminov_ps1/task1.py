import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import Axes3D

# Task1_A
mean = 1
var = 1
std_dev = np.sqrt(var)

fig, ax = plt.subplots(1, 1)

# x>=0 because the robot can't go throuth the obstacle
x = np.linspace(0, norm.ppf(0.99, mean, std_dev), 100)
prior_distribution = norm.pdf(x, mean, var)
ax.plot(x, prior_distribution, label='pdf')
ax.set_title('Task1_A.(x>=0)')
ax.legend()
plt.grid()
ax.set_xlabel('x, (m)')
plt.show()

# Task1_B
the_probability_of_the_collision = norm.cdf(x, mean, var)[0]
print('Task1_A: The_probability_of_the_collision', the_probability_of_the_collision)

# Task1_C
sensor_variance = 0.2
sensor_std_dev = np.sqrt(0.2)
sensor_mean = 0.75
z = x
likelihood_function = norm.pdf(z, sensor_mean, sensor_std_dev)
p_z = integrate.quad(lambda x_: norm.pdf(x_, sensor_mean, sensor_std_dev) * norm.pdf(x_, mean, var), -100, 100)[0] # I integrate (likelihood_function*prior_distribution) to find p(z)=const to normalize a posterior distribution.
posterior_distribution = likelihood_function * prior_distribution / p_z
posterior_distribution_normalization_check = integrate.quad(lambda x_: norm.pdf(x_, sensor_mean, sensor_std_dev) * norm.pdf(x_, mean, var) / p_z, -100, 100)
print('Task1_C. posterior distribution integral value:', posterior_distribution_normalization_check)
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(x, prior_distribution, label='Prior distribution')
ax1.plot(x, posterior_distribution, label='Posterior distribution')
ax1.set_title('Task1_C')
ax1.legend()
plt.grid()
ax1.set_xlabel('x, (m)')
plt.show()

# Task1_D
expected_value = x[np.argmax(posterior_distribution)]
print('Task1_D: Expected value of the posterior distance to the wall:', expected_value)


# Task1_E
def joint_pdf(x, z):
    prior_distribution = norm.pdf(x, mean, var)
    likelihood_function = norm.pdf(z, sensor_mean, sensor_std_dev)
    return likelihood_function * prior_distribution


X, Z = np.meshgrid(x, z)
joint_pdf_3D = joint_pdf(X, Z)
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.set_xlabel('x, (m)')
ax2.set_ylabel('z, (m)')
ax2.set_zlabel('Joint_pdf')
ax2.set_title('Task1_E. Joint pdf')
surf = ax2.plot_surface(X, Z, joint_pdf_3D, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig2.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
