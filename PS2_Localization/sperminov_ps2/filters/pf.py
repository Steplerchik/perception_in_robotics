"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle


class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, beta)

        self.M = num_particles
        self.particles = np.ones((self.M, 3)) * initial_state.mu[:, 0]  # Mx3 x,y,theta in a row. The particles are all in the mean
        self.X = self.particles

        # TODO add here specific class variables for the PF

    def predict(self, u):
        # TODO Implement here the PF, perdiction part

        for i in range(self.M):
            self.particles[i] = sample_from_odometry(self.particles[i], u, self._alphas)  # many noisy particles
        self.X = self.particles

        gaussian_parameters = get_gaussian_statistics(self.particles)  # parameters of the gaussian distribution of particles

        self._state_bar.mu = gaussian_parameters.mu
        self._state_bar.Sigma = gaussian_parameters.Sigma

    def update(self, z):
        # TODO implement correction step

        Qt = self._Q
        standard_deviation = np.sqrt(Qt)

        observation = z[0]
        lm_id = z[1]

        expected_observation = np.array([get_observation(self.particles[i], lm_id)[0] for i in range(self.M)])
        angle_deviations = np.array([wrap_angle(expected_observation[i] - observation) for i in range(self.M)])

        weights = gaussian().pdf(angle_deviations / standard_deviation)
        weights = weights / np.sum(weights)  # normalization

        self.particles = self.particles[self.low_variance_sampling(weights)]
        self.X = self.particles

        gaussian_parameters = get_gaussian_statistics(self.particles)
        self._state.mu = gaussian_parameters.mu
        self._state.Sigma = gaussian_parameters.Sigma

    def low_variance_sampling(self, weights):
        c = weights[0]
        i = 0
        r = uniform(0, 1/self.M)
        particle_numbers = []

        for m in range(self.M):
            U = r + m/self.M
            while U > c and i < len(weights):
                i += 1
                c += weights[i]
            particle_numbers += [i]
        return particle_numbers
