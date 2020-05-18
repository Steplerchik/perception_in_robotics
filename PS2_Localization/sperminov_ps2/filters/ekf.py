"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle
from field_map import FieldMap


class EKF(LocalizationFilter):
    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        self._state_bar.mu = get_prediction(self.mu, u)[np.newaxis].T

        Rt = get_motion_noise_covariance(u, self._alphas)
        Gt = self.get_jacobian_G(self.mu, u)
        Vt = self.get_jacobian_V(self.mu, u)

        self._state_bar.Sigma = Gt.dot(self.Sigma.dot(Gt.T)) + Vt.dot(Rt.dot(Vt.T))

    def update(self, z):
        # TODO implement correction step

        Qt = self._Q

        zi = z[0]
        lm_id = z[1]

        z_i = get_expected_observation(self.mu_bar, lm_id)[0]
        Hti = self.get_jacobian_H(self.mu_bar, lm_id)
        Sti = Hti.dot(self.Sigma_bar.dot(Hti[np.newaxis].T)) + Qt
        Kti = self.Sigma_bar.dot(Hti[np.newaxis].T / Sti)

        self._state_bar.mu += Kti * (zi - z_i)
        self._state_bar.Sigma = (np.eye(3) - Kti.dot(Hti[np.newaxis])).dot(self._state_bar.Sigma)

        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma

    def get_jacobian_G(self, state, motion):
        """
        :param state: The current state of the robot (format: [x, y, theta]).
        :param motion: The motion command to execute (format: [drot1, dtran, drot2]).
        :return: The Jacobian Gt.
        """

        assert isinstance(state, np.ndarray)
        assert isinstance(motion, np.ndarray)

        assert state.shape == (3,)
        assert motion.shape == (3,)

        x, y, theta = state
        drot1, dtran, drot2 = motion

        return np.array([[1, 0, -dtran * np.sin(theta + drot1)],
                        [0, 1, dtran * np.cos(theta + drot1)],
                        [0, 0, 1]])

    def get_jacobian_V(self, state, motion):
        """
        :param state: The current state of the robot (format: [x, y, theta]).
        :param motion: The motion command to execute (format: [drot1, dtran, drot2]).
        :return: The Jacobian Vt.
        """

        assert isinstance(state, np.ndarray)
        assert isinstance(motion, np.ndarray)

        assert state.shape == (3,)
        assert motion.shape == (3,)

        x, y, theta = state
        drot1, dtran, drot2 = motion

        return np.array([[-dtran * np.sin(theta + drot1), np.cos(theta + drot1), 0],
                        [dtran * np.cos(theta + drot1), np.sin(theta + drot1), 0],
                        [1, 0, 1]])

    def get_jacobian_H(self, state, lm_id):
        """
        :param state: The current state of the robot (format: [x, y, theta]).
        :param lm_id: The landmark id indexing into the landmarks list in the field map.
        :return: The Jacobian H.
        """

        assert isinstance(state, np.ndarray)
        assert state.shape == (3,)

        lm_id = int(lm_id)
        field_map = FieldMap()

        dx = field_map.landmarks_poses_x[lm_id] - state[0]
        dy = field_map.landmarks_poses_y[lm_id] - state[1]
        q = dx**2 + dy**2

        return np.array([dy/q, -dx/q, -1])
