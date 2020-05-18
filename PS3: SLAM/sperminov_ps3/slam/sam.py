"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from slam.slamBase import SlamBase
from tools.task import get_motion_noise_covariance

from tools.task import get_prediction
from tools.task import wrap_angle
from field_map import FieldMap

from tools.jacobian import state_jacobian, observation_jacobian, inverse_observation_jacobian
from tools.task import get_landmark_xy
from tools.task import get_expected_observation


class SAM(SlamBase):
    def predict(self, u, dt=None):

        self._state_est.mu = np.vstack((self._state_est.mu, get_prediction(self.mu_est[-3:], u)[np.newaxis].T))
        self._u = np.vstack((self._u, u[np.newaxis].T))
        self._a = np.zeros((len(self.mu_est), 1))

    def update(self, z):
        # TODO implement correction step

        Q = self._Q
        G = -np.eye(len(self._state_est.mu))

        initial_pre_multiply_for_transition = np.linalg.cholesky(np.linalg.inv(self._state_est.Sigma)).T
        G[0:3, 0:3] = initial_pre_multiply_for_transition.dot(G[0:3, 0:3])

        self._Sigma = np.zeros((3, 3, len(self._state_est.mu)//3))
        self._Sigma[0:3, 0:3, 0] = self._state_est.Sigma

        for i in range(3, len(self._state_est.mu), 3):

            Mi = get_motion_noise_covariance(self.u[i:i + 3], self._alphas)

            Gi, Vi = state_jacobian(self.mu_est[i - 3:i], self.u[i:i + 3])
            G[i:i + 3, i - 3:i] = Gi

            pre_multiply_for_transition = np.linalg.cholesky(np.linalg.inv(Vi.dot(Mi.dot(Vi.T)))).T
            self._Sigma[:, :, i//3] = Vi.dot(Mi.dot(Vi.T))

            G[i:i + 3, :] = pre_multiply_for_transition.dot(G[i:i + 3, :])

            self._a[i:i + 3] = self.mu_est[i:i+3][np.newaxis].T - get_prediction(self.mu_est[i-3:i], self.u[i:i + 3])[np.newaxis].T
            self._a[i:i + 3] = pre_multiply_for_transition.dot(self._a[i:i + 3])

        # Observation landmark vectors are created
        observed_lds = []
        for zk in z:
            ld_id = int(zk[2])
            observed_lds.extend([ld_id])
            zk_xy = get_landmark_xy(self.mu_est[-3:], zk[:2])
            L, W = inverse_observation_jacobian(self.mu_est[-3:], zk)

            if not ld_id in self._ld_ids:
                self._ld_ids.extend([ld_id])
                if self._ld_est.size == 0:
                    self._ld_est = zk_xy
                else:
                    self._ld_est = np.vstack((self._ld_est, zk_xy))

                if self._Sigma_ld.size == 0:
                    self._Sigma_ld = (L.dot(self.Sigma[:, :, -1].dot(L.T)) + W.dot(Q.dot(W.T)))[:, :, np.newaxis]
                else:
                    self._Sigma_ld = np.concatenate((self._Sigma_ld, (L.dot(self.Sigma[:, :, -1].dot(L.T)) + W.dot(Q.dot(W.T)))[:, :, np.newaxis]), axis=2)

            if self._z.size == 0:
                self._z = np.array([[zk[0]], [zk[1]]])
            else:
                self._z = np.vstack((self._z, np.array([[zk[0]], [zk[1]]])))

        if self._observed_ld_ids.size == 0:
            self._observed_ld_ids = np.array(observed_lds)

        self._observed_ld_ids = np.vstack((self._observed_ld_ids, np.array(observed_lds)))

        H = np.array([])
        J = np.array([])
        h = np.array([])
        pre_multiply_for_observation = np.linalg.cholesky(np.linalg.inv(Q)).T

        for i in range(3, len(self._state_est.mu), 3):
            for ld_id in self._observed_ld_ids[i//3]:
                j = self._ld_ids.index(ld_id)
                Hk, Jk = observation_jacobian(self.mu_est[i:i+3], self.ld_est[2*j:2*j+2])
                hk = get_expected_observation(self.mu_est[i:i+3], self.ld_est[2*j:2*j+2][np.newaxis].T)
                L, W = inverse_observation_jacobian(self.mu_est[i:i+3], hk.T[0])
                self._Sigma_ld[:, :, j] = L.dot(self.Sigma[:, :, i//3].dot(L.T)) + W.dot(Q.dot(W.T))

                H1 = np.zeros((2, len(self.mu_est)))
                H1[:, i:i+3] = Hk
                H1 = pre_multiply_for_observation.dot(H1)

                if H.size == 0:
                    H = H1
                else:
                    H = np.vstack((H, H1))

                J1 = np.zeros((2, len(self.ld_est)))
                J1[:, 2*j:2*j+2] = Jk
                J1 = pre_multiply_for_observation.dot(J1)

                if J.size == 0:
                    J = J1
                else:
                    J = np.vstack((J, J1))

                if h.size == 0:
                    h = hk
                else:
                    h = np.vstack((h, hk))

        Zeros = np.zeros((len(self.mu_est), len(self.ld_est)))

        A = np.block([
            [G, Zeros],
            [H, J]
        ])

        self._A = A

        self._c = self._z - h

        for i in range(0, len(self._c), 2):
            self._c[i:i+2] = pre_multiply_for_observation.dot(self._c[i:i+2])

        b = np.vstack((self._a, self._c))

        # So, for now we have A and b

        delta = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))

        res = (A.dot(delta) - b)
        if self._chi.size == 0:
            self._chi = (res.T.dot(res))[0]
        else:
            self._chi = np.hstack((self._chi, (res.T.dot(res))[0]))

        # Update

        self._state_est.mu = self._state_est.mu + delta[:len(self._state_est.mu)]
        self._ld_est = self._ld_est + delta[len(self._state_est.mu):]

        for i in range(0, len(self._state_est.mu), 3):
            self._state_est.mu[i+2] = wrap_angle(self._state_est.mu[i+2])


