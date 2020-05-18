#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt

def main():
    with np.load('out/EKF/output_data.npy') as data:
        ekf_filter_estimated_pose = data['mean_trajectory']
        ekf_covariance = data['covariance_trajectory']

    with np.load('out/EKF/input_data.npy') as data:
        ekf_actual_pose = data['real_robot_path']

    with np.load('out/PF/output_data.npy') as data:
        pf_filter_estimated_pose = data['mean_trajectory']
        pf_covariance = data['covariance_trajectory']

    with np.load('out/PF/input_data.npy') as data:
        pf_actual_pose = data['real_robot_path']

    ekf_pose_error = ekf_filter_estimated_pose - ekf_actual_pose
    pf_pose_error = pf_filter_estimated_pose - pf_actual_pose

    t = np.arange(0, 20, 0.1)
    print(len(t))
    print(len(ekf_pose_error[:, 0]))

    fig, ax = plt.subplots()
    ax.plot(t, ekf_pose_error[:, 0], 'b', label='x pose error')
    ax.plot(t, 3 * np.sqrt(ekf_covariance[0, 0, :]), 'r', label='+-3sigma')
    ax.plot(t, -3 * np.sqrt(ekf_covariance[0, 0, :]), 'r')
    ax.set_title('EKF x pose error')
    ax.set_xlabel('t')
    ax.legend()
    ax.grid()

    fig1, ax1 = plt.subplots()
    ax1.plot(t, ekf_pose_error[:, 1], 'b', label='y pose error')
    ax1.plot(t, 3 * np.sqrt(ekf_covariance[1, 1, :]), 'r', label='+-3sigma')
    ax1.plot(t, -3 * np.sqrt(ekf_covariance[1, 1, :]), 'r')
    ax1.set_title('EKF y pose error')
    ax1.set_xlabel('t')
    ax1.legend()
    ax1.grid()

    fig2, ax2 = plt.subplots()
    ax2.plot(t, ekf_pose_error[:, 2], 'b', label='theta pose error')
    ax2.plot(t, 3 * np.sqrt(ekf_covariance[2, 2, :]), 'r', label='+-3sigma')
    ax2.plot(t, -3 * np.sqrt(ekf_covariance[2, 2, :]), 'r')
    ax2.set_title('EKF theta pose error')
    ax2.set_xlabel('t')
    ax2.legend()
    ax2.grid()

    fig_, ax_ = plt.subplots()
    ax_.plot(t, pf_pose_error[:, 0], 'b', label='x pose error')
    ax_.plot(t, 3 * np.sqrt(pf_covariance[0, 0, :]), 'r', label='+-3sigma')
    ax_.plot(t, -3 * np.sqrt(pf_covariance[0, 0, :]), 'r')
    ax_.set_title('PF x pose error')
    ax_.set_xlabel('t')
    ax_.legend()
    ax_.grid()

    fig_1, ax_1 = plt.subplots()
    ax_1.plot(t, pf_pose_error[:, 1], 'b', label='y pose error')
    ax_1.plot(t, 3 * np.sqrt(pf_covariance[1, 1, :]), 'r', label='+-3sigma')
    ax_1.plot(t, -3 * np.sqrt(pf_covariance[1, 1, :]), 'r')
    ax_1.set_title('PF y pose error')
    ax_1.set_xlabel('t')
    ax_1.legend()
    ax_1.grid()

    fig_2, ax_2 = plt.subplots()
    ax_2.plot(t, pf_pose_error[:, 2], 'b', label='theta pose error')
    ax_2.plot(t, 3 * np.sqrt(pf_covariance[2, 2, :]), 'r', label='+-3sigma')
    ax_2.plot(t, -3 * np.sqrt(pf_covariance[2, 2, :]), 'r')
    ax_2.set_title('PF theta pose error')
    ax_2.set_xlabel('t')
    ax_2.legend()
    ax_2.grid()

    plt.show()


if __name__ == '__main__':
    main()
