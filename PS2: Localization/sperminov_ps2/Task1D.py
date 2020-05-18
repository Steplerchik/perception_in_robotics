#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt

def main():
    with np.load('out/output_data.npy') as data:
        filter_estimated_pose = data['mean_trajectory']
        covariance = data['covariance_trajectory']

    with np.load('out/input_data.npy') as data:
        actual_pose = data['real_robot_path']

    pose_error = filter_estimated_pose - actual_pose

    t = np.arange(0, 20, 0.1)
    print(len(t))

    fig, ax = plt.subplots()
    ax.plot(t, pose_error[:, 0], 'b', label='x pose error')
    ax.plot(t, 3 * np.sqrt(covariance[0, 0, :]), 'r', label='+-3sigma')
    ax.plot(t, -3 * np.sqrt(covariance[0, 0, :]), 'r')
    ax.set_title('PF x pose error. N = N/10')
    ax.set_xlabel('t')
    ax.legend()
    ax.grid()

    fig1, ax1 = plt.subplots()
    ax1.plot(t, pose_error[:, 1], 'b', label='y pose error')
    ax1.plot(t, 3 * np.sqrt(covariance[1, 1, :]), 'r', label='+-3sigma')
    ax1.plot(t, -3 * np.sqrt(covariance[1, 1, :]), 'r')
    ax1.set_title('PF y pose error. N = N/10')
    ax1.set_xlabel('t')
    ax1.legend()
    ax1.grid()

    fig2, ax2 = plt.subplots()
    ax2.plot(t, pose_error[:, 2], 'b', label='theta pose error')
    ax2.plot(t, 3 * np.sqrt(covariance[2, 2, :]), 'r', label='+-3sigma')
    ax2.plot(t, -3 * np.sqrt(covariance[2, 2, :]), 'r')
    ax2.set_title('PF theta pose error. N = N/10')
    ax2.set_xlabel('t')
    ax2.legend()
    ax2.grid()

    plt.show()


if __name__ == '__main__':
    main()
