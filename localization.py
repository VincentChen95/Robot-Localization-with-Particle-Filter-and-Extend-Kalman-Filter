import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import minimized_angle, plot_field, plot_robot, plot_path
from soccer_field import Field
import policies
from ekf import ExtendedKalmanFilter
from pf import ParticleFilter


def localize(env, policy, filt, x0, num_steps, plot=False):
    # Collect data from an entire rollout
    states_noisefree, states_real, action_noisefree, obs_noisefree, obs_real = \
            env.rollout(x0, policy, num_steps)
    states_filter = np.zeros(states_real.shape)
    states_filter[0, :] = x0.ravel()

    errors = np.zeros((num_steps, 3))
    position_errors = np.zeros(num_steps)
    mahalanobis_errors = np.zeros(num_steps)

    if plot:
        fig = env.get_figure()

    for i in range(num_steps):
        x_real = states_real[i+1, :].reshape((-1, 1))
        u_noisefree = action_noisefree[i, :].reshape((-1, 1))
        z_real = obs_real[i, :].reshape((-1, 1))
        marker_id = env.get_marker_id(i)

        if filt is None:
            mean, cov = x_real, np.eye(3)
        else:
            # filters only know the action and observation
            mean, cov = filt.update(env, u_noisefree, z_real, marker_id)
        states_filter[i+1, :] = mean.ravel()

        if plot:
            fig.clear()
            plot_field(env, marker_id)
            plot_robot(env, x_real, z_real)
            plot_path(env, states_noisefree[:i+1, :], 'g', 0.5)
            plot_path(env, states_real[:i+1, :], 'b')
            if filt is not None:
                plot_path(env, states_filter[:i+1, :2], 'r')
            fig.canvas.flush_events()

        errors[i, :] = (mean - x_real).ravel()
        errors[i, 2] = minimized_angle(errors[i, 2])
        position_errors[i] = np.linalg.norm(errors[i, :2])

        cond_number = np.linalg.cond(cov)
        if cond_number > 1e12:
            print('Badly conditioned cov (setting to identity):', cond_number)
            print(cov)
            cov = np.eye(3)
        mahalanobis_errors[i] = \
                errors[i:i+1, :].dot(np.linalg.inv(cov)).dot(errors[i:i+1, :].T)

    mean_position_error = position_errors.mean()
    mean_mahalanobis_error = mahalanobis_errors.mean()
    anees = mean_mahalanobis_error / 3

    if filt is not None:
        print('-' * 80)
        print('Mean position error:', mean_position_error)
        print('Mean Mahalanobis error:', mean_mahalanobis_error)
        print('ANEES:', anees)

    if plot:
        plt.show(block=True)

#    return position_errors
    return mean_position_error,anees


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filter_type', choices=('none', 'ekf', 'pf'),
        help='filter to use for localization')
    parser.add_argument(
        '--plot', action='store_true',
        help='turn on plotting')
    parser.add_argument(
        '--seed', type=int,
        help='random seed')
    parser.add_argument(
        '--num-steps', type=int, default=200,
        help='timesteps to simulate')

    # Noise scaling factors
    parser.add_argument(
        '--data-factor', type=float, default=1,
        help='scaling factor for motion and observation noise (data)')
    parser.add_argument(
        '--filter-factor', type=float, default=1,
        help='scaling factor for motion and observation noise (filter)')
    parser.add_argument(
        '--num-particles', type=int, default=100,
        help='number of particles (particle filter only)')

    return parser


if __name__ == '__main__':
    args = setup_parser().parse_args()
    print('Data factor:', args.data_factor)
    print('Filter factor:', args.filter_factor)

    if args.seed is not None:
        np.random.seed(args.seed)

    alphas = np.array([0.05**2, 0.005**2, 0.1**2, 0.01**2])
    beta = np.diag([np.deg2rad(5)**2])

    env = Field(args.data_factor * alphas, args.data_factor * beta)
    policy = policies.OpenLoopRectanglePolicy()

    initial_mean = np.array([180, 50, 0]).reshape((-1, 1))
    initial_cov = np.diag([10, 10, 1])
    
    mean_position_error = 0
    anees_ = 0
    
    for i in range(10):
        if args.filter_type == 'none':
            filt = None
        elif args.filter_type == 'ekf':
            filt = ExtendedKalmanFilter(
                initial_mean,
                initial_cov,
                args.filter_factor * alphas,
                args.filter_factor * beta
            )
        elif args.filter_type == 'pf':
            filt = ParticleFilter(
                initial_mean,
                initial_cov,
                args.num_particles,
                args.filter_factor * alphas,
                args.filter_factor * beta
            )
    
        # You may want to edit this line to run multiple localization experiments.
        position_error , anees = localize(env, policy, filt, initial_mean, args.num_steps, args.plot)
        mean_position_error += position_error
        anees_ += anees
    print('The mean position error for 10 times',mean_position_error/10.)
    print('The anees for 10 times',anees_/10.)