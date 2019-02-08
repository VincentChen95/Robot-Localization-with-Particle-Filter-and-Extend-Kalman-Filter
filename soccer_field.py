import numpy as np
import matplotlib.pyplot as plt

from utils import minimized_angle
import math

class Field:
    NUM_MARKERS = 6

    INNER_OFFSET_X = 32
    INNER_OFFSET_Y = 13

    INNER_SIZE_X = 420
    INNER_SIZE_Y = 270

    COMPLETE_SIZE_X = INNER_SIZE_X + 2 * INNER_OFFSET_X
    COMPLETE_SIZE_Y = INNER_SIZE_Y + 2 * INNER_OFFSET_Y

    MARKER_OFFSET_X = 21
    MARKER_OFFSET_Y = 0

    MARKER_DIST_X = 442
    MARKER_DIST_Y = 292

    MARKERS = (1, 2, 3, 4, 5, 6)

    MARKER_X_POS = {
        1: MARKER_OFFSET_X,
        2: MARKER_OFFSET_X + 0.5 * MARKER_DIST_X,
        3: MARKER_OFFSET_X + MARKER_DIST_X,
        4: MARKER_OFFSET_X + MARKER_DIST_X,
        5: MARKER_OFFSET_X + 0.5 * MARKER_DIST_X,
        6: MARKER_OFFSET_X,
    }

    MARKER_Y_POS = {
        1: MARKER_OFFSET_Y,
        2: MARKER_OFFSET_Y,
        3: MARKER_OFFSET_Y,
        4: MARKER_OFFSET_Y + MARKER_DIST_Y,
        5: MARKER_OFFSET_Y + MARKER_DIST_Y,
        6: MARKER_OFFSET_Y + MARKER_DIST_Y,
    }


    def __init__(self, alphas, beta):
        self.alphas = alphas
        self.beta = beta

    def G(self, x, u):
        """Compute the Jacobian of the dynamics with respect to the state."""
        prev_x, prev_y, prev_theta = x.ravel()
        rot1, trans, rot2 = u.ravel()
        # YOUR IMPLEMENTATION HERE
        state_matrix = np.array([[1,0,-trans*math.sin(prev_theta+rot1)],
                                     [0,1,trans*math.cos(prev_theta+rot1)],
                                     [0,0,1]])
        return state_matrix

    def V(self, x, u):
        """Compute the Jacobian of the dynamics with respect to the control."""
        prev_x, prev_y, prev_theta = x.ravel()
        rot1, trans, rot2 = u.ravel()
        # YOUR IMPLEMENTATION HERE
        control_matrix = np.array([[-trans*math.sin(prev_theta+rot1),math.cos(prev_theta+rot1),0],
                                     [trans*math.cos(prev_theta+rot1),math.sin(prev_theta+rot1),0],
                                     [1,0,1]])
        return control_matrix
        

    def H(self, x, marker_id):
        """Compute the Jacobian of the observation with respect to the state."""
        prev_x, prev_y, prev_theta = x.ravel()
        dx = self.MARKER_X_POS[marker_id] - x[0]
        dy = self.MARKER_Y_POS[marker_id] - x[1]
        
        h_x = np.asscalar( (dy/np.power(dx,2)) / (1 + np.power(dy/dx,2 ) ) ) 
        h_y = np.asscalar( -(1/dx)/(1 + np.power(dy/dx,2) ) )

        
        return np.array([ [minimized_angle(h_x), minimized_angle(h_y),  minimized_angle(-1)]   ])
        

    def forward(self, x, u):
        """Compute next state, given current state and action.

        Implements the odometry motion model.

        x: [x, y, theta]
        u: [rot1, trans, rot2]
        """
        prev_x, prev_y, prev_theta = x
        rot1, trans, rot2 = u

        x_next = np.zeros(x.size)
        theta = prev_theta + rot1
        x_next[0] = prev_x + trans * math.cos(theta)
        x_next[1] = prev_y + trans * math.sin(theta)
        x_next[2] = minimized_angle(theta + rot2)

        return x_next.reshape((-1, 1))

    def get_marker_id(self, step):
        """Compute the landmark ID at a given timestep."""
        return ((step // 2) % self.NUM_MARKERS) + 1

    def observe(self, x, marker_id):
        """Compute observation, given current state and landmark ID.

        x: [x, y, theta]
        marker_id: int
        """
        dx = self.MARKER_X_POS[marker_id] - x[0]
        dy = self.MARKER_Y_POS[marker_id] - x[1]
        return np.array(
            [minimized_angle(np.arctan2(dy, dx) - x[2])]
        ).reshape((-1, 1))

    def noise_from_motion(self, u, alphas):
        """Compute covariance matrix for noisy action.

        u: [rot1, trans, rot2]
        alphas: noise parameters for odometry motion model
        """
        variances = np.zeros(3)
        variances[0] = alphas[0] * u[0]**2 + alphas[1] * u[1]**2
        variances[1] = alphas[2] * u[1]**2 + alphas[3] * (u[0]**2 + u[2]**2)
        variances[2] = alphas[0] * u[2]**2 + alphas[1] * u[1]**2
        return np.diag(variances)

    def likelihood(self, innovation, beta):
        """Compute the likelihood of innovation, given covariance matrix beta.

        innovation: x - mean, column vector
        beta: noise parameters for landmark observation model
        """
        norm = np.sqrt(np.linalg.det(2 * np.pi * beta))
        inv_beta = np.linalg.inv(beta)

        return np.exp(-0.5 * innovation.T.dot(inv_beta).dot(innovation)) / norm

    def sample_noisy_action(self, u, alphas=None):
        """Sample a noisy action, given a desired action and noise parameters.

        u: desired action
        alphas: noise parameters for odometry motion model (default: data alphas)
        """
        if alphas is None:
            alphas = self.alphas

        cov = self.noise_from_motion(u, alphas)
        return np.random.multivariate_normal(u.ravel(), cov).reshape((-1, 1))

    def sample_noisy_observation(self, x, marker_id, beta=None):
        """Sample a noisy observation given a current state, landmark ID, and noise
        parameters.

        x: current state
        marker_id: int
        beta: noise parameters for landmark observation model (default: data beta)
        """
        if beta is None:
            beta = self.beta

        z = self.observe(x, marker_id)
        return np.random.multivariate_normal(z.ravel(), beta).reshape((-1, 1))

    def get_figure(self):
        return plt.figure(1)

    def rollout(self, x0, policy, num_steps, dt=0.1):
        """Collect data from an entire rollout."""
        states_noisefree = np.zeros((num_steps, 3))
        states_real = np.zeros((num_steps, 3))
        action_noisefree = np.zeros((num_steps, 3))
        obs_noisefree = np.zeros((num_steps, 1))
        obs_real = np.zeros((num_steps, 1))

        x_noisefree = x_real = x0
        for i in range(num_steps):
            t = i * dt

            u_noisefree = policy(x_real, t)
            x_noisefree = self.forward(x_noisefree, u_noisefree)

            u_real = self.sample_noisy_action(u_noisefree)
            x_real = self.forward(x_real, u_real)

            marker_id = self.get_marker_id(i)
            z_noisefree = self.observe(x_real, marker_id)
            z_real = self.sample_noisy_observation(x_real, marker_id)

            states_noisefree[i, :] = x_noisefree.ravel()
            states_real[i, :] = x_real.ravel()
            action_noisefree[i, :] = u_noisefree.ravel()
            obs_noisefree[i, :] = z_noisefree.ravel()
            obs_real[i, :] = z_real.ravel()

        states_noisefree = np.concatenate([x0.T, states_noisefree], axis=0)
        states_real = np.concatenate([x0.T, states_real], axis=0)

        return (
            states_noisefree, states_real,
            action_noisefree,
            obs_noisefree, obs_real
        )
