import numpy as np

from utils import minimized_angle

class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """        
        # Prediction step:  
        mu_t_bar = env.forward(self.mu,u)
        
        state_matrix = env.G(self.mu,u)
        
        control_matrix = env.V(self.mu,u)

        Motion_noise = env.noise_from_motion(u,self.alphas)
        
        pred_cov = np.dot(np.dot(state_matrix,self.sigma),state_matrix.T) + np.dot(np.dot(control_matrix,Motion_noise),control_matrix.T)
        
        # Correction step:
        expected_obs = env.observe(mu_t_bar, marker_id)
        
        H_t = env.H(mu_t_bar , marker_id)
       
        Q_t = np.array([[ self.beta[0][0] ] ])
        
        S_t = np.dot(np.dot(H_t,pred_cov),H_t.T) + Q_t
        
        Kalman_gain = np.dot(np.dot(pred_cov,H_t.T),np.linalg.inv(S_t ))
              
        self.mu = mu_t_bar + np.dot(Kalman_gain,( minimized_angle(z - expected_obs) ) )
          
        self.sigma = np.dot( np.eye(3) - np.dot(Kalman_gain,H_t), pred_cov)
        
        return self.mu, self.sigma
