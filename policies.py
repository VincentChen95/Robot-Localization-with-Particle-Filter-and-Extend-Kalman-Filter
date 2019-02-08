import numpy as np


class SoccerPolicy:
    def __call__(self, x, t):
        return np.zeros((3, 1))


class OpenLoopRectanglePolicy(SoccerPolicy):
    def __init__(self, dt=0.1):
        self.dt = dt

    def __call__(self, x, t):
        n = round(t / self.dt)
        index = n % round(5 / self.dt)

        if index == 2 * int(1 / self.dt):
            u = np.array([np.deg2rad(45), 100 * self.dt, np.deg2rad(45)])
        elif index == 4 * int(1 / self.dt):
            u = np.array([np.deg2rad(45), 0, np.deg2rad(45)])
        else:
            u = np.array([0, 100 * self.dt, 0])
        return u.reshape((-1, 1))
