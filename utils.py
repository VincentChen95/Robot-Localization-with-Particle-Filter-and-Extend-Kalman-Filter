import numpy as np
import matplotlib.pyplot as plt
plt.ion()


def minimized_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle >= np.pi:
        angle -= 2 * np.pi
    return angle


def plot_field(env, marker_id):
    """Plot the soccer field, highlighting the currently observed marker."""
    margin = 200

    ax = env.get_figure().gca(
        aspect='equal',
        xlim=(-margin, env.COMPLETE_SIZE_X + margin),
        ylim=(-margin, env.COMPLETE_SIZE_Y + margin)
    )

    for m in env.MARKERS:
        x, y = env.MARKER_X_POS[m], env.MARKER_Y_POS[m]
        plot_circle(
            ax, (x, y), radius=20,
            facecolor=('0.8' if m == marker_id else 'w'))

        plt.annotate(m, xy=(x, y), ha='center', va='center')


def plot_robot(env, x, z, radius=5):
    """Plot the robot on the soccer field."""
    ax = env.get_figure().gca()
    plot_circle(ax, x[:2], radius=radius, facecolor='c')

    # robot orientation
    ax.plot(
        [x[0], x[0] + np.cos(x[2]) * (radius + 5)],
        [x[1], x[1] + np.sin(x[2]) * (radius + 5)],
        'k')

    # observation
    ax.plot(
        [x[0], x[0] + np.cos(x[2] + z[0]) * 100],
        [x[1], x[1] + np.sin(x[2] + z[0]) * 100],
        'b', linewidth=0.5)


def plot_path(env, states, color, linewidth=1):
    """Plot a path of states."""
    ax = env.get_figure().gca()
    ax.plot(states[:, 0], states[:, 1], color=color, linewidth=linewidth)


def plot_circle(ax, xy, radius, edgecolor='k', facecolor='w', **kwargs):
    """Plot a circle."""
    circle = plt.Circle(
        xy,
        radius=radius,
        fill=True,
        edgecolor=edgecolor,
        facecolor=facecolor,
        **kwargs)
    ax.add_artist(circle)
