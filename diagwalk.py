import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import random
from typing import Tuple

# Initialize number of points
NUM_OF_POINTS = 10

# Establish the origin position; (0, 0) is the typical tuple
ORIGIN = (0, 0)

# Establish the time difference (essentially, the speed)
TIME_DELTA_MS = 1000

# Bool for whether a point that was just previously plotted can be selected
ALLOW_BACKTRACKING = False

# Configure the style for pyplot
plt.style.use("fivethirtyeight")

# Initialize x and y list of coordinates
x_vals = [ORIGIN[0]]
y_vals = [ORIGIN[1]]

# Create custom typing
Point = Tuple[float, float]


def pick_random_corner(point: Point) -> Point:
    """ Picks a random corner in one diagonal direction.

        The point is determined based off of the given point,
        where the new point is (x+-1, y+-1) randomly selected.

        Returns the new point (tuple).
    """
    (x, y) = point
    if random() < 0.5:
        x += 1
    else:
        x -= 1

    if random() < 0.5:
        y += 1
    else:
        y -= 1
    return (x, y)


def animate(i):
    """ Animate function to plot figures in pyplot. """
    (x_new, y_new) = pick_random_corner((x_vals[-1], y_vals[-1]))
    x_vals.append(x_new)
    y_vals.append(y_new)

    # Clear previous plot
    plt.cla()

    # Set title
    plt.title(f"Time {TIME_DELTA_MS*i/1000}s")

    # Create equal axes ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot item
    plt.plot(x_vals, y_vals, color="black", linestyle="solid", marker="o",
             markerfacecolor="blue", markersize=5, linewidth=1)


def main():
    """ Main function that plots the random walk. """
    ani = FuncAnimation(plt.gcf(), animate, interval=TIME_DELTA_MS)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
