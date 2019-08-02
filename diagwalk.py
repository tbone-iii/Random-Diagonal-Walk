import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import random, choice
from typing import Tuple
import winsound

# Initialize number of points
NUM_OF_POINTS = 100

# Establish the origin position; (0, 0) is the typical tuple
ORIGIN = (0, 0)

# Establish the time difference (essentially, the speed)
TIME_DELTA_MS = 150

# Bool for whether a point that was just previously plotted can be selected
ALLOW_BACKTRACK = False

# Configure the style for pyplot
plt.style.use("fivethirtyeight")

# Initialize x and y list of coordinates
x_vals = [ORIGIN[0]]
y_vals = [ORIGIN[1]]

# Create custom typing
Point = Tuple[float, float]


def beep(frequency_hz=1000, duration_ms=100):
    winsound.Beep(frequency_hz, duration_ms)


def pick_random_corner(point: Point, prev_point: Point = None,
                       allow_backtrack=ALLOW_BACKTRACK) -> Point:
    """ Picks a random corner in one diagonal direction.

        The point is determined based off of the given point,
        where the new point is (x+-1, y+-1) randomly selected.

        Point:   the current point to move from
        prev_point:   the point before the current point
        allow_backtrack:    whether backtracking to the previous point
                            is permitted


        Returns the new point (tuple).
    """
    (x, y) = point

    # Make the random selection of x and y
    if random() < 0.5:
        x_new = x + 1
    else:
        x_new = x - 1

    if random() < 0.5:
        y_new = y + 1
    else:
        y_new = y - 1

    if prev_point is not None:
        if not allow_backtrack and (x_new, y_new) == prev_point:
            # Find change in points in (x, y)
            dx, dy = (prev_point[0] - x), (prev_point[1] - y)

            # Create list of possible options for (dx, dy)
            options = ((dx, -dy), (-dx, dy), (-dx, -dy))

            # Choose a random option
            (dx, dy) = choice(options)

            # Determine new (x, y) point when backtracking is disallowed
            x_new = x + dx
            y_new = y + dy

            if (x_new, y_new) == prev_point:
                raise "Error! xnew, ynew cannot be prev point"

    return (x_new, y_new)


def animate(i):
    """ Animate function to plot figures in pyplot. """
    # TODO: Fix this mess.
    # Selects a random corner to plot
    if i >= 2:
        prev_point = x_vals[-2], y_vals[-2]
    else:
        prev_point = None
    (x_curr, y_curr) = x_vals[-1], y_vals[-1]
    (x_new, y_new) = pick_random_corner((x_curr, y_curr), prev_point)

    # Adds the new points to the x, y lists
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

    # Make a beep for each plot
    # beep()


def main():
    """ Main function that plots the random walk. """
    ani = FuncAnimation(plt.gcf(), animate, interval=TIME_DELTA_MS,
                        frames=NUM_OF_POINTS, repeat=False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
