import winsound
from collections import namedtuple
from functools import partial
from random import choice, random, seed
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# SET RANDOM SEED
# seed(a=42069)

# Initialize number of points
NUM_OF_POINTS = 20

# Establish the origin position; (0, 0) is the typical tuple
ORIGIN = (0, 0)

# Establish the time difference between points plotted
TIME_DELTA_MS = 100

# Bool for whether a point that was just previously plotted can be selected
ALLOW_BACKTRACK = False

# Set FPS of animation (frames per second)
FPS = 60

# Set the animation DPI (dots per inch), essentially resolution
DPI = 100

# Set the figure size in inches
FIGSIZE = (12, 10)

# Configure the style for pyplot
plt.style.use("fivethirtyeight")

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


def init_fig(fig, ax, artists):
    """ Initailizes the figure. Used to draw the first frame for the animation.
    """
    # Set plot title
    ax.set_title(f"Random Diagonal Walk")

    # Create equal axes ratio
    ax.set_aspect('equal', adjustable='box')

    return artists


def frame_iter(points_interp: Tuple[list, list]) -> Tuple[list, list]:
    """ Iterate through frames to determine new random point list. """
    # ? Possibly pass in number of points and use range instead of enumerate
    x_vals, y_vals = points_interp

    # Yield the x values and y values up to and including ith index
    for i, _ in enumerate(x_vals):
        yield x_vals[:(i + 1)], y_vals[:(i + 1)]


def update_artists(frames, artists):
    """ Update artists with data from each frame. """
    x_vals, y_vals = frames

    # These are the main marker points
    x_vals_main = x_vals[::(artists.smoothness + 1)]
    y_vals_main = y_vals[::(artists.smoothness + 1)]

    # ! Plot the intermediate points (interp)
    artists.interp_plot.set_data(x_vals, y_vals)
    artists.interp_plot.set_color("black")
    artists.interp_plot.set_linewidth(1)
    artists.interp_plot.set_markersize(0)
    # artists.interp_plot.set_marker(".")
    # artists.interp_plot.set_markerfacecolor("yellow")

    # ! Set the main plot data and the time text
    artists.main_plot.set_data(x_vals_main, y_vals_main)
    artists.ax_text.set_text(
        f"{len(x_vals)/FPS: 0.3f}s")

    # Create equal axes ratio and set lims
    artists.main_ax.set_aspect('equal', adjustable='box')
    artists.main_ax.set_xlim((min(x_vals) - 1/2, max(x_vals) + 1/2))
    artists.main_ax.set_ylim((min(y_vals) - 1/2, max(y_vals) + 1/2))

    # Set main plot properties
    artists.main_plot.set_color("black")
    artists.main_plot.set_linewidth(0)
    artists.main_plot.set_marker("o")
    artists.main_plot.set_markersize(5)
    artists.main_plot.set_markerfacecolor("blue")

    # # ? Show every point text
    # if len(x_vals_main) == 17 and artists.flag.flag is False:
    #     artists.flag.flag = True
    #     i = 0
    #     for x, y in zip(x_vals, y_vals):
    #         artists.main_ax.text(x=x, y=y, s=f"{i}")
    #         i += 1

    # ! Plot the origin with its own properties
    artists.origin_plot.set_data(x_vals[0], y_vals[0])
    artists.origin_plot.set_marker("o")
    artists.origin_plot.set_markersize(8)
    artists.origin_plot.set_markerfacecolor("red")

    # ! Plot the last point with its own properties
    artists.last_point_plot.set_data(x_vals[-1], y_vals[-1])
    artists.last_point_plot.set_marker("o")
    artists.last_point_plot.set_markersize(8)
    artists.last_point_plot.set_markerfacecolor("cyan")


def generate_random_points(num_of_points: int) -> Tuple[list, list]:
    """ Generates the random points for the random walk before the animation
        begins.

        Returns the list of x values and y values for the points.
    """
    # Initialize the list type for x_vals and y_vals
    x_vals = [0]
    y_vals = [0]

    # Begin random point generation loop
    for i in range(num_of_points - 1):
        # Selects a random corner to plot
        if i >= 2:
            prev_point = x_vals[-2], y_vals[-2]
        else:
            prev_point = None
        (x_curr, y_curr) = x_vals[-1], y_vals[-1]
        (x_new, y_new) = pick_random_corner((x_curr, y_curr), prev_point)
        x_vals.append(x_new)
        y_vals.append(y_new)
    return x_vals, y_vals


def linear_interpolator(x0, y0, x1, y1, num_of_points):
    """ Returns a list of points that interpolate between two points.
    """
    # TODO: Refactor to use NUMPY to vectorize
    slope = (y1 - y0) / (x1 - x0)

    def line_func(x):
        return slope * (x - x0) + y0

    dist = x1 - x0
    dx = dist/(num_of_points + 1)

    interp_points = ([], [])
    for i in range(num_of_points + 1):
        x_new = x0 + dx * i
        y_new = line_func(x_new)
        interp_points[0].append(x_new)
        interp_points[1].append(y_new)

    interp_points[0].append(x1)
    interp_points[1].append(y1)
    return interp_points


def linear_interp(points: Tuple[list, list],
                  smoothness: int = 50) -> Tuple[list, list]:
    """ Linearly interpolates between the points provided.

        points:     a tuple of two lists of points x and y
        smoothness: number of points to interpolate between each point

        E.G. ([1, 2, 3, 4], [1, 6, 9, 1]) <-> (x_vals, y_vals)
    """
    x_vals, y_vals = points
    interp_points = [[], []]

    # TODO: Refactor to make more Pythonic
    num_points = len(x_vals)
    for i in range(num_points - 1):
        x0, y0 = x_vals[i], y_vals[i]
        x1, y1 = x_vals[i + 1], y_vals[i + 1]
        interp_points_temp = linear_interpolator(
            x0=x0, y0=y0, x1=x1, y1=y1, num_of_points=smoothness)

        # Prevents double counting due to calling x0 and x1 twice
        if i == num_points - 2:
            interp_points[0] += interp_points_temp[0]
            interp_points[1] += interp_points_temp[1]
        else:
            interp_points[0] += interp_points_temp[0][:-1]
            interp_points[1] += interp_points_temp[1][:-1]

    return tuple(interp_points)


def main():
    """ Main function that plots the random walk. """
    # Create the plot
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.grid()

    # Generate the random walk points
    points = generate_random_points(num_of_points=NUM_OF_POINTS)

    # Calculate the smoothness based on time difference between points and FPS
    # Smoothness factor for interpolation (number of intermediate points)
    SMOOTHNESS = round(FPS * TIME_DELTA_MS/1000)

    # Generate the interpolated points
    points_interp = linear_interp(points=points, smoothness=SMOOTHNESS)

    # Initialize the artists with empty data
    Artists = namedtuple(
        "Artists",
        ("main_plot",
         "last_point_plot",
         "interp_plot",
         "origin_plot",
         "ax_text",
         "main_ax",
         "smoothness",
         "flag"
         )
    )

    # Create a flag class for debugging point text
    class Flag():
        flag = False

    artists = Artists(plt.plot([], [], animated=True)[0],
                      plt.plot([], [], animated=True)[0],
                      plt.plot([], [], animated=True)[0],
                      plt.plot([], [], animated=True)[0],
                      ax.text(x=0.69, y=0.90, s="",
                              transform=fig.transFigure),
                      ax,
                      SMOOTHNESS,
                      Flag)

    # Apply the plotting functions
    init = partial(init_fig, fig=fig, ax=ax, artists=artists)
    step = partial(frame_iter, points_interp=points_interp)
    update = partial(update_artists, artists=artists)

    # Create the animation
    anim = FuncAnimation(
        fig=fig,
        func=update,
        frames=step,
        init_func=init,
        repeat=False,
        save_count=len(points_interp[0])
    )

    # Save the animation
    anim.save(
        filename=r"media\sample1.mp4",
        fps=FPS,
        extra_args=['-vcodec', 'libx264'],
        dpi=DPI
    )
    # # Show the animation
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
