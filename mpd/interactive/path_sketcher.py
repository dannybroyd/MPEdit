"""
Interactive path sketcher using matplotlib.

Launch a GUI window where the user can:
  - Click and drag to draw a freehand path on the obstacle map
  - The raw sketch is automatically resampled to N evenly-spaced waypoints
  - Right-click to clear the sketch and start over
  - Press Enter or click "Done" to accept and return the path

Usage::

    from mpd.interactive.path_sketcher import PathSketcher

    sketcher = PathSketcher(env, q_pos_start, q_pos_goal, n_waypoints=64)
    path = sketcher.run()
    # path: (n_waypoints, 2) numpy array, or None if cancelled
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.interpolate import splprep, splev

from torch_robotics.torch_utils.torch_utils import to_numpy, DEFAULT_TENSOR_ARGS


class PathSketcher:
    """Interactive matplotlib-based path sketcher for 2D environments."""

    def __init__(
        self,
        env,
        q_pos_start,
        q_pos_goal,
        n_waypoints=64,
        tensor_args=DEFAULT_TENSOR_ARGS,
        figsize=(10, 10),
    ):
        """
        Args:
            env: environment instance with render(ax)
            q_pos_start: (2,) start position (path will be snapped to start here)
            q_pos_goal: (2,) goal position (path will be snapped to end here)
            n_waypoints: number of evenly-spaced waypoints to resample the sketch to
            tensor_args: device/dtype args
            figsize: figure size
        """
        self.env = env
        self.q_pos_start = to_numpy(q_pos_start) if not isinstance(q_pos_start, np.ndarray) else q_pos_start
        self.q_pos_goal = to_numpy(q_pos_goal) if not isinstance(q_pos_goal, np.ndarray) else q_pos_goal
        self.n_waypoints = n_waypoints
        self.tensor_args = tensor_args
        self.figsize = figsize

        # Drawing state
        self._raw_points = []  # list of (x, y) tuples from the mouse drag
        self._is_drawing = False
        self._sketch_line = None
        self._resampled_line = None
        self._result_path = None

    def run(self):
        """
        Open the interactive sketcher window. Blocks until user clicks Done or presses Enter.

        Returns:
            (n_waypoints, 2) numpy array of the sketched path, or None if cancelled/empty.
        """
        backend = matplotlib.get_backend()
        if backend == "agg" or backend == "Agg":
            matplotlib.use("TkAgg")
            plt.switch_backend("TkAgg")

        self.fig = plt.figure(figsize=self.figsize)

        # Main axes
        self.ax = self.fig.add_axes([0.08, 0.15, 0.84, 0.78])
        self._redraw()

        # ── Widgets ──────────────────────────────────────────────────────
        # Clear button
        ax_clear = self.fig.add_axes([0.25, 0.03, 0.15, 0.05])
        self.btn_clear = Button(ax_clear, "Clear Sketch")
        self.btn_clear.on_clicked(self._on_clear)

        # Done button
        ax_done = self.fig.add_axes([0.60, 0.03, 0.15, 0.05])
        self.btn_done = Button(ax_done, "Done ✓")
        self.btn_done.on_clicked(self._on_done)

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.fig.suptitle(
            "Path Sketcher — Click+drag to draw | Right-click: clear | Enter: done",
            fontsize=11, y=0.98,
        )

        plt.show(block=True)

        return self._result_path

    # ── Drawing ──────────────────────────────────────────────────────────

    def _redraw(self):
        """Redraw the environment and overlays."""
        self.ax.clear()
        self.env.render(self.ax)

        # Start and goal markers
        self.ax.scatter(
            self.q_pos_start[0], self.q_pos_start[1],
            color="blue", marker="X", s=250, zorder=100, label="Start",
        )
        self.ax.scatter(
            self.q_pos_goal[0], self.q_pos_goal[1],
            color="red", marker="X", s=250, zorder=100, label="Goal",
        )

        # Draw raw sketch points
        if self._raw_points:
            pts = np.array(self._raw_points)
            self._sketch_line, = self.ax.plot(
                pts[:, 0], pts[:, 1],
                color="orange", linewidth=1.5, alpha=0.5, linestyle="-",
                zorder=6, label="Raw sketch",
            )
        else:
            self._sketch_line = None

        # Draw resampled path (if computed)
        if self._result_path is not None:
            self._resampled_line, = self.ax.plot(
                self._result_path[:, 0], self._result_path[:, 1],
                color="green", linewidth=2.5, zorder=7, label="Smoothed path",
            )
        else:
            self._resampled_line = None

        n_pts = len(self._raw_points)
        self.ax.set_title(
            f"Sketch points: {n_pts}" + ("  (release mouse to smooth)" if self._is_drawing else ""),
            fontsize=11,
        )
        self.ax.legend(loc="upper left", fontsize=9)
        self.fig.canvas.draw_idle()

    # ── Event handlers ───────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 3:  # Right-click — clear
            self._on_clear(None)
            return

        if event.button == 1:  # Left-click — start drawing
            self._is_drawing = True
            self._raw_points = [(event.xdata, event.ydata)]
            self._result_path = None
            self._redraw()

    def _on_motion(self, event):
        if not self._is_drawing or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        self._raw_points.append((event.xdata, event.ydata))

        # Draw incrementally for responsiveness
        pts = np.array(self._raw_points)
        if self._sketch_line is not None:
            self._sketch_line.set_data(pts[:, 0], pts[:, 1])
        else:
            self._sketch_line, = self.ax.plot(
                pts[:, 0], pts[:, 1],
                color="orange", linewidth=1.5, alpha=0.5, zorder=6,
            )
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if not self._is_drawing:
            return
        self._is_drawing = False

        if len(self._raw_points) < 3:
            # Not enough points for a smooth path
            self._redraw()
            return

        # Resample the raw sketch into a smooth path
        self._result_path = self._resample_sketch()
        self._redraw()

    def _on_key(self, event):
        if event.key == "enter":
            self._on_done(None)

    def _on_clear(self, event):
        self._raw_points = []
        self._result_path = None
        self._sketch_line = None
        self._resampled_line = None
        self._redraw()

    def _on_done(self, event):
        # If there are raw points but no resampled path yet, resample now
        if self._result_path is None and len(self._raw_points) >= 3:
            self._result_path = self._resample_sketch()
        plt.close(self.fig)

    # ── Path processing ──────────────────────────────────────────────────

    def _resample_sketch(self):
        """
        Convert raw mouse-drag points into a smooth, evenly-spaced path.
        The first waypoint is snapped to start, the last to goal.

        Returns:
            (n_waypoints, 2) numpy array
        """
        pts = np.array(self._raw_points)  # (M, 2)

        # Prepend start, append goal
        pts = np.vstack([self.q_pos_start.reshape(1, 2), pts, self.q_pos_goal.reshape(1, 2)])

        # Remove near-duplicate consecutive points to avoid splprep issues
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate([[True], diffs > 1e-6])
        pts = pts[keep]

        if len(pts) < 4:
            # Not enough for spline — just linearly interpolate
            t_param = np.linspace(0, 1, self.n_waypoints)
            t_orig = np.linspace(0, 1, len(pts))
            path = np.column_stack([
                np.interp(t_param, t_orig, pts[:, 0]),
                np.interp(t_param, t_orig, pts[:, 1]),
            ])
            return path

        # Fit a B-spline to the sketch
        try:
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=0.001, k=min(3, len(pts) - 1))
            u_new = np.linspace(0, 1, self.n_waypoints)
            x_new, y_new = splev(u_new, tck)
            path = np.column_stack([x_new, y_new])
        except Exception:
            # Fallback to linear interpolation
            t_param = np.linspace(0, 1, self.n_waypoints)
            # Parameterize by cumulative arc length
            dists = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            dists /= dists[-1]
            path = np.column_stack([
                np.interp(t_param, dists, pts[:, 0]),
                np.interp(t_param, dists, pts[:, 1]),
            ])

        # Snap endpoints exactly to start and goal
        path[0] = self.q_pos_start
        path[-1] = self.q_pos_goal

        return path
