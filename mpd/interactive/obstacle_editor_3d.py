"""
Interactive 3D obstacle editor using matplotlib.

Launch a GUI window where the user can:
  - Use X, Y, Z sliders to position a new sphere
  - Use a radius slider to control obstacle size
  - Click "Add Sphere" to place the obstacle at the current slider position
  - Click "Remove Last" to undo the last placed obstacle
  - Press Enter or click "Done" to accept and return the modification list

Usage::

    from mpd.interactive.obstacle_editor_3d import ObstacleEditor3D

    editor = ObstacleEditor3D(env, tensor_args=tensor_args)
    modifications = editor.run()
    # modifications: list of dicts, e.g.
    # [{"type": "add_sphere", "center": [0.2, 0.3, 0.5], "radius": 0.1}, ...]
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import torch

from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch, DEFAULT_TENSOR_ARGS


def _plot_wireframe_sphere(ax, center, radius, color="red", alpha=0.3):
    """Draw a wireframe sphere on a 3D axis for preview/annotation."""
    u = np.linspace(0, 2 * np.pi, 16)
    v = np.linspace(0, np.pi, 12)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.5)


class ObstacleEditor3D:
    """Interactive matplotlib-based obstacle editor for 3D environments."""

    def __init__(
        self,
        env,
        tensor_args=DEFAULT_TENSOR_ARGS,
        initial_radius=0.15,
        existing_path=None,
        q_pos_start=None,
        q_pos_goal=None,
        robot=None,
        figsize=(14, 10),
    ):
        """
        Args:
            env: environment instance with render(ax), dim == 3
            tensor_args: device/dtype for creating torch obstacles
            initial_radius: default radius for sphere obstacles
            existing_path: optional (N, D) reference path to display
            q_pos_start: optional start configuration (rendered via robot FK)
            q_pos_goal: optional goal configuration (rendered via robot FK)
            robot: optional robot instance (for rendering start/goal configs)
            figsize: figure size
        """
        self.env = env
        self.tensor_args = tensor_args
        self.radius = initial_radius
        self.existing_path = to_numpy(existing_path) if existing_path is not None else None
        self.q_pos_start = to_numpy(q_pos_start) if q_pos_start is not None else None
        self.q_pos_goal = to_numpy(q_pos_goal) if q_pos_goal is not None else None
        self.robot = robot
        self.figsize = figsize

        # Compute axis ranges from environment limits
        limits_np = to_numpy(env.limits) if not isinstance(env.limits, np.ndarray) else env.limits
        self.x_min, self.x_max = float(limits_np[0][0]), float(limits_np[1][0])
        self.y_min, self.y_max = float(limits_np[0][1]), float(limits_np[1][1])
        self.z_min, self.z_max = float(limits_np[0][2]), float(limits_np[1][2])

        # Current slider positions
        self.x_coord = 0.0
        self.y_coord = 0.0
        self.z_coord = 0.0

        # State
        self.modifications = []
        self._added_objects = []
        self._done = False

        # Precompute FK positions for start/goal (if robot provided)
        self._start_pos_3d = None
        self._goal_pos_3d = None
        self._path_pos_3d = None
        if robot is not None:
            try:
                if q_pos_start is not None:
                    q_s = to_torch(q_pos_start, **tensor_args).unsqueeze(0)
                    fk_s = robot.fk_map_collision(q_s)
                    self._start_pos_3d = to_numpy(fk_s.squeeze(0))
                if q_pos_goal is not None:
                    q_g = to_torch(q_pos_goal, **tensor_args).unsqueeze(0)
                    fk_g = robot.fk_map_collision(q_g)
                    self._goal_pos_3d = to_numpy(fk_g.squeeze(0))
                if existing_path is not None:
                    q_path = to_torch(existing_path, **tensor_args)
                    if q_path.dim() == 2:
                        q_path = q_path.unsqueeze(0)
                    fk_path = robot.fk_map_collision(q_path)
                    self._path_pos_3d = to_numpy(fk_path.squeeze(0))
            except Exception as e:
                print(f"[ObstacleEditor3D] FK computation failed: {e}")
                print("  Will skip rendering start/goal/path in task space.")

    def run(self):
        """
        Open the interactive 3D editor window. Blocks until user clicks Done or presses Enter.

        Returns:
            list of modification dicts, e.g.:
            [{"type": "add_sphere", "center": [x, y, z], "radius": r}, ...]
        """
        backend = matplotlib.get_backend()
        if backend.lower() == "agg":
            matplotlib.use("TkAgg")
            plt.switch_backend("TkAgg")

        self.fig = plt.figure(figsize=self.figsize)

        # Main 3D plot area
        self.ax = self.fig.add_subplot(111, projection="3d",
                                       computed_zorder=False)
        self.fig.subplots_adjust(bottom=0.32, left=0.08, right=0.95)
        self._redraw()

        # ── Sliders ──────────────────────────────────────────────────────
        slider_left, slider_width = 0.20, 0.50

        ax_x_slider = self.fig.add_axes([slider_left, 0.22, slider_width, 0.025])
        self.slider_x = Slider(ax_x_slider, "X", self.x_min, self.x_max,
                               valinit=self.x_coord, valstep=0.05)
        self.slider_x.on_changed(self._on_x_changed)

        ax_y_slider = self.fig.add_axes([slider_left, 0.19, slider_width, 0.025])
        self.slider_y = Slider(ax_y_slider, "Y", self.y_min, self.y_max,
                               valinit=self.y_coord, valstep=0.05)
        self.slider_y.on_changed(self._on_y_changed)

        ax_z_slider = self.fig.add_axes([slider_left, 0.16, slider_width, 0.025])
        self.slider_z = Slider(ax_z_slider, "Z", self.z_min, self.z_max,
                               valinit=self.z_coord, valstep=0.05)
        self.slider_z.on_changed(self._on_z_changed)

        ax_r_slider = self.fig.add_axes([slider_left, 0.13, slider_width, 0.025])
        self.slider_radius = Slider(ax_r_slider, "Radius", 0.02, 0.40,
                                    valinit=self.radius, valstep=0.01)
        self.slider_radius.on_changed(self._on_radius_changed)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_w = 0.13

        ax_add = self.fig.add_axes([0.15, 0.04, btn_w, 0.05])
        self.btn_add = Button(ax_add, "Add Sphere")
        self.btn_add.on_clicked(self._on_add)

        ax_remove = self.fig.add_axes([0.32, 0.04, btn_w, 0.05])
        self.btn_remove = Button(ax_remove, "Remove Last")
        self.btn_remove.on_clicked(self._on_undo)

        ax_clear = self.fig.add_axes([0.49, 0.04, btn_w, 0.05])
        self.btn_clear = Button(ax_clear, "Clear All")
        self.btn_clear.on_clicked(self._on_clear)

        ax_done = self.fig.add_axes([0.72, 0.04, btn_w, 0.05])
        self.btn_done = Button(ax_done, "Done ✓")
        self.btn_done.on_clicked(self._on_done)

        # Connect keyboard
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.fig.suptitle(
            "3D Obstacle Editor — Use X/Y/Z sliders to position, click Add Sphere | Enter: done",
            fontsize=10, y=0.99,
        )

        plt.show(block=True)
        return self.modifications

    # ── Drawing ──────────────────────────────────────────────────────────

    def _redraw(self):
        """Redraw the 3D environment and all overlays."""
        self.ax.clear()
        self.env.render(self.ax)

        # Draw the existing reference path in task space
        if self._path_pos_3d is not None:
            pts = self._path_pos_3d  # (N, n_links, 3)
            if pts.ndim == 3:
                # Use the last link (end-effector) position
                ee = pts[:, -1, :]  # (N, 3)
                self.ax.plot(ee[:, 0], ee[:, 1], ee[:, 2],
                             color="blue", linewidth=2.0, alpha=0.7, label="Existing path")

        # Draw start configuration via robot FK
        if self._start_pos_3d is not None:
            pts = self._start_pos_3d  # (n_links, 3)
            if pts.ndim == 2:
                self.ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                                c="green", s=120, marker="o", zorder=10,
                                edgecolors="black", linewidths=1.5, label="Start")

        # Draw goal configuration via robot FK
        if self._goal_pos_3d is not None:
            pts = self._goal_pos_3d  # (n_links, 3)
            if pts.ndim == 2:
                self.ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                                c="red", s=120, marker="*", zorder=10,
                                edgecolors="black", linewidths=1.5, label="Goal")

        # Draw a crosshair / preview sphere at current slider position
        preview_center = [self.x_coord, self.y_coord, self.z_coord]
        _plot_wireframe_sphere(self.ax, preview_center, self.radius,
                               color="lime", alpha=0.15)
        self.ax.scatter(*preview_center, c="lime", s=30, marker="+", zorder=15)

        n_mods = len(self.modifications)
        self.ax.set_title(
            f"Modifications: {n_mods}  |  Preview: ({self.x_coord:.2f}, "
            f"{self.y_coord:.2f}, {self.z_coord:.2f}), r={self.radius:.2f}",
            fontsize=10,
        )

        if self._start_pos_3d is not None or self._goal_pos_3d is not None:
            self.ax.legend(loc="upper left", fontsize=8)

        self.fig.canvas.draw_idle()

    # ── Event handlers ───────────────────────────────────────────────────

    def _on_key(self, event):
        if event.key == "enter":
            self._on_done(None)
        elif event.key == "u":
            self._on_undo(None)

    def _on_x_changed(self, val):
        self.x_coord = val
        self._redraw()

    def _on_y_changed(self, val):
        self.y_coord = val
        self._redraw()

    def _on_z_changed(self, val):
        self.z_coord = val
        self._redraw()

    def _on_radius_changed(self, val):
        self.radius = val
        self._redraw()

    def _on_add(self, event):
        """Add a sphere at the current slider coordinates."""
        self._add_obstacle(self.x_coord, self.y_coord, self.z_coord)
        self._redraw()

    def _on_undo(self, event):
        if not self.modifications:
            return
        last = self.modifications.pop()
        last_obj = self._added_objects.pop() if self._added_objects else None

        if last["type"].startswith("add") and last_obj is not None:
            if last_obj in self.env.obj_extra_list:
                self.env.obj_extra_list.remove(last_obj)
                self.env.update_objects_extra()

        self._redraw()

    def _on_clear(self, event):
        while self.modifications:
            self._on_undo(None)
        self._redraw()

    def _on_done(self, event):
        self._done = True
        plt.close(self.fig)

    # ── Obstacle manipulation ────────────────────────────────────────────

    def _add_obstacle(self, x, y, z):
        """Add a sphere obstacle at (x, y, z)."""
        from mpd.utils.obstacle_editing import add_sphere_obstacle

        center = np.array([x, y, z])
        obj = add_sphere_obstacle(self.env, center, self.radius,
                                  tensor_args=self.tensor_args)
        self.modifications.append({
            "type": "add_sphere",
            "center": center.tolist(),
            "radius": float(self.radius),
        })
        self._added_objects.append(obj)
        print(f"  Added sphere at ({x:.2f}, {y:.2f}, {z:.2f}) r={self.radius:.2f}")
