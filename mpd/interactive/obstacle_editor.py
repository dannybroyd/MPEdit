"""
Interactive obstacle editor using matplotlib.

Launch a GUI window where the user can:
  - Left-click to add a sphere obstacle at the click position
  - Right-click to remove the nearest extra obstacle
  - Use a slider to adjust obstacle radius before placing
  - Toggle between sphere and box obstacle types
  - Press 'u' to undo the last action
  - Press Enter or click "Done" to accept and return the modification list

Usage::

    from mpd.interactive.obstacle_editor import ObstacleEditor

    editor = ObstacleEditor(env, tensor_args=tensor_args)
    modifications = editor.run()
    # modifications: list of dicts, e.g.
    # [{"type": "add_sphere", "center": [0.2, 0.3], "radius": 0.1}, ...]
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Circle, Rectangle

from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch, DEFAULT_TENSOR_ARGS


class ObstacleEditor:
    """Interactive matplotlib-based obstacle editor for 2D environments."""

    def __init__(
        self,
        env,
        tensor_args=DEFAULT_TENSOR_ARGS,
        initial_radius=0.08,
        initial_box_size=0.10,
        existing_path=None,
        q_pos_start=None,
        q_pos_goal=None,
        figsize=(10, 10),
    ):
        """
        Args:
            env: environment instance with render(ax), add_objects_extra(), etc.
            tensor_args: device/dtype for creating torch obstacles
            initial_radius: default radius for sphere obstacles
            initial_box_size: default half-extent for box obstacles
            existing_path: optional (N, 2) path to display as reference
            q_pos_start: optional (2,) start position to display
            q_pos_goal: optional (2,) goal position to display
            figsize: figure size
        """
        self.env = env
        self.tensor_args = tensor_args
        self.radius = initial_radius
        self.box_size = initial_box_size
        self.existing_path = existing_path
        self.q_pos_start = to_numpy(q_pos_start) if q_pos_start is not None else None
        self.q_pos_goal = to_numpy(q_pos_goal) if q_pos_goal is not None else None
        self.figsize = figsize

        # State
        self.modifications = []  # list of dicts describing each action
        self._added_objects = []  # parallel list of ObjectField refs for undo
        self.obstacle_type = "sphere"  # "sphere" or "box"
        self._done = False

    def run(self):
        """
        Open the interactive editor window. Blocks until user clicks Done or presses Enter.

        Returns:
            list of modification dicts, e.g.:
            [{"type": "add_sphere", "center": [x, y], "radius": r}, ...]
        """
        # Use an interactive backend
        backend = matplotlib.get_backend()
        if backend == "agg" or backend == "Agg":
            matplotlib.use("TkAgg")
            plt.switch_backend("TkAgg")

        self.fig = plt.figure(figsize=self.figsize)

        # Main plot area (leave room at bottom for widgets)
        self.ax = self.fig.add_axes([0.08, 0.20, 0.84, 0.72])
        self._redraw()

        # ── Widgets ─────────────────────────────────────────────────────
        # Radius slider
        ax_slider = self.fig.add_axes([0.20, 0.10, 0.55, 0.03])
        self.slider_radius = Slider(
            ax_slider, "Size", 0.02, 0.30, valinit=self.radius, valstep=0.01,
        )
        self.slider_radius.on_changed(self._on_slider_changed)

        # Obstacle type radio buttons
        ax_radio = self.fig.add_axes([0.02, 0.01, 0.12, 0.08])
        self.radio = RadioButtons(ax_radio, ("sphere", "box"), active=0)
        self.radio.on_clicked(self._on_radio_changed)

        # Undo button
        ax_undo = self.fig.add_axes([0.55, 0.02, 0.10, 0.05])
        self.btn_undo = Button(ax_undo, "Undo")
        self.btn_undo.on_clicked(self._on_undo)

        # Clear button
        ax_clear = self.fig.add_axes([0.67, 0.02, 0.10, 0.05])
        self.btn_clear = Button(ax_clear, "Clear All")
        self.btn_clear.on_clicked(self._on_clear)

        # Done button
        ax_done = self.fig.add_axes([0.82, 0.02, 0.12, 0.05])
        self.btn_done = Button(ax_done, "Done ✓")
        self.btn_done.on_clicked(self._on_done)

        # Connect mouse and keyboard events
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.fig.suptitle(
            "Obstacle Editor — Left-click: add | Right-click: remove nearest | U: undo | Enter: done",
            fontsize=11, y=0.97,
        )

        plt.show(block=True)

        return self.modifications

    # ── Drawing ──────────────────────────────────────────────────────────

    def _redraw(self):
        """Redraw the environment and all overlays."""
        self.ax.clear()
        self.env.render(self.ax)

        # Draw existing reference path
        if self.existing_path is not None:
            path_np = to_numpy(self.existing_path) if not isinstance(self.existing_path, np.ndarray) else self.existing_path
            self.ax.plot(
                path_np[:, 0], path_np[:, 1],
                color="blue", linewidth=2.0, alpha=0.6, linestyle="--",
                label="Current Path", zorder=5,
            )

        # Start/goal
        if self.q_pos_start is not None:
            self.ax.scatter(
                self.q_pos_start[0], self.q_pos_start[1],
                color="blue", marker="X", s=200, zorder=100, label="Start",
            )
        if self.q_pos_goal is not None:
            self.ax.scatter(
                self.q_pos_goal[0], self.q_pos_goal[1],
                color="red", marker="X", s=200, zorder=100, label="Goal",
            )

        # Show a "ghost" preview of what would be placed
        if self.obstacle_type == "sphere":
            self.ax.set_xlabel(f"Sphere mode (r={self.radius:.2f})")
        else:
            self.ax.set_xlabel(f"Box mode (half-size={self.box_size:.2f})")

        n_mods = len(self.modifications)
        self.ax.set_title(f"Modifications: {n_mods}", fontsize=11)

        if self.existing_path is not None or self.q_pos_start is not None:
            self.ax.legend(loc="upper left", fontsize=9)

        self.fig.canvas.draw_idle()

    # ── Event handlers ───────────────────────────────────────────────────

    def _on_click(self, event):
        """Handle mouse clicks on the main axes."""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if event.button == 1:  # Left click — add obstacle
            self._add_obstacle(x, y)
        elif event.button == 3:  # Right click — remove nearest extra obstacle
            self._remove_nearest(x, y)

        self._redraw()

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == "enter":
            self._on_done(None)
        elif event.key == "u":
            self._on_undo(None)

    def _on_slider_changed(self, val):
        self.radius = val
        self.box_size = val
        if self.obstacle_type == "sphere":
            self.ax.set_xlabel(f"Sphere mode (r={self.radius:.2f})")
        else:
            self.ax.set_xlabel(f"Box mode (half-size={self.box_size:.2f})")
        self.fig.canvas.draw_idle()

    def _on_radio_changed(self, label):
        self.obstacle_type = label

    def _on_undo(self, event):
        if not self.modifications:
            return
        last = self.modifications.pop()
        last_obj = self._added_objects.pop() if self._added_objects else None

        if last["type"].startswith("add") and last_obj is not None:
            # Remove the object from env
            if last_obj in self.env.obj_extra_list:
                self.env.obj_extra_list.remove(last_obj)
                self.env.update_objects_extra()
        elif last["type"] == "remove_sphere":
            # Re-add the removed obstacle (we stored its data)
            from torch_robotics.environments.primitives import MultiSphereField, ObjectField
            center = np.array([last["removed_center"]])
            radius = np.array([last["removed_radius"]])
            centers_t = to_torch(center, **self.tensor_args)
            radii_t = to_torch(radius, **self.tensor_args)
            # Insert back into the fixed object's sphere field
            obj_fixed = self.env.obj_fixed_list[0]
            for primitive in obj_fixed.fields:
                if hasattr(primitive, "centers"):
                    import torch
                    primitive.centers = torch.cat([primitive.centers, centers_t], dim=0)
                    primitive.radii = torch.cat([primitive.radii, radii_t], dim=0)
                    break
            self.env.build_sdf_grid(compute_sdf_obj_fixed=True, compute_sdf_obj_extra=True)

        self._redraw()

    def _on_clear(self, event):
        """Remove all extra obstacles added in this session."""
        # Undo all additions
        while self.modifications:
            self._on_undo(None)
        self._redraw()

    def _on_done(self, event):
        self._done = True
        plt.close(self.fig)

    # ── Obstacle manipulation ────────────────────────────────────────────

    def _add_obstacle(self, x, y):
        """Add an obstacle at position (x, y)."""
        from mpd.utils.obstacle_editing import add_sphere_obstacle, add_box_obstacle

        center = np.array([x, y])
        if self.obstacle_type == "sphere":
            obj = add_sphere_obstacle(self.env, center, self.radius, tensor_args=self.tensor_args)
            self.modifications.append({
                "type": "add_sphere",
                "center": center.tolist(),
                "radius": float(self.radius),
            })
        else:
            sizes = np.array([self.box_size, self.box_size])
            obj = add_box_obstacle(self.env, center, sizes, tensor_args=self.tensor_args)
            self.modifications.append({
                "type": "add_box",
                "center": center.tolist(),
                "sizes": sizes.tolist(),
            })
        self._added_objects.append(obj)

    def _remove_nearest(self, x, y):
        """Remove the fixed obstacle nearest to (x, y)."""
        # First check extra obstacles (user-added)
        if self.env.obj_extra_list:
            best_dist = float("inf")
            best_idx = -1
            for i, obj in enumerate(self.env.obj_extra_list):
                for field in obj.fields:
                    if hasattr(field, "centers"):
                        centers_np = to_numpy(field.centers)
                        for c in centers_np:
                            d = np.linalg.norm(c[:2] - np.array([x, y]))
                            if d < best_dist:
                                best_dist = d
                                best_idx = i

            if best_idx >= 0 and best_dist < 0.3:
                removed_obj = self.env.obj_extra_list.pop(best_idx)
                self.env.update_objects_extra()
                # Find and remove from our tracking
                if removed_obj in self._added_objects:
                    idx_in_added = self._added_objects.index(removed_obj)
                    self._added_objects.pop(idx_in_added)
                    self.modifications.pop(idx_in_added)
                self._redraw()
                return

        # Then check fixed obstacles
        obj_fixed = self.env.obj_fixed_list[0] if self.env.obj_fixed_list else None
        if obj_fixed is None:
            return

        for primitive in obj_fixed.fields:
            if hasattr(primitive, "centers"):
                centers_np = to_numpy(primitive.centers)
                dists = np.linalg.norm(centers_np[:, :2] - np.array([[x, y]]), axis=1)
                nearest_idx = int(np.argmin(dists))

                if dists[nearest_idx] > 0.3:
                    return  # Too far, no removal

                # Store removal data for undo
                removed_center = centers_np[nearest_idx].tolist()
                removed_radius = float(to_numpy(primitive.radii)[nearest_idx])

                from mpd.utils.obstacle_editing import remove_fixed_obstacle_by_index
                remove_fixed_obstacle_by_index(self.env, nearest_idx, tensor_args=self.tensor_args)

                self.modifications.append({
                    "type": "remove_sphere",
                    "index": nearest_idx,
                    "removed_center": removed_center,
                    "removed_radius": removed_radius,
                })
                self._added_objects.append(None)  # placeholder for undo tracking
                break
