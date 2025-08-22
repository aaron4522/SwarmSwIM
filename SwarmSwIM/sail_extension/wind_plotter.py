import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from ..animator2D import Plotter
from .physics import WindField, components_from_vector


class WindVisualization:
    """Wind visualization for Plotter class"""

    def __init__(
        self,
        wind_field,
        density=5,
        arrow_color="blue",
        show_contours=True,
        min_arrow_speed=1,
    ):
        """
        Initialize wind visualization

        Args:
            wind_field: WindField instance to visualize
            density: m^2 per arrow (higher = less dense)
            arrow_color: Color for all wind arrows
            show_contours: Whether to show wind speed contours
            min_arrow_size: Culls arrows below this speed from being graphed
        """
        self.wind_field = wind_field
        self.density = density
        self.arrow_color = arrow_color
        self.min_arrow_speed = min_arrow_speed
        self.show_contours = show_contours
        self.contour_collection = None
        self.quiver_plot = None
        self.colorbar = None
        self.time = 0.0
        self._prev_x_range = []
        self._prev_y_range = []
        self._prev_mask = []

    def create_wind_contour(self, ax, x_range, y_range):
        if self.contour_collection is not None:
            for coll in self.contour_collection.collections:
                coll.remove()
            self.contour_collection = None

        x_start, x_end = min(x_range), max(x_range)
        y_start, y_end = min(y_range), max(y_range)

        # Create wind speed contours if enabled
        if self.show_contours:
            width = x_end - x_start
            height = y_end - y_start
            # Create a finer grid for contours
            contour_resolution = max(20, int(min(width, height) / 2))
            x_contour = np.linspace(x_range[0], x_range[1], contour_resolution)
            y_contour = np.linspace(y_range[0], y_range[1], contour_resolution)
            X_contour, Y_contour = np.meshgrid(x_contour, y_contour)

            # Calculate wind speeds at all grid points
            wind_speeds = np.zeros_like(X_contour)
            for i in range(len(y_contour)):
                for j in range(len(x_contour)):
                    wind_vector = self.wind_field.get_wind_at_position(
                        [X_contour[i, j], Y_contour[i, j], 0], self.time
                    )
                    wind_speeds[i, j] = np.linalg.norm(wind_vector)

            # Create contour levels
            max_speed = np.max(wind_speeds)
            if max_speed > 0:
                levels = np.linspace(0, max_speed, 15)
                self.contour_collection = ax.contourf(
                    X_contour,
                    Y_contour,
                    wind_speeds,
                    levels=levels,
                    cmap="viridis",
                    alpha=0.6,
                    zorder=0,
                )

    def create_wind_grid(self, ax, x_range, y_range):
        """Create a grid of wind arrows using quiver for better performance"""
        # Create arrow grid
        x_start, x_end = min(x_range), max(x_range)
        y_start, y_end = min(y_range), max(y_range)

        x_points = np.arange(x_start, x_end, self.density)
        y_points = np.arange(y_start, y_end, self.density)

        if len(x_points) == 0 or len(y_points) == 0:
            return

        # Create meshgrid for vectorized calculations
        X, Y = np.meshgrid(x_points, y_points)

        # Calculate wind vectors
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                wind_vector = self.wind_field.get_wind_at_position(
                    [X[i, j], Y[i, j], 0], self.time
                )
                U[i, j] = wind_vector[0]
                V[i, j] = -wind_vector[1]  # Flip Y component for inverted display axis

        # Filter out very small winds
        magnitude = np.sqrt(U**2 + V**2)
        mask = magnitude >= self.min_arrow_speed
        X = X[mask]
        Y = Y[mask]
        U = U[mask]
        V = V[mask]

        # Update wind grid when axis limits change or recreate if needed
        grid_changed = (
            self.quiver_plot is None
            or not np.array_equal(self._prev_x_range, x_range)
            or not np.array_equal(self._prev_y_range, y_range)
            or not np.array_equal(self._prev_mask, mask)
        )

        if grid_changed:
            if self.quiver_plot is not None:
                self.quiver_plot.remove()

            self.quiver_plot = ax.quiver(
                X,
                Y,
                U,
                V,
                color=self.arrow_color,
                alpha=0.7,
                zorder=2,
                width=0.003,
                headwidth=3,
                headlength=4,
            )
            self._prev_x_range = x_range
            self._prev_y_range = y_range
            self._prev_mask = mask
        else:
            # User hasn't panned/zoomed the simulation, no need to redraw the entire plot--just direction & magnitude
            self.quiver_plot.set_UVC(U, V)

        return self.quiver_plot

    def add_wind_legend(self, fig, ax):
        """Add a wind speed legend and colorbar to the plot"""
        if self.show_contours and self.contour_collection is not None:
            # Remove existing colorbar if it exists
            if hasattr(self, "colorbar") and self.colorbar is not None:
                self.colorbar.remove()

            # Add colorbar for wind speed contours
            self.colorbar = fig.colorbar(
                self.contour_collection,
                ax=ax,
                orientation="vertical",
                pad=0.02,
                aspect=16,
                shrink=0.8,
            )
            self.colorbar.set_label(
                "Wind Speed (m/s)", size=12, rotation=90, labelpad=15
            )
            return self.colorbar
        return None

    def update_time(self, dt):
        self.time += dt


class WindPlotter(Plotter):
    """Extended Plotter class with wind visualization capabilities"""

    def __init__(
        self,
        simulator,
        wind_field=WindField(),
        SIZE=30,
        artistics=[],
        show_wind=True,
        wind_grid_density=3,
        show_waypoints=True,
        show_wind_contours=True,
        wind_update_interval=10,
    ):
        """
        Initialize WindPlotter with wind visualization

        Args:
            simulator: Simulation instance
            wind_field: WindField instance for visualization
            SIZE: Plot size limit
            artistics: Additional artistic elements
            show_wind: Whether to show wind visualization
            wind_grid_density: Spacing between wind arrows
            show_waypoints: Whether to show waypoints
            show_wind_contours: Whether to show wind speed contours
            wind_update_interval: Number of frames between wind visualization updates
        """
        super().__init__(simulator, SIZE, artistics)

        self.show_wind = show_wind
        self.show_waypoints = show_waypoints
        self.show_wind_contours = show_wind_contours
        self.wind_field = wind_field
        self.wind_update_interval = wind_update_interval

        if self.show_wind:
            self.wind_viz = WindVisualization(
                self.wind_field,
                density=wind_grid_density,
                show_contours=show_wind_contours,
            )
            # Initialize with default bounds, will be updated dynamically
            self.wind_viz.create_wind_grid(self.ax, (-SIZE, SIZE), (-SIZE, SIZE))

    def check_waypoints(self):
        """add or remove additional waypoints with ongoing simulation"""
        # Add new waypoints
        for waypoint in self.sim.waypoints:
            if not waypoint["name"] in self.animation:
                self.add_waypoint(waypoint)

        # Remove old waypoints no longer referenced in simulation
        temporary_namelist = [wp["name"] for wp in self.sim.waypoints]
        keys_to_remove = []
        for key in self.animation:
            if not key in temporary_namelist:
                keys_to_remove.append(key)
        self.detections = {
            key: self.animation[key]
            for key in self.animation
            if key not in keys_to_remove
        }

    def add_waypoint(self, waypoint):
        """Add a waypoint to the animation"""
        waypoint_marker = plt.Circle(
            (waypoint["x"], waypoint["y"]),
            radius=0.5,
            color="orange",
            fill=True,
            alpha=1.0,
            zorder=10,
        )
        self.ax.add_patch(waypoint_marker)

        waypoint_label = self.ax.text(
            waypoint["x"],
            waypoint["y"] + 2,
            waypoint["name"],
            ha="center",
            va="bottom",
            color="orange",
            fontweight="bold",
            zorder=11,
        )
        self.animation[waypoint["name"]] = {
            "marker": waypoint_marker,
            "label": waypoint_label,
        }

    def draw_agent_label(self, agent):
        """Create a label showing agent speed, heading, and position that follows each agent"""
        # label_text = f"{agent.name}: {agent.vel:.1f}m/s | H:{agent.psi:.0f}°"
        label_text = f"{agent}"

        # Offset text from agent's current position
        x = agent.pos[0]
        y = agent.pos[1] + 1

        agent_stats = self.ax.text(
            x,
            y,
            label_text,
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9),
            horizontalalignment="center",
            verticalalignment="top",
            linespacing=0.5,
        )

        self.animation[agent.name]["legend"] = agent_stats

    # TODO: better way? super()?
    def update_plot(self, callback=None):
        """Override with wind visualization updates"""

        def update(frame):
            self.check_agents()
            artist_list = []

            # Update agents (from parent class logic)
            for agent in self.sim.agents:
                # add position to list
                self.animation[agent.name]["x"] = np.append(
                    self.animation[agent.name]["x"], agent.pos[0]
                )
                self.animation[agent.name]["y"] = np.append(
                    self.animation[agent.name]["y"], agent.pos[1]
                )
                # Pop excess
                if len(self.animation[agent.name]["x"]) > 300:
                    self.animation[agent.name]["x"] = np.delete(
                        self.animation[agent.name]["x"], 0
                    )
                if len(self.animation[agent.name]["y"]) > 300:
                    self.animation[agent.name]["y"] = np.delete(
                        self.animation[agent.name]["y"], 0
                    )
                # Update the plot lines paths
                self.animation[agent.name]["line"].set_data(
                    self.animation[agent.name]["x"], self.animation[agent.name]["y"]
                )
                # Update the polygon coordinates
                pts = self.calculate_triangle(agent)
                self.animation[agent.name]["figure"].set_xy(pts)

                # Update agent legend
                self.draw_agent_label(agent)

                # add to artists list
                artist_list.extend(
                    [
                        self.animation[agent.name]["line"],
                        self.animation[agent.name]["figure"],
                        self.animation[agent.name]["legend"],
                    ]
                )

            # Update wind visualization if enabled
            if self.show_wind:
                self.wind_viz.update_time(self.sim.Dt)

                # Refresh wind grid and contours less frequently for better performance
                if frame % self.wind_update_interval == 0:
                    # Update wind grid for time-varying wind
                    self.wind_viz.create_wind_grid(
                        self.ax,
                        np.round(self.ax.get_xlim()),
                        np.round(self.ax.get_ylim()),
                    )

                    if self.show_wind_contours:
                        self.wind_viz.create_wind_contour(
                            self.ax,
                            np.round(self.ax.get_xlim()),
                            np.round(self.ax.get_ylim()),
                        )
                        self.wind_viz.add_wind_legend(self.fig2, self.ax)

                # Always add existing wind grid to artist list for persistence
                if self.wind_viz.quiver_plot is not None:
                    artist_list.append(self.wind_viz.quiver_plot)

                # Always add existing contour collection to artist list for persistence
                if self.wind_viz.contour_collection is not None and hasattr(
                    self.wind_viz.contour_collection, "collections"
                ):
                    artist_list.extend(self.wind_viz.contour_collection.collections)

            if self.show_waypoints:
                self.check_waypoints()
                for waypoint in self.sim.waypoints:
                    artist_list.extend(
                        [
                            self.animation[waypoint["name"]]["marker"],
                            self.animation[waypoint["name"]]["label"],
                        ]
                    )

            self.ax.relim()
            return artist_list

        # get interval for real-time
        interval = max(1, int(self.sim.Dt * 1000))
        # return self.lines_list #, p
        ani = FuncAnimation(
            self.fig2, update, frames=range(10000), interval=interval, blit=True
        )

        if callback:
            ani.event_source.add_callback(callback)
        plt.show()


if __name__ == "__main__":
    pass
