import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from ..animator2D import Plotter
from .physics import WindField


class WindVisualization:
    """Wind visualization for Plotter class"""

    def __init__(self, wind_field, grid_size=5, arrow_scale=1.0, arrow_color="blue"):
        """
        Initialize wind visualization

        Args:
            wind_field: WindField instance to visualize
            grid_size: Spacing between wind arrows in the grid
            arrow_scale: Scale factor for arrow size
            arrow_color: Color for all wind arrows
        """
        self.wind_field = wind_field
        self.grid_size = grid_size
        self.arrow_scale = arrow_scale
        self.arrow_color = arrow_color
        self.wind_arrows = []
        self.time = 0.0

    def create_wind_grid(self, ax, x_range, y_range):
        """Create a grid of wind arrows on the given axes"""
        for arrow in self.wind_arrows:
            arrow.remove()
        self.wind_arrows.clear()

        x_points = np.arange(x_range[0], x_range[1], self.grid_size)
        y_points = np.arange(y_range[0], y_range[1], self.grid_size)

        for x in x_points:
            for y in y_points:
                wind_vector = self.wind_field.get_wind_at_position([x, y, 0], self.time)

                if np.linalg.norm(wind_vector) > 0.1:
                    magnitude = np.linalg.norm(wind_vector)
                    direction = np.arctan2(wind_vector[1], wind_vector[0])

                    arrow_length = magnitude * self.arrow_scale * 0.4
                    dx = arrow_length * np.cos(direction)
                    dy = arrow_length * np.sin(direction)

                    head_width = max(0.2, arrow_length * 0.1)
                    head_length = max(0.15, arrow_length * 0.08)

                    arrow = ax.arrow(
                        x,
                        y,
                        dx,
                        dy,
                        head_width=head_width,
                        head_length=head_length,
                        fc=self.arrow_color,
                        ec=self.arrow_color,
                        alpha=0.7,
                        zorder=1,
                    )
                    self.wind_arrows.append(arrow)

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
        wind_grid_size=5,
    ):
        """
        Initialize WindPlotter with wind visualization

        Args:
            simulator: Simulation instance
            wind_field: WindField instance for visualization
            SIZE: Plot size limit
            artistics: Additional artistic elements
            show_wind: Whether to show wind visualization
            wind_grid_size: Spacing between wind arrows
        """
        super().__init__(simulator, SIZE, artistics)

        self.show_wind = show_wind
        self.wind_field = wind_field
        self.SIZE = SIZE

        if self.show_wind:
            self.wind_viz = WindVisualization(self.wind_field, grid_size=wind_grid_size)
            self.wind_viz.create_wind_grid(self.ax, (-SIZE, SIZE), (-SIZE, SIZE))

            self.add_wind_legend()

    def add_wind_legend(self):
        """Add a wind speed legend to the plot"""
        from matplotlib.lines import Line2D

        # Create custom legend elements for different wind speeds
        legend_elements = []
        wind_speeds = [2.5, 5.0, 10.0]  # m/s

        for speed in wind_speeds:
            line_width = max(1, speed * 0.5)  # Scale line width with speed

            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=self.wind_viz.arrow_color,
                    linewidth=line_width,
                    label=f"{speed} m/s wind",
                    marker=">",
                    markersize=8,
                    markerfacecolor=self.wind_viz.arrow_color,
                    markeredgecolor=self.wind_viz.arrow_color,
                )
            )

        self.ax.legend(
            handles=legend_elements,
            title="Wind Speed",
            loc="center left",
            bbox_to_anchor=(1.02, 0.05),
            frameon=True,
            fancybox=True,
            shadow=True,
        )

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
                # add to artists list
                artist_list.extend(
                    [
                        self.animation[agent.name]["line"],
                        self.animation[agent.name]["figure"],
                    ]
                )

            # Update wind visualization if enabled
            if self.show_wind and hasattr(self, "wind_viz"):
                # Update wind field time
                self.wind_viz.update_time(self.sim.Dt)

                # Periodically refresh wind grid (every 10 frames for performance)
                if frame % 10 == 0:
                    self.wind_viz.create_wind_grid(
                        self.ax, (-self.SIZE, self.SIZE), (-self.SIZE, self.SIZE)
                    )

                # Add wind arrows to artist list
                artist_list.extend(self.wind_viz.wind_arrows)

            self.ax.relim()
            return artist_list

        # Set up animation with the custom update function
        interval = max(1, int(self.sim.Dt * 1000))
        ani = FuncAnimation(
            self.fig2, update, frames=range(10000), interval=interval, blit=True
        )

        if callback:
            try:
                ani.event_source.add_callback(callback)
            except AttributeError:
                # Fallback if event_source is not available
                pass
        plt.show()


if __name__ == "__main__":
    pass
