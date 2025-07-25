# Classes for calculating the physical forces/disturbances applied to agents in the sim
import numpy as np
import math
from ..agent_class import Agent
from .. import sim_functions


class WindField:
    """Wind field generator with spatial and temporal variations"""

    def __init__(
        self,
        base_wind_speed=5.0,
        base_wind_direction=0.0,
        turbulence_intensity=0.05,
        temporal_frequency=10.0,
        rng=np.random.default_rng(),
    ):
        """
        Initialize wind field with advanced turbulence modeling

        Args:
            base_wind_speed: Base wind speed in m/s
            base_wind_direction: Base wind direction in degrees NED convention
            turbulence_intensity: Standard deviation in speed and direction of wind
            temporal_frequency: Rate of change in wind speed/direction in seconds
            rng: Random number generator
        """
        self.base_wind_speed = base_wind_speed  # m/s
        self.base_wind_direction = base_wind_direction  # degrees
        self.base_wind = vector_from_components(
            self.base_wind_speed, self.base_wind_direction
        )
        self.turbulence_intensity = turbulence_intensity
        self.temporal_frequency = temporal_frequency  # seconds
        self.rng = rng

    def get_wind_at_position(self, position, time):
        """
        Get wind vector at a specific position and time

        Args:
            position: Position [x, y] or [x, y, z] in meters
            time: Current simulation time in seconds

        Returns:
            wind_vector: np.array([x, y]) in m/s
        """
        # TODO: return no wind if agent below Z surface level?
        # if agent.pos[2] < 0:
        #   return np.array([0.0, 0.0]) # no wind force applied to agents below sea level

        # Add spatial variations using vortex field
        # TODO: add vortex component using sim_class.py:VortexField:current_vortex_calculate(agent)
        turbulent_wind = np.array([0.0, 0.0])

        # Add temporal noise
        temporal_noise = self.rng.normal(0, self.turbulence_intensity, 2)
        temporal_noise *= np.sin(2 * np.pi * time / (self.temporal_frequency * 0.3))

        total_wind = self.base_wind + turbulent_wind + temporal_noise

        return total_wind


def vector_from_components(magnitude, direction):
    """Convert magnitude and direction (degrees) of a force to a vector"""
    direction_rad = np.deg2rad(direction)
    return magnitude * np.array(
        [np.cos(direction_rad), np.sin(direction_rad)]  # X component  # Y component
    )


# def apply_force(self, agent: Agent):
#         """apply space and time-dependent force to an agent based on local wind current"""
#         if agent.pos[2] < 0:
#             return  # no wind force applied to agents below sea level

#         agent.cmd_forces = (
#             self.global_current_vector
#         )  # Todo assuming nothing else writes to force

#         # Formula from https://www.spinnakersailing.com/apparent-wind/
#         angle_of_attack = (agent.psi - self.direction) % 180
#         apparent_wind_speed = math.sqrt((self.magnitude ** 2)
#                                         + (agent.incurrent_velocity ** 2)
#                                         + (2 * self.magnitude * agent.incurrent_velocity * math.cos(math.radians(self.direction))))
#         # Formula from https://pubs.aip.org/physicstoday/article/61/2/38/413188/The-physics-of-sailingSails-and-keels-like
#         f = agent.sail_area * (apparent_wind_speed ** 2)
#         drag = agent.surface_area * (agent.incurrent_velocity ** 2)

if __name__ == "__main__":
    pass
