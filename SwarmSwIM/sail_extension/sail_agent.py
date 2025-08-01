import numpy as np
from SwarmSwIM.agent_class import Agent
from .physics import WindField


class SailAgent(Agent):
    """Extended class for wind propelled agents"""

    def __init__(
        self,
        name,
        Dt=0.1,
        initialPosition=np.array([0.0, 0.0, 0.0]),
        initialHeading=0.0,
        agent_xml="sail_extension/sail_agent_default.xml",
        rng=None,
    ):

        super().__init__(name, Dt, initialPosition, initialHeading, agent_xml, rng)

        # Sailing-specific parameters
        # TODO: add to xml instead of hardcode
        self.sail_angle = 0.0  # degrees relative to vessel centerline

        # Command initilization
        self.cmd_sail_angle = 0

    def calculate_speed(self, true_wind):
        """
        Calculate the speed and turn rate of sailboat based on angle to wind

        Args:
            true_wind: np.array([wind_north, wind_east]) in m/s localized at agent
        """

        # TODO: make better formula accounting for sail angle, sail area, drag, etc.
        base_speed = 2.0  # TODO: replace with sail area mod

        # Get wind magnitude and direction
        wind_magnitude = np.linalg.norm(true_wind)
        wind_direction = np.rad2deg(np.arctan2(true_wind[1], true_wind[0]))

        # Scale speed with wind direction
        relative_wind_angle = wind_direction - self.psi
        if relative_wind_angle > 180:
            relative_wind_angle -= 360
        if relative_wind_angle < -180:
            relative_wind_angle += 360

        # Speed up on downwind, slow on upwind
        speed_mult_from_wind = np.cos(np.deg2rad(relative_wind_angle))

        effective_speed = (
            base_speed * speed_mult_from_wind
        ) + 0.2  # Add base speed to prevent getting stuck

        # Update agent pos
        vel_x = effective_speed * np.cos(np.deg2rad(self.psi))
        vel_y = effective_speed * np.sin(np.deg2rad(self.psi))

        self.pos[0] += vel_x * self.Dt
        self.pos[1] += vel_y * self.Dt

        # TODO other way to update velocity other than manual position? inertial? step? ideal? local?
        # self.cmd_local_vel = np.array([vel_x, vel_y])

        # TURN RATE
        # TODO: no apparent effect
        # self.cmd_yawrate = wind_magnitude * 0.5

    def calculate_turn_rate(self, true_wind):
        """
        Calculate the effective turn rate of the sailboat based on angle to wind & agent velocity

        Args:
            true_wind: np.array([wind_north, wind_east]) in m/s
        """
        # TODO: make better formula
        # Find or add velocity attribute (not incurrent_vel?)
        # Add rudder authority config?
        # Add velocity drag penalty from turning?
        return

    @property
    def cmd_sail_angle(self):
        """
        Set sail angle relative to vessel centerline

        Args:
            angle_degrees: sail angle in degrees (-180 to +180)
        """
        return self.sail_angle

    @cmd_sail_angle.setter
    def cmd_sail_angle(self, input):
        self.sail_angle = np.clip(input, -180, 180)


# python3 -m SwarmSwIM.SwarmSwIM.sail_extension.sail_agent
if __name__ == "__main__":
    pass
