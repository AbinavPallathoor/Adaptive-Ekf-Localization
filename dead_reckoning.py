"""Wheel-encoder dead reckoning (odometry) for the differential-drive robot.

Integrates wheel-angle changes from a known start pose. Position and heading
come from the encoders alone, so errors accumulate without correction. The
pose is that of the axle midpoint, which stays put when the robot turns in place.
"""

import math

# Geometry from models/differential_drive.xml.
WHEEL_RADIUS = .06
TRACK = .30


def wheel_motion(left, right):
    """Distance (m) and heading change (rad) from left/right wheel travel (m)."""
    return (left + right)/2, (right - left)/TRACK


class DeadReckoning:
    def __init__(self, x, y, heading, left_angle, right_angle):
        """Start at a known axle-midpoint pose (m, m, rad) and wheel angles (rad)."""
        self.x, self.y, self.heading = x, y, heading
        self.left_angle, self.right_angle = left_angle, right_angle

    def update(self, left_angle, right_angle):
        """Advance by the wheel-angle change since the last update."""
        left = WHEEL_RADIUS*(left_angle - self.left_angle)
        right = WHEEL_RADIUS*(right_angle - self.right_angle)
        self.left_angle, self.right_angle = left_angle, right_angle
        distance, turn = wheel_motion(left, right)
        # Midpoint heading: exact for straight lines, second-order for arcs.
        middle = self.heading + turn/2
        self.x += distance*math.cos(middle)
        self.y += distance*math.sin(middle)
        self.heading += turn
