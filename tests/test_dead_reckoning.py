from pathlib import Path
import math
import sys

import mujoco
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dead_reckoning as dr
import robot


def test_straight_line_moves_by_wheel_arc():
    odometry = dr.DeadReckoning(0, 0, 0, 0, 0)
    odometry.update(10, 10)
    assert odometry.x == pytest.approx(dr.WHEEL_RADIUS * 10)
    assert odometry.y == pytest.approx(0)
    assert odometry.heading == pytest.approx(0)


def test_turn_in_place_keeps_position():
    odometry = dr.DeadReckoning(0, 0, 0, 0, 0)
    quarter = (math.pi/2) * dr.TRACK / (2*dr.WHEEL_RADIUS)
    odometry.update(-quarter, quarter)
    assert odometry.heading == pytest.approx(math.pi/2)
    assert odometry.x == pytest.approx(0)
    assert odometry.y == pytest.approx(0)


def test_constants_match_model():
    model = mujoco.MjModel.from_xml_path(str(robot.MODEL))
    left = model.body('left_wheel')
    right = model.body('right_wheel')
    assert model.geom_size[left.geomadr[0]][0] == pytest.approx(dr.WHEEL_RADIUS)
    assert left.pos[1] - right.pos[1] == pytest.approx(dr.TRACK)
    # The tracked point is the axle midpoint, where the robot turns in place.
    axle = model.site('axle')
    assert axle.pos[0] == pytest.approx(left.pos[0])
    assert axle.pos[1] == pytest.approx((left.pos[1] + right.pos[1])/2)


def test_true_pose_barely_moves_when_turning_in_place():
    model, data = robot.load()
    start = robot.true_pose(data)
    robot.apply_target(model, data, (-3, 3), round(1.43/model.opt.timestep))
    robot.apply_target(model, data, (0, 0), 500)
    end = robot.true_pose(data)
    assert end[2] > 1.3
    assert math.dist(start[:2], end[:2]) < .01


def test_geometry_matches_robot_module():
    assert (robot.WHEEL_RADIUS, robot.TRACK) == (dr.WHEEL_RADIUS, dr.TRACK)
