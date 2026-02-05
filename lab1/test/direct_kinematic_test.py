import numpy as np
import pytest
import sys
import os

# Allow import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from lab1.src.direct_kinematics import direct_kinematics_position

@pytest.mark.parametrize(
    "joint_angles, position_expected, euler_expected",
    [
        (
            [0, 0, 0, 0, 0, 0],
            [190, 0, 308],
            [0, 90, 0],
        ),
        (
            [10, 0, 15, 0, -20, -50],
            [192.51, 33.945, 281.748],
            [-63.26, 78.831, 13.697],
        ),
        (
            [15, 16, -14, 7, 37, 50],
            [204.293, 60.055, 254.821],
            [-156.474, 47.040, -151.773],
        ),
        (
            [52, -13, -35, -38, 76, 91],
            [77.768, 31.618, 357.916],
            [-147.977, 66.959, -149.095],
        ),
        (
            [-55, -1, 27, -72, -59, 58],
            [140.060, -100.536, 252.390],
            [28.678, 89.197, -29.425],
        ),
        (
            [-55.008, -0.939, 27.644, -72.537, -59.184, 58.447],
            [139.999, -99.999, 250.002],
            [-19.598, 89.999, 19.597],
        ),
        (
            [46.287, 6.624, -5.171, 39.165, -59.322, 73.873],
            [147.299, 99.049, 349.801],
            [-0.485, 49.179, 131.053],
        ),
    ],
)
def test_direct_kinematics(joint_angles: list, position_expected: list, euler_expected: list):
    position, euler_angles = direct_kinematics_position(joint_angles)

    # Check position
    assert np.allclose(position, position_expected, atol=1e-3)

    # Check Euler angles
    # Near representation singularity (89˚ < |β| < 90°), alpha and gamma are ill-conditioned
    if 89.0 < abs(euler_expected[1]) < 90.0:
        # Alpha (check with bigger tolerance)
        assert np.isclose(euler_angles[0], euler_expected[0], atol=0.5)
        # Beta (Unchanged)
        assert np.isclose(euler_angles[1], euler_expected[1], atol=1e-3)
        # Gamma (check with bigger tolerance)
        assert np.isclose(euler_angles[2], euler_expected[2], atol=0.5)
    else:
        assert np.allclose(euler_angles, euler_expected, atol=1e-3)
