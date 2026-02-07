import numpy as np
import pytest

from lab1.src.forward_kinematics import forward_kinematics_position
from lab1.src.inverse_kinematics import inverse_kinematics

@pytest.mark.parametrize(
    "joint_angles_expected, position, euler",
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
def test_inverse_kinematics(joint_angles_expected: list, position: list, euler: list):
    solution = inverse_kinematics(x=position[0], y=position[1], z=position[2],
                                  alpha=euler[0], beta=euler[1], gamma=euler[2],
                                  verbose=False)

    assert solution is not None, "No solution found by inverse kinematics."

    resulting_pos, resulting_eul = forward_kinematics_position(solution)

    # Check position
    assert np.allclose(resulting_pos, position, atol=1e-2)

    # Near representation singularity (89˚ < |β| < 90°), alpha and gamma are ill-conditioned
    if 89.0 < abs(euler[1]) < 90.0:
        # Alpha (check with bigger tolerance)
        assert np.isclose(resulting_eul[0], euler[0], atol=0.5)
        # Beta (Unchanged)
        assert np.isclose(resulting_eul[1], euler[1], atol=0.1)
        # Gamma (check with bigger tolerance)
        assert np.isclose(resulting_eul[2], euler[2], atol=0.5)
    else:
        assert np.allclose(resulting_eul, euler, atol=0.1)
