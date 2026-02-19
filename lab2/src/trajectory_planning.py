import numpy as np

from phantom_params import *
from registration import compute_registration_transform

def trajectory_planning(
        chosen_tumor: str,
        bead_1_position_world: np.array,
        bead_2_position_world: np.array,
        bead_3_position_world: np.array,
) -> str:
    """
    Compute the trajectory for the Meca500 robotic arm to reach the target tumor from the starting point.
    The trajectory is computed in joint space using inverse kinematics and a registration transform.

    Parameters:
        chosen_tumor (str): The chosen tumor to target ("pink" or "orange").
        bead_1_position_world (np.array): Position of bead 1 in world coordinates (mm).
        bead_2_position_world (np.array): Position of bead 2 in world coordinates (mm).
        bead_3_position_world (np.array): Position of bead 3 in world coordinates (mm).
    """
    # Compute the registration transform from phantom to world coordinates
    R_reg, t_reg = compute_registration_transform(
        bead_1_position_phantom,
        bead_2_position_phantom,
        bead_3_position_phantom,
        bead_1_position_world,
        bead_2_position_world,
        bead_3_position_world
    )

    # Select target and starting points based on the chosen tumor
    if chosen_tumor == "pink":
        target_point_phantom = pink_tumor_position_phantom
        starting_point_phantom = pink_starting_point_phantom
    elif chosen_tumor == "orange":
        target_point_phantom = orange_tumor_position_phantom
        starting_point_phantom = orange_starting_point_phantom
    else:
        raise ValueError("Invalid tumor choice. Must be 'pink' or 'orange'.")

    # Transform target and starting points to world coordinates
    target_point_world = R_reg @ target_point_phantom + t_reg
    starting_point_world = R_reg @ starting_point_phantom + t_reg