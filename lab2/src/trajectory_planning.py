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
    Compute the trajectory for the Meca500 robotic arm to reach the target tumor from the starting
    point. The trajectory is computed in joint space using inverse kinematics and a registration
    transform.

    Parameters:
        chosen_tumor (str): The chosen tumor to target ("pink" or "orange").
        bead_1_position_world (np.array): Position of bead 1 in world coordinates (mm).
        bead_2_position_world (np.array): Position of bead 2 in world coordinates (mm).
        bead_3_position_world (np.array): Position of bead 3 in world coordinates (mm).

    Returns:
        instructions (str): A string containing the instructions for the Meca500 robotic arm to
        follow the computed trajectory.
    """

    # Compute the registration transform from phantom to world coordinates
    R_reg, t_reg = compute_registration_transform(
        bead_1_position_phantom,
        bead_2_position_phantom,
        bead_3_position_phantom,
        bead_1_position_world,
        bead_2_position_world,
        bead_3_position_world,
        tolerance = 1e-4,
        max_itteration = 10
    )

    # Select target and starting points based on the chosen tumor
    if chosen_tumor == "pink":
        starting_point_phantom = pink_starting_point_phantom
        insertion_point_phantom = pink_insertion_point_phantom
        target_point_phantom = pink_tumor_position_phantom
    elif chosen_tumor == "orange":
        starting_point_phantom = orange_starting_point_phantom
        insertion_point_phantom = orange_insertion_point_phantom
        target_point_phantom = orange_tumor_position_phantom
    else:
        raise ValueError("Invalid tumor choice. Must be 'pink' or 'orange'.")

    # Transform target and starting points to world coordinates
    starting_point_world = R_reg @ starting_point_phantom + t_reg
    insertion_point_world = R_reg @ insertion_point_phantom + t_reg
    target_point_world = R_reg @ target_point_phantom + t_reg

    # Generate instructions for the Meca500 robotic arm to follow the computed trajectory
    instructions = f"""
    SetTRF(0,0,53.8,0,0,0)
    SetJointVel(10)
    SetCartLinVel(10)
    SetCartAngVel(15)
    MovePose({starting_point_world[0]:.2f},{starting_point_world[1]:.2f},{starting_point_world[2]:.2f},0,180,0)
    Delay(3000)
    MoveLin({insertion_point_world[0]:.2f},{insertion_point_world[1]:.2f},{insertion_point_world[2]:.2f},0,180,0)
    SetCartLinVel(5)
    MoveLin({target_point_world[0]:.2f},{target_point_world[1]:.2f},{target_point_world[2]:.2f},0,180,0)
    Delay(15000)
    MoveLin({insertion_point_world[0]:.2f},{insertion_point_world[1]:.2f},{insertion_point_world[2]:.2f},0,180,0)
    SetCartLinVel(10)
    MoveLin({starting_point_world[0]:.2f},{starting_point_world[1]:.2f},{starting_point_world[2]:.2f},0,180,0)
    MoveJoints(0,0,0,0,0,0)
    """

    return instructions
    

if __name__ == "__main__":

    chosen_tumor = input("Enter the chosen tumor to target (pink/orange): ")

    bead_1_position_world_x = float(input("Enter the x-coordinate of bead 1 in world coordinates (mm): "))
    bead_1_position_world_y = float(input("Enter the y-coordinate of bead 1 in world coordinates (mm): "))
    bead_1_position_world_z = float(input("Enter the z-coordinate of bead 1 in world coordinates (mm): "))
    bead_1_position_world = np.array([bead_1_position_world_x, bead_1_position_world_y, bead_1_position_world_z], dtype=float)

    bead_2_position_world_x = float(input("Enter the x-coordinate of bead 2 in world coordinates (mm): "))
    bead_2_position_world_y = float(input("Enter the y-coordinate of bead 2 in world coordinates (mm): "))
    bead_2_position_world_z = float(input("Enter the z-coordinate of bead 2 in world coordinates (mm): "))
    bead_2_position_world = np.array([bead_2_position_world_x, bead_2_position_world_y, bead_2_position_world_z], dtype=float)

    bead_3_position_world_x = float(input("Enter the x-coordinate of bead 3 in world coordinates (mm): "))
    bead_3_position_world_y = float(input("Enter the y-coordinate of bead 3 in world coordinates (mm): "))
    bead_3_position_world_z = float(input("Enter the z-coordinate of bead 3 in world coordinates (mm): "))
    bead_3_position_world = np.array([bead_3_position_world_x, bead_3_position_world_y, bead_3_position_world_z], dtype=float)

    instructions = trajectory_planning(
           chosen_tumor = chosen_tumor,
           bead_1_position_world = bead_1_position_world,
           bead_2_position_world = bead_2_position_world,
           bead_3_position_world = bead_3_position_world
    )

    print("\nINSTRUCTIONS FOR MECA500 ROBOTIC ARM:\n")
    print(instructions)