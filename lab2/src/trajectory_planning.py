import numpy as np

from lab2.src.phantom_params import *
from lab2.src.registration import compute_registration_transform
from lab1.src.transforms import rotmat_to_euler_xyz

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
    T_reg = compute_registration_transform(
        bead_1_position_phantom,
        bead_2_position_phantom,
        bead_3_position_phantom,
        bead_1_position_world,
        bead_2_position_world,
        bead_3_position_world
    )
    R_reg = T_reg[0:3, 0:3]
    t_reg = T_reg[0:3, 3]

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

    # Compute orientation from trajectory direction

    # Direction from starting point to insertion point (descending motion)
    direction = target_point_world - starting_point_world
    direction = direction / np.linalg.norm(direction)

    # Reference axis
    x_ref = np.array([1.0, 0.0, 0.0])

    # Build tool coordinate frame with z-axis aligned with trajectory
    z_tool = direction
    y_tool = np.cross(z_tool, x_ref)
    y_tool = y_tool / np.linalg.norm(y_tool)
    x_tool = np.cross(y_tool, z_tool)

    # Rotation matrix (tool -> world)
    R_tool = np.column_stack((x_tool, y_tool, z_tool))

    # Convert rotation matrix to Euler angles (alpha, beta, gamma)
    euler_angles = rotmat_to_euler_xyz(R_tool)

    # Generate instructions for the Meca500 robotic arm to follow the computed trajectory
    instructions = f"""
SetTRF(0,0,53.8,0,0,0)
SetJointVel(10)
SetCartLinVel(10)
SetCartAngVel(15)
MovePose({starting_point_world[0]:.2f},{starting_point_world[1]:.2f},{starting_point_world[2]:.2f},{euler_angles[0]:.2f},{euler_angles[1]:.2f},{euler_angles[2]:.2f})
Delay(3)
MoveLin({insertion_point_world[0]:.2f},{insertion_point_world[1]:.2f},{insertion_point_world[2]:.2f},{euler_angles[0]:.2f},{euler_angles[1]:.2f},{euler_angles[2]:.2f})
Delay(1)
SetCartLinVel(5)
MoveLin({target_point_world[0]:.2f},{target_point_world[1]:.2f},{target_point_world[2]:.2f},{euler_angles[0]:.2f},{euler_angles[1]:.2f},{euler_angles[2]:.2f})
Delay(15)
MoveLin({insertion_point_world[0]:.2f},{insertion_point_world[1]:.2f},{insertion_point_world[2]:.2f},{euler_angles[0]:.2f},{euler_angles[1]:.2f},{euler_angles[2]:.2f})
SetCartLinVel(10)
MoveLin({starting_point_world[0]:.2f},{starting_point_world[1]:.2f},{starting_point_world[2]:.2f},{euler_angles[0]:.2f},{euler_angles[1]:.2f},{euler_angles[2]:.2f})
MoveJoints(0,0,0,0,0,0)
    """

    return instructions
    

if __name__ == "__main__":

    chosen_tumor = "orange"
    bead_1_position_world = np.array([-88.191, 149.478, 54.032], dtype=float)
    bead_2_position_world = np.array([-65.506, 124.148, 54.167], dtype=float)
    bead_3_position_world = np.array([-136.058, 53.315, 54.494], dtype=float)

    instructions = trajectory_planning(
           chosen_tumor = chosen_tumor,
           bead_1_position_world = bead_1_position_world,
           bead_2_position_world = bead_2_position_world,
           bead_3_position_world = bead_3_position_world
    )
    print("\nENTERED PARAMETERS:\n")
    print(f"Chosen Tumor: {chosen_tumor}")
    print(f"Bead 1 Position (World): {bead_1_position_world}")
    print(f"Bead 2 Position (World): {bead_2_position_world}")
    print(f"Bead 3 Position (World): {bead_3_position_world}")

    print("\nTRAJECTORY INSTRUCTIONS FOR MECA500 ROBOTIC ARM:")
    print(instructions)