import numpy as np

from lab2.src.phantom_params import *
from lab2.src.registration import compute_registration_transform
from lab2.src.utils import euler_from_direction


def trajectory_planning(
        biopsy_mode : str,
        chosen_tumor: str,
        bead_1_position_world: np.array,
        bead_2_position_world: np.array,
        bead_3_position_world: np.array,
) -> str:
    """
    Compute the trajectory for the Meca500 robotic arm to reach the target tumor from the starting
    point.

    Parameters:
        biopsy_mode (str): The biopsy mode ("touch" or "CMD")
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

    # Define the starting, insertion and target points based on the chosen tumor
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

    # Register those points to world coordinates
    starting_point_world = R_reg @ starting_point_phantom + t_reg
    insertion_point_world = R_reg @ insertion_point_phantom + t_reg
    target_point_world = R_reg @ target_point_phantom + t_reg

    # Compute trajectory direction from starting point to target point
    direction = target_point_world - starting_point_world
    direction = direction / np.linalg.norm(direction)

    # Compute insertion angle from direction
    insertion_angles = euler_from_direction(direction)
    
    # Generate instructions for the Meca500 robotic arm to follow the computed trajectory
    if biopsy_mode == "touch":
        instructions = f"""
SetTRF(0,0,53.8,0,0,0)
SetJointVel(10)
SetCartLinVel(10)
SetCartAngVel(15)
MovePose({starting_point_world[0]:.2f},{starting_point_world[1]:.2f},{starting_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
Delay(3)
MoveLin({insertion_point_world[0]:.2f},{insertion_point_world[1]:.2f},{insertion_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
Delay(1)
SetCartLinVel(5)
MoveLin({target_point_world[0]:.2f},{target_point_world[1]:.2f},{target_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
Delay(15)
MoveLin({insertion_point_world[0]:.2f},{insertion_point_world[1]:.2f},{insertion_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
SetCartLinVel(10)
MoveLin({starting_point_world[0]:.2f},{starting_point_world[1]:.2f},{starting_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
MoveJoints(0,0,0,0,0,0)
        """
    elif biopsy_mode == "CMD":
        # Define four points in the same world xy plane as the target
        mvt_length = 2 #mm
        CMD_point_1 = target_point_world + [mvt_length, 0, 0]
        CMD_point_2 = target_point_world - [mvt_length, 0, 0]
        CMD_point_3 = target_point_world + [0, mvt_length, 0]
        CMD_point_4 = target_point_world - [0, mvt_length, 0]

        # Compute the direction for each point (from insertion point to CMD point)
        CMD_dir_1 = CMD_point_1 - insertion_point_world 
        CMD_dir_1 = CMD_dir_1 / np.linalg.norm(CMD_dir_1)
        CMD_dir_2 = CMD_point_2 - insertion_point_world
        CMD_dir_2 = CMD_dir_2 / np.linalg.norm(CMD_dir_2)
        CMD_dir_3 = CMD_point_3 - insertion_point_world
        CMD_dir_3 = CMD_dir_3 / np.linalg.norm(CMD_dir_3)
        CMD_dir_4 = CMD_point_4 - insertion_point_world
        CMD_dir_4 = CMD_dir_4 / np.linalg.norm(CMD_dir_4)

        # Compute the Euler angles for each direction
        CMD_angles_1 = euler_from_direction(CMD_dir_1)
        CMD_angles_2 = euler_from_direction(CMD_dir_2)
        CMD_angles_3 = euler_from_direction(CMD_dir_3)
        CMD_angles_4 = euler_from_direction(CMD_dir_4)

        # Update the instructions
        instructions = f"""
SetTRF(0,0,53.8,0,0,0)
SetJointVel(10)
SetCartLinVel(10)
SetCartAngVel(15)
MovePose({starting_point_world[0]:.2f},{starting_point_world[1]:.2f},{starting_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
Delay(3)
MoveLin({insertion_point_world[0]:.2f},{insertion_point_world[1]:.2f},{insertion_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
Delay(1)
SetCartLinVel(5)
MoveLin({target_point_world[0]:.2f},{target_point_world[1]:.2f},{target_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
Delay(5)
MoveLin({CMD_point_1[0]:.2f},{CMD_point_1[1]:.2f},{CMD_point_1[2]:.2f},{CMD_angles_1[0]:.2f},{CMD_angles_1[1]:.2f},{CMD_angles_1[2]:.2f})
MoveLin({target_point_world[0]:.2f},{target_point_world[1]:.2f},{target_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
MoveLin({CMD_point_2[0]:.2f},{CMD_point_2[1]:.2f},{CMD_point_2[2]:.2f},{CMD_angles_2[0]:.2f},{CMD_angles_2[1]:.2f},{CMD_angles_2[2]:.2f})
MoveLin({target_point_world[0]:.2f},{target_point_world[1]:.2f},{target_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
MoveLin({CMD_point_3[0]:.2f},{CMD_point_3[1]:.2f},{CMD_point_3[2]:.2f},{CMD_angles_3[0]:.2f},{CMD_angles_3[1]:.2f},{CMD_angles_3[2]:.2f})
MoveLin({target_point_world[0]:.2f},{target_point_world[1]:.2f},{target_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
MoveLin({CMD_point_4[0]:.2f},{CMD_point_4[1]:.2f},{CMD_point_4[2]:.2f},{CMD_angles_4[0]:.2f},{CMD_angles_4[1]:.2f},{CMD_angles_4[2]:.2f})
MoveLin({target_point_world[0]:.2f},{target_point_world[1]:.2f},{target_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
Delay(5)
MoveLin({insertion_point_world[0]:.2f},{insertion_point_world[1]:.2f},{insertion_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
SetCartLinVel(10)
MoveLin({starting_point_world[0]:.2f},{starting_point_world[1]:.2f},{starting_point_world[2]:.2f},{insertion_angles[0]:.2f},{insertion_angles[1]:.2f},{insertion_angles[2]:.2f})
MoveJoints(0,0,0,0,0,0)
        """
    else :
        raise ValueError("Invalid biopsy mode. Must be 'touch' or 'CMD'.")

    return instructions
    

if __name__ == "__main__":

    biopsy_mode = "touch"
    chosen_tumor = "orange"

    bead_1_position_world = np.array([31.861, 171.536, 54.084], dtype=float)
    bead_2_position_world = np.array([16.001, 202.145, 53.834], dtype=float)
    bead_3_position_world = np.array([102.439, 253.157, 54.303], dtype=float)

    instructions = trajectory_planning(
        biopsy_mode = biopsy_mode,
        chosen_tumor = chosen_tumor,
        bead_1_position_world = bead_1_position_world,
        bead_2_position_world = bead_2_position_world,
        bead_3_position_world = bead_3_position_world
    )
    print("\nENTERED PARAMETERS:\n")
    print(f"Chosen Mode: {biopsy_mode}")
    print(f"Chosen Tumor: {chosen_tumor}")
    print(f"Bead 1 Position (World): {bead_1_position_world}")
    print(f"Bead 2 Position (World): {bead_2_position_world}")
    print(f"Bead 3 Position (World): {bead_3_position_world}")

    print("\nTRAJECTORY INSTRUCTIONS FOR MECA500 ROBOTIC ARM:")
    print(instructions)