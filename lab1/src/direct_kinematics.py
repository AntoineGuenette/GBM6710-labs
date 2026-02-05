import numpy as np

from meca500_params import *
from transforms import rotmat_x_deg, rotmat_y_deg, rotmat_z_deg, transform_mat, rotmat_to_euler_xyz

def direct_kinematics_T(joint_angles: list) -> np.array:
    """
    Calculate the homogeneous transformation matrix of the end-effector given joint angles.

    Parameters:
        joint_angles (list): Joint angles in degrees (theta 1 to theta 6).

    Returns:
        T_6_0 (np.array): 4x4 homogeneous transformation matrix.
    """
    # Rotation matrices (offsets + joints)
    R_1_0 = np.eye(3) @ rotmat_z_deg(joint_angles[0])
    R_2_1 = rotmat_x_deg(-90) @ rotmat_z_deg(joint_angles[1])
    R_3_2 = rotmat_z_deg(-90) @ rotmat_z_deg(joint_angles[2])
    R_4_3 = rotmat_x_deg(-90) @ rotmat_z_deg(joint_angles[3])
    R_5_4 = rotmat_x_deg(90)  @ rotmat_z_deg(joint_angles[4])
    R_6_5 = rotmat_x_deg(-90) @ rotmat_z_deg(180) @ rotmat_z_deg(joint_angles[5])

    # Transformations
    T_6_0 = (
        transform_mat(R_1_0, P_1org_0) @
        transform_mat(R_2_1, P_2org_1) @
        transform_mat(R_3_2, P_3org_2) @
        transform_mat(R_4_3, P_4org_3) @
        transform_mat(R_5_4, P_5org_4) @
        transform_mat(R_6_5, P_6org_5)
    )

    return T_6_0

def direct_kinematics_position(joint_angles: list, verbose: bool=True) -> tuple:
    """
    Calculate the end-effector position and euler angles of the meca500 robotic arm given joint
    angles.

    Parameters:
        joint_angles (list): Joint angles in degrees (theta 1 to theta 6).
        verbose (bool): Whether to print verbose information about representation singularities.

    Returns:
        position (np.array): The 3D position of the end-effector.
        euler_angles (np.array): The Euler angles of the end-effector.
    """

    # Joint limits check
    for angle, (low, high) in zip(joint_angles, JOINT_LIMITS):
        if not (low <= angle <= high):
            raise ValueError(f"Joint angle {angle} out of bounds [{low}, {high}]")

    T_6_0 = direct_kinematics_T(joint_angles)

    position = T_6_0[0:3, 3]
    euler_angles = rotmat_to_euler_xyz(T_6_0[0:3, 0:3], verbose=verbose)

    return position, euler_angles


if __name__ == "__main__":

    print("\n --- Meca500 Direct Kinematics --- \n")

    joint_1_angle = float(input("Enter joint 1 angle (degrees): "))
    joint_2_angle = float(input("Enter joint 2 angle (degrees): "))
    joint_3_angle = float(input("Enter joint 3 angle (degrees): "))
    joint_4_angle = float(input("Enter joint 4 angle (degrees): "))
    joint_5_angle = float(input("Enter joint 5 angle (degrees): "))
    joint_6_angle = float(input("Enter joint 6 angle (degrees): "))

    joint_angles = [joint_1_angle, joint_2_angle, joint_3_angle,
                    joint_4_angle, joint_5_angle, joint_6_angle]
    
    # Calculate the end-effector position and euler angles
    position, euler_angles = direct_kinematics_position(joint_angles)
    print("\nEnd-Effector Position and Euler Angles:")
    print(f"(x={position[0]:.3f}mm, y={position[1]:.3f}mm, z={position[2]:.3f}mm)")
    print(f"(α={euler_angles[0]:.3f}˚, β={euler_angles[1]:.3f}˚, γ={euler_angles[2]:.3f}˚)\n")
