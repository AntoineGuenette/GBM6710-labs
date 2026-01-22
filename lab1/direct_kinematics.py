import numpy as np
import scipy as sc
from scipy.spatial.transform import Rotation as rot
import matplotlib.pyplot as plt

def direct_kinematics(joint_angles):
    """
    Calculate the end-effector position of the meca500 robotic arm given joint angles.

    Parameters:
        joint_angles (list or np.array): Joint angles in degrees (theta 1 to theta 6).

    Returns:
        position (np.array): The 3D position of the end-effector.
    """

    # Verify that the angles are in the correct range of values
    constraints = [(-175, 175), (-70, 90), (-135, 70), (-170, 170), (-115, 115), (-180, 180),]
    for angle, constraint in zip(joint_angles, constraints):
        if not (constraint[0] <= angle <= constraint[1]):
            raise ValueError(f"Joint angle {angle} is out of bounds {constraint}")
        
    # Length of each link in mm
    L1 = 90
    L2 = 45
    L3 = 135
    L4x = 38
    L4y = 60 #TODO check if correct
    L5 = 60 #TODO check if correct
    L6 = 70

    # Position vectors of each joint in their respective frames
    P_1org_0 = np.array([0, 0, L1]) # Joint 1 position in base frame
    P_2org_1 = np.array([0, 0, L2]) # Joint 2 position in joint 1 frame
    P_3org_2 = np.array([0, -L3, 0]) # Joint 3 position in joint 2 frame
    P_4org_3 = np.array([L4x, L4y, 0]) # Joint 4 position in joint 3 frame
    P_5org_4 = np.array([0, 0, L5]) # Joint 5 position in joint 4 frame
    P_6org_5 = np.array([0, L6, 0]) # Joint 6 position in joint 5 frame

    # Rotation matrices for each joint
    R_1_0 = euler_angles_to_rot_mat(0, 0, joint_angles[0]) # Rotation from base frame to joint 1 frame
    R_2_1 = euler_angles_to_rot_mat(-90, 0, joint_angles[1]) # Rotation from joint 1 frame to joint 2 frame
    R_3_2 = euler_angles_to_rot_mat(0, 0, joint_angles[2] - 90) # Rotation from joint 2 frame to joint 3 frame
    R_4_3 = euler_angles_to_rot_mat(-90, 0, joint_angles[3]) # Rotation from joint 3 frame to joint 4 frame
    R_5_4 = euler_angles_to_rot_mat(90, 0, joint_angles[4]) # Rotation from joint 4 frame to joint 5 frame
    R_6_5 = euler_angles_to_rot_mat(-90, 180, joint_angles[5]) # Rotation from joint 5 frame to joint 6 frame

    # Homogeneous transformation matrices for each joint
    T_1_0 = T(R_1_0, P_1org_0)
    T_2_1 = T(R_2_1, P_2org_1)
    T_3_2 = T(R_3_2, P_3org_2)
    T_4_3 = T(R_4_3, P_4org_3)
    T_5_4 = T(R_5_4, P_5org_4)
    T_6_5 = T(R_6_5, P_6org_5)

    # Overall transformation from base frame to end-effector frame
    T_6_0 = T_1_0 @ T_2_1 @ T_3_2 @ T_4_3 @ T_5_4 @ T_6_5

    # End-effector position in base frame
    position = T_6_0[0:3, 3]

    return position

def euler_angles_to_rot_mat(theta_x, theta_y, theta_z):
    """
    Computes the rotation matrix from Euler angles.

    Parameters:
        theta_x (float): Rotation angle around x-axis in degrees.
        theta_y (float): Rotation angle around y-axis in degrees.
        theta_z (float): Rotation angle around z-axis in degrees.

    Returns:
        R (np.array): 3x3 rotation matrix.
    """
    R = rot.from_euler(seq='xyz', angles=[theta_x, theta_y, theta_z], degrees=True)
    return R.as_matrix()

def T(R: np.array, P: np.array):
    """
    Create a homogeneous transformation matrix from rotation matrix R and position vector P.

    Parameters:
        R (np.array): 3x3 rotation matrix.
        P (np.array): 3x1 position vector.

    Returns:
        T (np.array): 4x4 homogeneous transformation matrix.
    """
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,3] = P
    return T

joint_angles = [10, 0, 15, 0, -20, -50]  # Example joint angles in degrees

# Calculate the end-effector position
position = direct_kinematics(joint_angles)
print(f"End-effector position: {position[0]:.3f} mm, {position[1]:.3f} mm, {position[2]:.3f} mm")