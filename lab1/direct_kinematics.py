import numpy as np
import scipy as sc
from scipy.spatial.transform import Rotation as rot
import matplotlib.pyplot as plt

def direct_kinematics(joint_angles):
    """
    Calculate the end-effector position and euler angles of the meca500 robotic arm given joint angles.

    Parameters:
        joint_angles (list or np.array): Joint angles in degrees (theta 1 to theta 6).

    Returns:
        position (np.array): The 3D position of the end-effector.
        euler_angles (np.array): The Euler angles of the end-effector.
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
    L4y = 60
    L5 = 60
    L6 = 70

    # Position vectors of each joint in their respective frames

    P_1org_0 = np.array([0, 0, L1])         # Joint 1 position in base frame
    P_2org_1 = np.array([0, 0, L2])         # Joint 2 position in joint 1 frame
    P_3org_2 = np.array([0, -L3, 0])        # Joint 3 position in joint 2 frame
    P_4org_3 = np.array([L4x, L4y, 0])      # Joint 4 position in joint 3 frame
    P_5org_4 = np.array([0, 0, L5])         # Joint 5 position in joint 4 frame
    P_6org_5 = np.array([0, L6, 0])         # Joint 6 position in joint 5 frame

    # Rotation matrices for each joint
    # Important: mechanical offsets (fixed) are separated from joint rotations (variable)

    # Joint 1
    R_1_0_offset = np.eye(3)                # No mechanical offset for joint 1
    R_1_0_joint  = Rz(joint_angles[0])      # Rotation around local z-axis
    R_1_0 = R_1_0_offset @ R_1_0_joint

    # Joint 2
    R_2_1_offset = Rx(-90)                  # Mechanical offset between joint 1 and joint 2 frames
    R_2_1_joint  = Rz(joint_angles[1])      # Rotation around joint 2 local z-axis
    R_2_1 = R_2_1_offset @ R_2_1_joint

    # Joint 3
    R_3_2_offset = Rz(-90)                  # Mechanical offset for joint 3
    R_3_2_joint  = Rz(joint_angles[2])      # Rotation around joint 3 local z-axis
    R_3_2 = R_3_2_offset @ R_3_2_joint

    # Joint 4
    R_4_3_offset = Rx(-90)                  # Mechanical offset for joint 4
    R_4_3_joint  = Rz(joint_angles[3])      # Rotation around joint 4 local z-axis
    R_4_3 = R_4_3_offset @ R_4_3_joint

    # Joint 5
    R_5_4_offset = Rx(90)                   # Mechanical offset for joint 5
    R_5_4_joint  = Rz(joint_angles[4])      # Rotation around joint 5 local z-axis
    R_5_4 = R_5_4_offset @ R_5_4_joint

    # Joint 6
    R_6_5_offset = Rx(-90) @ Ry(180)        # Mechanical offset for joint 6
    R_6_5_joint  = Rz(joint_angles[5])      # Rotation around joint 6 local z-axis
    R_6_5 = R_6_5_offset @ R_6_5_joint

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

    # End-effector Euler angles in base frame (not used in this function)
    R_6_0 = T_6_0[0:3, 0:3]
    rx, ry, rz = myRotm2eul(R_6_0)
    euler_angles = [rx, ry, rz]

    return position, euler_angles

def Rz(theta):
    """
    Rotation matrix for a rotation around the local z-axis.

    Parameters:
        theta (float): Rotation angle around z-axis in degrees.

    Returns:
        R (np.array): 3x3 rotation matrix.
    """
    return rot.from_euler('z', theta, degrees=True).as_matrix()

def Ry(theta):
    """
    Rotation matrix for a rotation around the local y-axis.

    Parameters:
        theta (float): Rotation angle around y-axis in degrees.

    Returns:
        R (np.array): 3x3 rotation matrix.
    """
    return rot.from_euler('y', theta, degrees=True).as_matrix()

def Rx(theta):
    """
    Rotation matrix for a rotation around the local x-axis.

    Parameters:
        theta (float): Rotation angle around x-axis in degrees.

    Returns:
        R (np.array): 3x3 rotation matrix.
    """
    return rot.from_euler('x', theta, degrees=True).as_matrix()

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

def myRotm2eul(R):
    """
    Return the euler angles from a rotation matrix following the mobile xyz convention.

    Parameters:
        R (np.array): 3x3 rotation matrix

    Returns:
        alpha (float): rotation around x' in degrees
        beta (float): rotation around y' in degrees
        gamma (float): rotation around z' in degrees
    """

    # Numerical tolerance for singularity detection
    eps = 1e-9

    if abs(R[0, 2] - 1) < eps or abs(R[0, 2] + 1) < eps:
        # Representation singularity
        alpha_rad = 0.0
        beta_rad = R[0, 2] * (np.pi / 2)
        gamma_rad = np.arctan2(R[1, 0], R[1, 1])
    else:
        alpha_rad = np.arctan2(-R[1, 2], R[2, 2])
        beta_rad = np.arcsin(R[0, 2])
        gamma_rad = np.arctan2(-R[0, 1], R[0, 0])

    # Convert radians to degrees
    alpha = np.degrees(alpha_rad) 
    beta = np.degrees(beta_rad)
    gamma = np.degrees(gamma_rad)

    # Adjust angles to be in the desired range
    alpha = alpha + 180 if alpha < 0 else alpha - 180 if alpha > 0 else alpha
    beta = -beta

    # For gamma, no adjustment needed, even if the angles do not correspond to the expected values.
    # From Mecademic documentation (https://mecademic.com/insights/academic-tutorials/space-orientation-euler-angles/):
    # In the chosen Euler angle convention, angles α and β define this direction, while angle γ is
    # ignored because it corresponds to a parasitic rotation that is uncontrollable.

    return alpha, beta, gamma

# Tests with different joint angles
joint_angles = [0, 0, 0, 0, 0, 0]
# joint_angles = [10, 0, 15, 0, -20, -50]
# joint_angles = [15, 16, -14, 7, 37, 50]
# joint_angles = [52, -13, -35, -38, 76, 91]
# joint_angles = [-55, -1, 27, -72, -59, 58]
# joint_angles = [-55.008, -0.939, 27.644, -72.537, -59.184, 58.447]
# joint_angles = [46.287, 6.624, -5.171, 39.165, -59.322, 73.873]

# Calculate the end-effector position and euler angles
position, euler_angles = direct_kinematics(joint_angles)
print("\nEnd-Effector Position and Euler Angles:")
print(f"(x={position[0]:.3f}mm, y={position[1]:.3f}mm, z={position[2]:.3f}mm)")
print(f"(α={euler_angles[0]:.3f}˚, β={euler_angles[1]:.3f}˚, γ={euler_angles[2]:.3f}˚)\n")