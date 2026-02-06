import numpy as np
from scipy.spatial.transform import Rotation as R

from lab1.src.meca500_params import *

def rotmat_x_deg(theta: float) -> np.array:
    """Rotation matrix around x-axis (degrees)."""
    return R.from_euler('x', theta, degrees=True).as_matrix()

def rotmat_y_deg(theta: float) -> np.array:
    """Rotation matrix around y-axis (degrees)."""
    return R.from_euler('y', theta, degrees=True).as_matrix()

def rotmat_z_deg(theta: float) -> np.array:
    """Rotation matrix around z-axis (degrees)."""
    return R.from_euler('z', theta, degrees=True).as_matrix()

def transform_mat(R: np.array, P: np.array) -> np.array:
    """
    Homogeneous transformation matrix from rotation R and position P.
    """
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = P
    return T

def rotmat_to_euler_xyz(rotmat: np.array, verbose: bool=True) -> tuple:
    """
    Return the euler angles from a rotation matrix following the mobile xyz convention.

    Parameters:
        rotmat (np.array): 3x3 rotation matrix
        verbose (bool): Whether to print verbose information about representation singularities.

    Returns:
        alpha (float): rotation around x' in degrees
        beta (float): rotation around y' in degrees
        gamma (float): rotation around z' in degrees
    """
    if rotmat[0, 2] == 1 or rotmat[0, 2] == -1:
        # Representation singularity (|β| = 90°)
        alpha_rad = 0
        beta_rad = rotmat[0, 2] * (np.pi / 2)
        gamma_rad = np.arctan2(rotmat[1, 0], rotmat[1, 1])
    else:
        alpha_rad = np.arctan2(-rotmat[1, 2], rotmat[2, 2])
        beta_rad = np.arcsin(rotmat[0, 2])
        gamma_rad = np.arctan2(-rotmat[0, 1], rotmat[0, 0])

    # Convert radians to degrees
    alpha = np.degrees(alpha_rad) 
    beta = np.degrees(beta_rad)

    # Warn when approaching representation singularity (89˚ < |β| < 90°)
    if 89.0 < abs(beta) < 90.0 and verbose:
        print(f"WARNING : Close to a representation singularity : |β| = {abs(beta):.3f}˚. α may be ill-conditioned.")
        
    gamma = np.degrees(gamma_rad)

    return alpha, beta, gamma

def numerical_jacobian(joint_angles, p_target, R_target, eps_deg: float = 1e-2):
    """
    Compute the numerical Jacobian J (6x6) and the error vector e (6,) for a given target pose.

    The error vector is defined as:
        e = [p_target - p_current, rotvec(R_target * R_current^T)]

    - position error in millimeters
    - orientation error in radians

    Parameters:
        joint_angles (np.array): Joint angles in degrees (6,)
        p_target (np.array): Target position (3,)
        R_target (np.array): Target rotation matrix (3x3)
        eps_deg (float): Finite difference step in degrees

    Returns:
        J (np.array): Numerical Jacobian matrix (6x6)
        e (np.array): Error vector at the current configuration (6,)
    """
    from direct_kinematics import direct_kinematics_T

    # Forward kinematics at the current configuration
    T0 = direct_kinematics_T(joint_angles)
    p_current = T0[:3, 3]
    R_current = T0[:3, :3]
    euler_current = np.array(rotmat_to_euler_xyz(R_current, verbose=False))
    euler_target = np.array(rotmat_to_euler_xyz(R_target, verbose=False))

    # Error at the current configuration
    e0 = np.hstack([
        p_target - p_current,
        np.deg2rad(euler_target - euler_current)
    ])

    # Numerical Jacobian initialization
    J = np.zeros((6, 6))

    # Finite difference approximation
    for i in range(6):
        q_perturbed = joint_angles.copy()
        q_perturbed[i] += eps_deg

        T1 = direct_kinematics_T(q_perturbed)
        p_perturbed = T1[:3, 3]
        R_perturbed = T1[:3, :3]
        euler_perturbed = np.array(rotmat_to_euler_xyz(R_perturbed, verbose=False))

        e1 = np.hstack([
            p_target - p_perturbed,
            np.deg2rad(euler_target - euler_perturbed)
        ])

        eps_rad = np.deg2rad(eps_deg)
        J[:, i] = (e1 - e0) / eps_rad

    return J, e0

