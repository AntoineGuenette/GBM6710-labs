import numpy as np
from scipy.spatial.transform import Rotation as rot

def Rx(theta):
    """Rotation around x-axis (degrees)."""
    return rot.from_euler('x', theta, degrees=True).as_matrix()

def Ry(theta):
    """Rotation around y-axis (degrees)."""
    return rot.from_euler('y', theta, degrees=True).as_matrix()

def Rz(theta):
    """Rotation around z-axis (degrees)."""
    return rot.from_euler('z', theta, degrees=True).as_matrix()

def T(R: np.array, P: np.array):
    """
    Homogeneous transformation matrix from rotation R and position P.
    """
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = P
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
    if R[0, 2] == 1 or R[0, 2] == -1:
        # Representation singularity (|β| = 90°)
        alpha_rad = 0
        beta_rad = R[0, 2] * (np.pi / 2)
        gamma_rad = np.arctan2(R[1, 0], R[1, 1])
    else:
        alpha_rad = np.arctan2(-R[1, 2], R[2, 2])
        beta_rad = np.arcsin(R[0, 2])
        gamma_rad = np.arctan2(-R[0, 1], R[0, 0])

    # Convert radians to degrees
    alpha = np.degrees(alpha_rad) 
    beta = np.degrees(beta_rad)

    # Warn when approaching representation singularity (89˚ < |β| < 90°)
    if 89.0 < abs(beta) < 90.0:
        print(f"WARNING : Close to a representation singularity : |β| = {abs(beta):.3f}˚. α may be ill-conditioned.")
        
    gamma = np.degrees(gamma_rad)

    # Adjust angles to be in the desired range
    alpha = alpha + 180 if alpha < 0 else alpha - 180 if alpha > 0 else alpha
    beta = -beta

    # For gamma, no adjustment needed, even if the angles do not correspond to the expected values.
    # From Mecademic documentation (https://mecademic.com/insights/academic-tutorials/space-orientation-euler-angles/):
    # In the chosen Euler angle convention, angles α and β define this direction, while angle γ is
    # ignored because it corresponds to a parasitic rotation that is uncontrollable.

    return alpha, beta, gamma