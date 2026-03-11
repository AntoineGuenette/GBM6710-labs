import numpy as np

from lab1.src.transforms import rotmat_to_euler_xyz

def euler_from_direction(direction: np.array) -> np.array:
    """
    Compute Euler XYZ angles from a direction vector.

    This function constructs a tool coordinate frame whose z-axis is aligned with the given
    direction vector. The corresponding rotation matrix (tool frame expressed in the world frame) is
    then converted to Euler angles using the XYZ convention.

    Parameters:
        direction (np.ndarray): A 3-element vector representing the desired direction of the tool 
            z-axis in the world coordinate frame. The vector should be normalized.

    Returns:
        euler_angles (np.ndarray): A 3-element array containing the Euler angles (alpha, beta,
            gamma) following the XYZ convention, expressed in radians.
    """

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

    return euler_angles