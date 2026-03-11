import numpy as np

from lab1.src.transforms import rotmat_to_euler_xyz

def euler_from_direction(direction: np.array) -> np.array:

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