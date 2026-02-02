import alphashape
import numpy as np
import matplotlib.pyplot as plt
import os

from meca500_params import *
from direct_kinematics import direct_kinematics

def sample_xz_slice(num_samples_per_joint: int=10) -> np.array:
    """
    Sample the attainable volume of the meca500 robotic arm by evaluating the direct kinematics at
    various joint angle combinations in the XZ plane.

    Parameters:
        num_samples_per_joint (int): Number of samples to take for each joint.

    Returns:
        positions (np.array): Array of attainable end-effector positions.
    """
    joint_ranges = [np.linspace(low, high, num_samples_per_joint) for low, high in JOINT_LIMITS]
    positions = []

    # Reduced joint space
    theta1_fixed = 0.0 # Fix joint 1 to sample XZ plane
    theta4_fixed = 0.0 # Fix joint 4 to sample XZ plane
    theta6_fixed = 0.0 # Fix joint 6 since it affects only end-effector orientation

    # Iterate over remaining joints
    for theta2 in joint_ranges[1]:
        for theta3 in joint_ranges[2]:
                for theta5 in joint_ranges[4]:
                    joint_angles = [theta1_fixed, theta2, theta3, theta4_fixed, theta5, theta6_fixed]
                    pos, _ = direct_kinematics(joint_angles)
                    positions.append(pos)
    
    # Convert to numpy array
    positions = np.array(positions)

    return positions

def sample_xy_slices(z_slices: np.array, num_samples_per_joint: int=10) -> dict:
    """
    Sample the attainable volume of the meca500 robotic arm by evaluating the direct kinematics at
    various joint angle combinations in multiple XY planes.

    Parameters:
        z_slices (np.array): Array of Z slices to sample.
        num_samples_per_joint (int): Number of samples to take for each joint.

    Returns:
        slices (dict): keys = z_slice (float), values = np.array of positions
    """
    joint_ranges = [np.linspace(low, high, num_samples_per_joint) for low, high in JOINT_LIMITS]
    slices = {z: [] for z in z_slices}

    # Reduced joint space
    theta5_vals = np.linspace(JOINT_LIMITS[4][0], JOINT_LIMITS[4][1], 3) # Sample fewer values for joint 5
    theta4_fixed = 0.0 # Fix joint 4 to reduce complexity
    theta6_fixed = 0.0 # Fix joint 6 since it affects only end-effector orientation

    # Iterate over remaining joints
    for theta1 in joint_ranges[0]:
        for theta2 in joint_ranges[1]:
            for theta3 in joint_ranges[2]:
                for theta5 in theta5_vals:
                    joint_angles = [theta1, theta2, theta3, theta4_fixed, theta5, theta6_fixed]
                    pos, _ = direct_kinematics(joint_angles)
                    z_idx = np.argmin(np.abs(z_slices - pos[2]))
                    slices[z_slices[z_idx]].append(pos)

    # Convert to numpy arrays
    for z in slices:
        slices[z] = np.array(slices[z])

    return slices

def plot_xz_contour(z_slices: np.array, positions: np.array, alpha: float=0.1, savepath: str=None):
    """
    Plot the contour of the reachable region in the XZ plane.

    Parameters:
        z_slices (np.array): Array of Z slices used for sampling.
        positions (np.array): Array of attainable end-effector positions (Nx3).
        alpha (float): Alpha parameter for alpha shape
        savepath (str): Directory to save the figure (optional)
    """
    _, ax = plt.subplots()
    z_colors = get_z_slice_colors(z_slices)

    # Extract XZ coordinates
    points_xz = positions[:, [0, 2]]

    # Compute alpha shape
    shape = alphashape.alphashape(points_xz, alpha)
    x, z = shape.exterior.xy

    # Plot contour
    ax.plot(x, z, color='tab:green', linestyle='-', linewidth=2)
    ax.fill(x, z, color='tab:green', alpha=0.25)

    # Plot horizontal lines for Z slices
    for z_slice in z_slices:
        ax.axhline(
            y=z_slice,
            color=z_colors[z_slice],
            linestyle='--',
            linewidth=2,
            label=f"z ≈ {int(z_slice)} mm"
        )

    ax.set_aspect('equal')
    ax.set_title('Région atteignable du Meca500 (plan XZ)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.legend(fontsize=8)

    # Save figure if savepath is provided
    if savepath is not None:
        file_name = 'meca500_attainable_xz.png'
        file_path = os.path.join(savepath, file_name)
        plt.savefig(file_path, dpi=300)
        print(f"Figure saved to {file_path}")
    
def plot_xy_slices_contours(slices: dict, alpha: float=0.01, savepath: str=None):
    """
    Plot filled alpha-shape contours for multiple XY slices.

    Parameters:
        slices (dict): Output of sample_xy_slices function
        alpha (float): Alpha parameter for alpha shape
        savepath (str): Directory to save the figure (optional)
    """
    _, ax = plt.subplots()
    z_colors = get_z_slice_colors(np.array(list(slices.keys())))

    for z, points in slices.items():
        # Skip slices with insufficient points
        if points.shape[0] < 10:
            continue

        # Extract XY coordinates
        points_xy = points[:, [0, 1]]

        # Remove extreme outliers (robustness)
        r = np.linalg.norm(points_xy, axis=1)
        mask = r < np.percentile(r, 99)
        points_xy = points_xy[mask]

        # Compute alpha shape
        shape = alphashape.alphashape(points_xy, alpha)
        if shape.geom_type == "Polygon":
            polygons = [shape]
        elif shape.geom_type == "MultiPolygon":
            polygons = [max(shape.geoms, key=lambda p: p.area)]
        else:
            continue
        poly = polygons[0]
        x, y = poly.exterior.xy

        # Plot contours
        ax.plot(x, y, color=z_colors[z], linestyle='--', linewidth=2, label=f"z ≈ {int(z)} mm")
        ax.fill(x, y, color='tab:green', alpha=0.25)

    ax.set_aspect('equal')
    ax.set_title("Région atteignable du Meca500 (plans XY)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.legend(fontsize=8)

    # Save figure if savepath is provided
    if savepath is not None:
        file_name = "meca500_attainable_xy_slices.png"
        file_path = os.path.join(savepath, file_name)
        plt.savefig(file_path, dpi=300)
        print(f"Figure saved to {file_path}")

def get_z_slice_colors(z_slices: np.array, cmap_name: str="plasma") -> dict:
    """
    Generate a consistent color mapping for Z slices.

    Parameters:
        z_slices (np.array): Array of Z slice values.
        cmap_name (str): Matplotlib colormap name.

    Returns:
        custom_color_map (dict): keys = z_slice, values = RGBA color
    """
    cmap = plt.get_cmap(cmap_name)
    z_min, z_max = np.min(z_slices), np.max(z_slices)
    norm = plt.Normalize(vmin=z_min, vmax=z_max)
    return {z: cmap(norm(z)) for z in z_slices}


if __name__ == "__main__":

    # Define paths
    script_path = os.path.dirname(os.path.abspath(__file__))
    figs_path = os.path.join(script_path, '../figs/')

    # Define Z slices to sample
    z_slices = np.array([-50, 0, 150, 250, 350, 450])

    print("\n --- Meca500 Reachable Volume Sampling ---")

    # XZ plot
    print("\nSampling XZ slice...")
    attainable_xz_positions = sample_xz_slice(num_samples_per_joint=30)
    print("Plotting XZ slice...")
    plot_xz_contour(z_slices, attainable_xz_positions, alpha=0.1, savepath=figs_path)

    # XY plot
    print("\nSampling XY slices...")
    xy_slices = sample_xy_slices(z_slices, num_samples_per_joint=30)
    print("Plotting XY slices...")
    plot_xy_slices_contours(xy_slices, alpha=0.01, savepath=figs_path)
