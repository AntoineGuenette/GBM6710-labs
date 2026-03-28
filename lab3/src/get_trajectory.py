import numpy as np
import os

from lab3.src.phantom_params import *
from lab3.src.point_selection import get_grid_points, get_ball_points, get_phantom_points
from lab3.src.reconstruction import get_calib_mat, get_camera_center, triangulate_points
from lab2.src.registration import compute_registration_transform

def get_trajectory() -> str:
    """
    Compute the trajectory of the system using stereo calibration, triangulation and registration
    in order to estimate the position of the phantom and relevant points in world coordinates.

    Parameters:
        None

    Returns:
        instructions (str): A string containing the instructions for the Meca500 robotic arm to
        follow the computed trajectory.
    """

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lab3_dir = os.path.join(script_dir, '..')
    img_dir = os.path.join(lab3_dir, 'images')
    img_cam1_path = os.path.join(img_dir, 'ex_imgs', 'AimgLb.png')
    img_cam2_path = os.path.join(img_dir, 'ex_imgs', 'AimgRb.png')

    # Get camera point
    pts_grid_cam1 = get_grid_points(img_cam1_path)
    pts_grid_cam2 = get_grid_points(img_cam2_path)

    # Define world points
    pts_grid_world = np.array(
        [ #  (x_max, y_max, 0.0)                                                                 (x_max, y_min, 0.0)
            [[287.5, 287.5, 0.0], [287.5, 262.5, 0.0], [287.5, 237.5, 0.0], [287.5, 212.5, 0.0], [287.5, 187.5, 0.0]], 
            [[262.5, 287.5, 0.0], [262.5, 262.5, 0.0], [262.5, 237.5, 0.0], [262.5, 212.5, 0.0], [262.5, 187.5, 0.0]],
            [[237.5, 287.5, 0.0], [237.5, 262.5, 0.0], [237.5, 237.5, 0.0], [237.5, 212.5, 0.0], [237.5, 187.5, 0.0]],
            [[212.5, 287.5, 0.0], [212.5, 262.5, 0.0], [212.5, 237.5, 0.0], [212.5, 212.5, 0.0], [212.5, 187.5, 0.0]],
            [[187.5, 287.5, 0.0], [187.5, 262.5, 0.0], [187.5, 237.5, 0.0], [187.5, 212.5, 0.0], [187.5, 187.5, 0.0]],
        ] #  (x_min, y_max, 0.0)                                                                 (x_min, y_min, 0.0)
    )
    pts_ball_world = np.array( # Change whith real calib data (chosen positions)
        [[212.5, 212.5, 150.0], # (x_max, y_max, 150.0)
         [212.5, 87.5, 150.0], # (x_max, y_min, 150.0)
         [87.5, 212.5, 150.0]] # (x_min, y_max, 150.0)
    )

    # Compute calibration matrices
    calib_mat_cam1 = get_calib_mat(pts_grid_cam1, pts_grid_world)
    calib_mat_cam2 = get_calib_mat(pts_grid_cam2, pts_grid_world)

    # Get ball points
    pts_ball_cam1 = get_ball_points(img_cam1_path)
    pts_ball_cam2 = get_ball_points(img_cam2_path)

    # Get camera positions in world coordinates
    C_cam1 = get_camera_center(calib_mat_cam1, pts_ball_cam1, pts_ball_world)
    C_cam2 = get_camera_center(calib_mat_cam2, pts_ball_cam2, pts_ball_world)
    print("Camera 1:", C_cam1)
    print("Camera 2:", C_cam2)

    # Get phantom points in cam coordinates
    pts_phantom_cam1 = get_phantom_points(img_cam1_path)
    pts_phantom_cam2 = get_phantom_points(img_cam2_path)

    # Triangulate the points to get phantom points in world coordinates
    pts_world = triangulate_points()

    # Compute the registration transform from phantom to world coordinates
    T_reg = compute_registration_transform(
        a_1 = 0, # Change when phantom_params.py is written
        a_2 = 0, # Change when phantom_params.py is written
        a_3 = 0, # Change when phantom_params.py is written
        b_1 = pts_world[0],
        b_2 = pts_world[1],
        b_3 = pts_world[2]
    )
    R_reg = T_reg[0:3, 0:3]
    t_reg = T_reg[0:3, 3]

    # Define important points (obstacle + target)

    # Register those points to world coordinates

    # Show augmented images (highlight obstacle and target)

    # Print directions
    instructions = ""

    return instructions

if __name__ == "__main__":
    get_trajectory()
    