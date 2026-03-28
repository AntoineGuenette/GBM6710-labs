import numpy as np
import os

from lab3.src.phantom_params import *
from lab3.src.point_selection import get_grid_points, get_ball_points
from lab3.src.calibration import calibrate_camera
from lab2.src.registration import compute_registration_transform
from lab2.src.utils import euler_from_direction

if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lab3_dir = os.path.join(script_dir, '..')
    img_dir = os.path.join(lab3_dir, 'images')
    img_cam1_path = os.path.join(img_dir, 'ex_imgs', 'AimgLb.png')
    img_cam2_path = os.path.join(img_dir, 'ex_imgs', 'AimgRb.png')

    # Get image points
    pts_grid_cam1 = get_grid_points(img_cam1_path)
    pts_grid_cam2 = get_grid_points(img_cam2_path)
    pts_ball_cam1 = get_ball_points(img_cam1_path)
    pts_ball_cam2 = get_ball_points(img_cam2_path)
    pts_cam1 = np.vstack((pts_grid_cam1.reshape(-1, 2), pts_ball_cam1.reshape(-1, 2)))
    pts_cam2 = np.vstack((pts_grid_cam2.reshape(-1, 2), pts_ball_cam2.reshape(-1, 2)))
    pts_cam1 = pts_cam1.reshape(-1, 2)
    pts_cam2 = pts_cam2.reshape(-1, 2)

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
    pts_world = np.vstack((pts_grid_world.reshape(-1, 3), pts_ball_world.reshape(-1, 3)))
    pts_world = pts_world.reshape(-1, 3)

    # Compute calibration matrices with DLT method
    calib_params_cam1 = calibrate_camera(pts_cam1, pts_world)
    calib_params_cam2 = calibrate_camera(pts_cam2, pts_world)

    # Show camera center in world coordinates
    coord_cam1 = calib_params_cam1['C']
    coord_cam2 = calib_params_cam2['C']
    print(coord_cam1)
    print(coord_cam2)
    
    # Registration
        # get_phantom_points
        # compute_registration_transform

    # Show augmented images (highlight obstacle and target)

    # Print directions
    