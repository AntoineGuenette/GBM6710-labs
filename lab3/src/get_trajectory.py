import numpy as np
import os

from lab3.src.phantom_params import *
from lab3.src.utils import get_points
from lab3.src.calibration import get_calib_mat
from lab2.src.registration import compute_registration_transform
from lab2.src.utils import euler_from_direction

if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lab3_dir = os.path.join(script_dir, '..')
    img_dir = os.path.join(lab3_dir, 'images')
    img_cam1_path = os.path.join(img_dir, 'calib_imgs', 'imageL.png')
    img_cam2_path = os.path.join(img_dir, 'calib_imgs', 'imageR.png')

    # Get camera point
    pts_cam1 = get_points(img_cam1_path)
    pts_cam2 = get_points(img_cam2_path)

    # Define world points
    pts_world = np.array(
        [ #  (x_max, y_max, 0)                                        (x_max, y_min, 0)
            [[287.5, 287.5, 0], [287.5, 262.5, 0], [287.5, 237.5, 0], [287.5, 212.5, 0]], 
            [[262.5, 287.5, 0], [262.5, 262.5, 0], [262.5, 237.5, 0], [262.5, 212.5, 0]],
            [[237.5, 287.5, 0], [237.5, 262.5, 0], [237.5, 237.5, 0], [237.5, 212.5, 0]],
            [[212.5, 287.5, 0], [212.5, 262.5, 0], [212.5, 237.5, 0], [212.5, 212.5, 0]],
        ] #  (x_min, y_max, 0)                                        (x_min, y_min, 0)
    )

    # Compute calibration matrices
    calib_mat_cam1 = get_calib_mat(pts_cam1, pts_world)
    calib_mat_cam2 = get_calib_mat(pts_cam2, pts_world)

    # Get camera positions in world coordinates

    # Registration

    # Show augmented images (highlight obstacle and target)

    # Print directions
    