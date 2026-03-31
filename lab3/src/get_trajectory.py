import numpy as np
import os
import sys

# Calcule le chemin vers la racine du projet (deux niveaux au-dessus de src)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Ajoute ce chemin au système pour que Python voie le dossier 'lab2'
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from phantom_params import *
from point_selection import get_grid_points, get_ball_points, get_phantom_points
from reconstruction import get_calib_mat, get_camera_center, triangulate_points
from lab2.src.registration import compute_registration_transform
from lab2.src.utils import euler_from_direction

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
    calib_img_cam1_path = os.path.join(img_dir, 'ex_imgs', 'AimgLb.png')
    calib_img_cam2_path = os.path.join(img_dir, 'ex_imgs', 'AimgRb.png')
    img_cam1_path = os.path.join(img_dir, 'ex_imgs', 'AimgLc.png')
    img_cam2_path = os.path.join(img_dir, 'ex_imgs', 'AimgRc.png')

    # Get camera point
    pts_grid_cam1 = get_grid_points(calib_img_cam1_path)
    pts_grid_cam2 = get_grid_points(calib_img_cam2_path)

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
    pts_ball_cam1 = get_ball_points(calib_img_cam1_path)
    pts_ball_cam2 = get_ball_points(calib_img_cam2_path)

    # Get camera positions in world coordinates
    C_cam1 = get_camera_center(calib_mat_cam1, pts_ball_cam1, pts_ball_world)
    C_cam2 = get_camera_center(calib_mat_cam2, pts_ball_cam2, pts_ball_world)

    print("Camera 1:", C_cam1)
    print("Camera 2:", C_cam2)

    # Get phantom points in cam coordinates
    pts_phantom_cam1 = get_phantom_points(img_cam1_path)
    pts_phantom_cam2 = get_phantom_points(img_cam2_path)


    # Triangulate the points to get phantom points in world coordinates
    pts_world = triangulate_points(pts_ball_cam1, pts_ball_cam2, C_cam1, C_cam2, calib_mat_cam1, calib_mat_cam2)
    print(f"Points triangulation: {pts_world}")
    box_corners_world = triangulate_points(C_cam1, pts_phantom_cam1, C_cam2, pts_phantom_cam2)


    # Compute the registration transform from phantom to world coordinates
    box_corners_phantom = np.array([box_back_right_phantom, box_back_left_phantom, box_front_right_phantom, box_front_left_phantom])
    T_reg = compute_registration_transform(
        A = box_corners_phantom,
        B = box_corners_world
    )
    R_reg = T_reg[0:3, 0:3]
    t_reg = T_reg[0:3, 3]

    # Register obstacle corners to world coordinates
    obs_top_right_world = R_reg @ obs_top_right_phantom + t_reg
    obs_top_left_world = R_reg @ obs_top_left_phantom + t_reg
    obs_bottom_right_world = R_reg @ obs_bottom_right_phantom + t_reg
    obs_bottom_left_world = R_reg @ obs_bottom_left_phantom + t_reg

    # Register target corners to world coordinates
    trg_top_right_world = R_reg @ trg_top_right_phantom + t_reg
    trg_top_left_world = R_reg @ trg_top_left_phantom + t_reg
    trg_bottom_right_world = R_reg @ trg_bottom_right_phantom + t_reg
    trg_bottom_left_world = R_reg @ trg_bottom_left_phantom + t_reg

    # Register trajectory points to world coordinates
    contact_point_world = R_reg @ contact_point_phantom + t_reg
    deflection_point_world = R_reg @ deflection_point_phantom + t_reg
    middle_point_world = R_reg @ middle_point_phantom + t_reg
    starting_point_world = R_reg @ starting_point_phantom + t_reg

    # Compute effector direction
    direction = middle_point_world - starting_point_world
    direction = direction / np.linalg.norm(direction)
    effector_angles = euler_from_direction(direction)
    
    # Show augmented images (highlight obstacle and target)

    # Print directions
    instructions = f"""
SetTRF(0,0,71.3,0,0,0)
SetJointVel(10)
SetCartLinVel(10)
SetCartAngVel(15)
MovePose({starting_point_world[0]:.2f},{starting_point_world[1]:.2f},{starting_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
Delay(3)
MoveLin({middle_point_world[0]:.2f},{middle_point_world[1]:.2f},{middle_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
MoveLin({contact_point_world[0]:.2f},{contact_point_world[1]:.2f},{contact_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
Delay(1)
SetCartLinVel(5)
MoveLin({deflection_point_world[0]:.2f},{deflection_point_world[1]:.2f},{deflection_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
Delay(5)
MoveLin({contact_point_world[0]:.2f},{contact_point_world[1]:.2f},{contact_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
MoveLin({middle_point_world[0]:.2f},{middle_point_world[1]:.2f},{middle_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
SetCartLinVel(10)
MoveLin({starting_point_world[0]:.2f},{starting_point_world[1]:.2f},{starting_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
MoveJoints(0,0,0,0,0,0)
        """

    return instructions

if __name__ == "__main__":
    get_trajectory()
    