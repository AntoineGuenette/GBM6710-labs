import numpy as np
import os

from lab3.src.phantom_params import *
from lab3.src.point_selection import get_grid_points, get_ball_points, get_phantom_points
from lab3.src.reconstruction import get_calib_mat, get_camera_center, triangulate_points
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
    calib_img_cam1_path = os.path.join(img_dir, 'imgs', 'calib_cam1.jpg')
    calib_img_cam2_path = os.path.join(img_dir, 'imgs', 'calib_cam2.jpg')
    img_cam1_path = os.path.join(img_dir, 'imgs', 'cam1.jpg')
    img_cam2_path = os.path.join(img_dir, 'imgs', 'cam2.jpg')

    # Get grid points in camera coordinates
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
        [[262.5, 262.5, 150.0], # (x_max, y_max, 150.0)
         [262.5, 37.5, 150.0], # (x_max, y_min, 150.0)
         [37.5, 262.5, 150.0]] # (x_min, y_max, 150.0)
    )

    # Compute calibration matrices
    calib_mat_cam1 = get_calib_mat(pts_grid_cam1, pts_grid_world)
    calib_mat_cam2 = get_calib_mat(pts_grid_cam2, pts_grid_world)

    # Get ball points in camera coordinates
    pts_ball_cam1 = get_ball_points(calib_img_cam1_path)
    pts_ball_cam2 = get_ball_points(calib_img_cam2_path)

    # Get camera positions in world coordinates
    C_cam1 = get_camera_center(calib_mat_cam1, pts_ball_cam1, pts_ball_world)
    C_cam2 = get_camera_center(calib_mat_cam2, pts_ball_cam2, pts_ball_world)

    # Get phantom points in camera coordinates
    pts_phantom_cam1 = get_phantom_points(img_cam1_path)
    pts_phantom_cam2 = get_phantom_points(img_cam2_path)

    # Triangulate the points to get phantom points in world coordinates
    box_corners_world = triangulate_points(
        pts_phantom_cam1,
        pts_phantom_cam2,
        C_cam1,
        C_cam2,
        calib_mat_cam1,
        calib_mat_cam2
    )

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

    # Compute safe rotation angle for joint 1 using axis intersection (ray from contact point)
    p = contact_point_world.copy()
    d = -direction.copy()

    # Work in XY plane
    p_xy = p[:2]
    d_xy = d[:2]

    eps = 1e-6
    t_candidates = []

    # Intersection with x = 0 (Y axis)
    if abs(d_xy[0]) > eps:
        t_x = -p_xy[0] / d_xy[0]
        if t_x > 0:
            y_at_tx = p_xy[1] + t_x * d_xy[1]
            t_candidates.append((t_x, 'y', y_at_tx))

    # Intersection with y = 0 (X axis)
    if abs(d_xy[1]) > eps:
        t_y = -p_xy[1] / d_xy[1]
        if t_y > 0:
            x_at_ty = p_xy[0] + t_y * d_xy[0]
            t_candidates.append((t_y, 'x', x_at_ty))

    # Select closest intersection
    if len(t_candidates) == 0:
        safe_rot = 0  # fallback
    else:
        t_min, axis, coord = min(t_candidates, key=lambda x: x[0])
        
        if axis == 'x':  # crossed y = 0 → X axis
            if coord >= 0:
                safe_rot = 0      # +X
            else:
                safe_rot = 170    # -X
        else:  # axis == 'y', crossed x = 0 → Y axis
            if coord >= 0:
                safe_rot = 90     # +Y
            else:
                safe_rot = -90    # -Y
    
    # Show augmented images (highlight obstacle and target)

    # Print directions
    instructions = f"""
SetTRF(0,0,71.3,0,0,0)
SetJointVel(10)
SetCartLinVel(10)
SetCartAngVel(15)
MoveJoints({safe_rot},0,0,0,0,0)
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
MoveJoints({safe_rot},0,0,0,0,0)
MoveJoints(0,0,0,0,0,0)
        """

    return instructions

if __name__ == "__main__":
    instructions = get_trajectory()

    print(f"\nTRAJECTORY INSTRUCTIONS FOR MECA500 ROBOTIC ARM:\n{instructions}")