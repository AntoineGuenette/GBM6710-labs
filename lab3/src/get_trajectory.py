import numpy as np
import os
import cv2

from lab3.src.phantom_params import *
from lab3.src.point_selection import generate_grid_world, get_grid_points, get_ball_points, get_phantom_points
from lab3.src.reconstruction import (get_calib_mat, get_camera_center, triangulate_points, project_points,
                                     plot_3D_results, plot_3D_results_cam_calib)
from lab2.src.registration import compute_registration_transform
from lab2.src.utils import euler_from_direction

def compute_safe_rotation(contact_point_world: np.ndarray, direction: np.ndarray) -> float:
    """
    Compute a safe base rotation angle (joint 1) by finding the intersection between a ray and the
    XY axes.

    The ray originates from the contact point and follows the opposite of the effector direction. 
    The closest intersection with either the X or Y axis determines a safe quadrant-based rotation.

    Parameters:
        contact_point_world (np.ndarray): 3D coordinates of the contact point.
        direction (np.ndarray): Unit direction vector of the end-effector.

    Returns:
        float: Safe rotation angle in degrees for joint 1.
    """

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
        return 0  # fallback

    t_min, axis, coord = min(t_candidates, key=lambda x: x[0])

    if axis == 'x':  # crossed y = 0 → X axis
        return 0 if coord >= 0 else 170
    else:  # crossed x = 0 → Y axis
        return 90 if coord >= 0 else -90


def get_trajectory(
        xmax_ymax_grid_point_world:tuple,
        xmax_ymax_ball_point_world:list,
        xmax_ymin_ball_point_world:list,
        xmin_ymax_ball_point_world:list,
        side:str
    ) -> str:
    """
    Compute a robotic trajectory using stereo vision, 3D reconstruction, and registration.

    Parameters:
        xmax_ymax_grid_point_world (tuple): (x, y) coordinates of the top-right grid point in the
            world frame.
        xmax_ymax_ball_point_world (list): [x, y, z] coordinates of the top-right reference ball in
            the world frame.
        xmax_ymin_ball_point_world (list): [x, y, z] coordinates of the bottom-right reference ball
            in the world frame.
        xmin_ymax_ball_point_world (list): [x, y, z] coordinates of the top-left reference ball in
            the world frame.
        side (str): side to approach the phantom (left, right or center)

    Returns:
        str: A formatted string containing the sequence of instructions for the Meca500 robotic arm.
    """

    # Define constants
    EFFECTOR_WIDTH = 9 # mm
    DEFLECTION_HEIGHT = 6 # mm

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
    pts_grid_world = generate_grid_world(xmax_ymax_grid_point_world[0], xmax_ymax_grid_point_world[1])
    pts_ball_world = np.array(
        [xmax_ymax_ball_point_world,
         xmax_ymin_ball_point_world,
         xmin_ymax_ball_point_world]
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

    plot_3D_results_cam_calib(C_cam1, C_cam2)

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

    # Register box bottom interior corners to world coordinates
    box_front_interior_right_world = R_reg @ box_front_interior_right_phantom + t_reg
    box_front_interior_left_world = R_reg @ box_front_interior_left_phantom + t_reg

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

    # Register contact point to world coordinates
    contact_point_world = R_reg @ contact_point_phantom + t_reg

    # Compute trajectory points
    deflection_point_world = contact_point_world - np.array([0.0, 0.0, DEFLECTION_HEIGHT])
    middle_point_world = contact_point_world + np.array([0.0, 0.0, 1.25 * EFFECTOR_WIDTH])

    # Compute adaptive starting point based on accessible triangle
    z_contact = contact_point_world[2]

    left_proj = box_front_interior_left_world.copy()
    right_proj = box_front_interior_right_world.copy()
    left_proj[2] = z_contact
    right_proj[2] = z_contact

    # Shrink triangle by moving interior corners inward to clear the effector
    inward_dir = box_front_interior_left_world - box_front_interior_right_world
    inward_dir = inward_dir / np.linalg.norm(inward_dir)

    # Move both corners inward
    left_proj = left_proj - 1.5 * EFFECTOR_WIDTH * inward_dir
    right_proj = right_proj + 1.5 * EFFECTOR_WIDTH * inward_dir

    # Recompute directions with adjusted corners
    dir_left = left_proj - contact_point_world
    dir_right = right_proj - contact_point_world

    # Normalize directions
    dir_left = dir_left / np.linalg.norm(dir_left)
    dir_right = dir_right / np.linalg.norm(dir_right)

    # Choose direction (left, right, or center)
    if side == 'l':
        direction_choice = dir_left
        STARTING_DISTANCE = 75 # mm
    elif side == 'r':
        direction_choice = dir_right
        STARTING_DISTANCE = 75 # mm
    elif side == 'c':
        direction_choice = (dir_left + dir_right) / np.linalg.norm(dir_left + dir_right)
        STARTING_DISTANCE = 50 # mm
    else:
        raise ValueError("side must be 'l', 'r', or 'c'")

    # Compute starting point
    starting_point_world = contact_point_world + STARTING_DISTANCE * direction_choice

    # Adjust Z height to match obstacle bottom average
    z_obs = 0.5 * (obs_bottom_right_world[2] + obs_bottom_left_world[2])
    starting_point_world[2] = z_obs

    # Compute effector direction
    direction = middle_point_world - starting_point_world
    direction = direction / np.linalg.norm(direction)
    effector_angles = euler_from_direction(direction)

    # Compute safe rotation angle for joint 1 using axis intersection (ray from contact point)
    safe_rot = compute_safe_rotation(contact_point_world, direction)

    # Find the obstacle corners in each camera coordinates
    obs_world = np.array([
        obs_top_left_world,
        obs_top_right_world,
        obs_bottom_right_world,
        obs_bottom_left_world
    ])
    obs_img_cam1 = project_points(obs_world, C_cam1, calib_mat_cam1)
    obs_img_cam2 = project_points(obs_world, C_cam2, calib_mat_cam2)

    # Find the target corners in each camera coordinates
    trg_world = np.array([
        trg_top_left_world,
        trg_top_right_world,
        trg_bottom_right_world,
        trg_bottom_left_world
    ])
    trg_img_cam1 = project_points(trg_world, C_cam1, calib_mat_cam1)
    trg_img_cam2 = project_points(trg_world, C_cam2, calib_mat_cam2)

    plot_3D_results(C_cam1, C_cam2, box_corners_world, pts_ball_world)

    # Show augmented images (highlight obstacle and target with filled polygons)
    img_cam1 = cv2.imread(img_cam1_path, cv2.IMREAD_GRAYSCALE)
    img_cam2 = cv2.imread(img_cam2_path, cv2.IMREAD_GRAYSCALE)

    # Convert grayscale to BGR so we can overlay colors
    img_cam1 = cv2.cvtColor(img_cam1, cv2.COLOR_GRAY2BGR)
    img_cam2 = cv2.cvtColor(img_cam2, cv2.COLOR_GRAY2BGR)

    # Convert to OpenCV polygon format (int32 and shape Nx1x2)
    obs_cam1_int = obs_img_cam1.astype(np.int32).reshape((-1,1,2))
    trg_cam1_int = trg_img_cam1.astype(np.int32).reshape((-1,1,2))
    obs_cam2_int = obs_img_cam2.astype(np.int32).reshape((-1,1,2))
    trg_cam2_int = trg_img_cam2.astype(np.int32).reshape((-1,1,2))

    # Draw colored polygons on grayscale background
    cv2.fillPoly(img_cam1, [obs_cam1_int], (0, 0, 255))
    cv2.fillPoly(img_cam1, [trg_cam1_int], (0, 255, 0))

    cv2.fillPoly(img_cam2, [obs_cam2_int], (0, 0, 255))
    cv2.fillPoly(img_cam2, [trg_cam2_int], (0, 255, 0))

    cv2.imshow("Camera 1", img_cam1)
    cv2.waitKey(50)
    cv2.imshow("Camera 2", img_cam2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print directions
    instructions = f"""
SetTRF(0,0,71.3,0,0,0)
SetJointVel(10)
SetCartLinVel(10)
SetCartAngVel(15)
MoveJoints({safe_rot},0,0,0,0,0)
MovePose({starting_point_world[0]:.2f},{starting_point_world[1]:.2f},{starting_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
Delay(1)
MoveLin({middle_point_world[0]:.2f},{middle_point_world[1]:.2f},{middle_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
MoveLin({contact_point_world[0]:.2f},{contact_point_world[1]:.2f},{contact_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
SetCartLinVel(5)
MoveLin({deflection_point_world[0]:.2f},{deflection_point_world[1]:.2f},{deflection_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
Delay(3)
MoveLin({contact_point_world[0]:.2f},{contact_point_world[1]:.2f},{contact_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
MoveLin({middle_point_world[0]:.2f},{middle_point_world[1]:.2f},{middle_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
SetCartLinVel(10)
MoveLin({starting_point_world[0]:.2f},{starting_point_world[1]:.2f},{starting_point_world[2]:.2f},{effector_angles[0]:.2f},{effector_angles[1]:.2f},{effector_angles[2]:.2f})
MoveJoints({safe_rot},0,0,0,0,0)
MoveJoints(0,0,0,0,0,0)
        """

    return instructions

if __name__ == "__main__":

    print("\n --- Meca500 Trajectory Computing --- \n")

    # Select side
    side = input("Please entre the side of the approach (l, r or c): ")
    while side not in ['l', 'r', 'c']:
        side = input("Please entre the side of the approach (l, r or c): ")

    # Compute trajectory
    instructions = get_trajectory(
        xmax_ymax_grid_point_world=(262.5, 262.5),
        xmax_ymax_ball_point_world=[287.5, 287.5, 95.0],
        xmax_ymin_ball_point_world=[285.5, 37.5, 145.0],
        xmin_ymax_ball_point_world=[37.5, 287.5, 145.0],
        side=side
    )

    print(f"\nTRAJECTORY INSTRUCTIONS FOR MECA500 ROBOTIC ARM:\n{instructions}")