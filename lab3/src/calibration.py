import numpy as np

def grid_to_vec(world_points):
    (rows,col,_) = world_points.shape

    coords = np.zeros((rows*col,2))
    for k in range(rows):
        for l in range(col):
            index = k * col + l
            coords[index,0] = world_points[k,l,0]
            coords[index,1] = world_points[k,l,1]
    return coords

def get_calib_mat(camera_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    # Convert the points in a grid form to a vector form
    world_points = grid_to_vec(world_points)
    camera_points = grid_to_vec(camera_points)

    # Get coordinate vectors
    m_vector = camera_points[:,0]
    n_vector = camera_points[:,1]
    x_vector = world_points[:,0]
    y_vector = world_points[:,1]

    # Compute calibration matrix (U)
    M = np.stack(arrays = (m_vector ** 2, m_vector * n_vector, n_vector ** 2, m_vector, n_vector, np.ones_like(m_vector)), axis =1 )
    R = np.stack(arrays = (x_vector, y_vector), axis = 1)
    U = np.linalg.inv(M.T @ M) @ M.T @ R

    return U

def get_camera_center(U: np.ndarray,
                      ball_img_pts: np.ndarray,
                      ball_world_pts: np.ndarray) -> np.ndarray:
    """
    Estimate the camera center using image points and corresponding 3D world points by
    intersecting projection rays in a least-squares sense.

    Parameters:
        U (np.ndarray): Calibration matrix obtained from planar calibration (not used directly
                        for inversion, only kept for consistency with pipeline).
        ball_img_pts (np.ndarray): Array of shape (N, 2) containing 2D image coordinates (u, v).
        ball_world_pts (np.ndarray): Array of shape (N, 3) containing corresponding 3D world
                                    coordinates (X, Y, Z).

    Returns:
        C (np.ndarray): Estimated camera center in world coordinates (shape: (3,)).
    """

    # Convert inputs to numpy arrays (safety)
    ball_img_pts = np.asarray(ball_img_pts)
    ball_world_pts = np.asarray(ball_world_pts)

    rays = []
    points = []

    # Build rays from image points (pinhole approximation)
    for (u, v), Pw in zip(ball_img_pts, ball_world_pts):
        # Direction vector in camera frame
        d = np.array([u, v, 1.0])
        d = d / np.linalg.norm(d)

        rays.append(d)
        points.append(Pw)

    rays = np.array(rays)
    points = np.array(points)

    # Solve least squares intersection of rays
    A = np.zeros((3, 3))
    b = np.zeros(3)

    for d, p in zip(rays, points):
        I = np.eye(3)
        A += I - np.outer(d, d)
        b += (I - np.outer(d, d)) @ p

    # Solve system
    C = np.linalg.solve(A, b)

    return C

def triangulate_points():
    # to be implemented
    pass