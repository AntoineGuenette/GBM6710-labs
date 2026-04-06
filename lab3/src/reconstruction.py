import numpy as np
import matplotlib.pyplot as plt

def grid_to_vec(points):
    """
    Convert a grid of 2D points into a vectorized list of coordinates.

    Parameters:
        points (np.ndarray): Array of shape (rows, cols, 2) containing 2D points.

    Returns:
        coords (np.ndarray): Array of shape (rows*cols, 2) containing flattened coordinates.
    """
    (rows,col,_) = points.shape

    coords = np.zeros((rows*col,2))
    for k in range(rows):
        for l in range(col):
            index = k * col + l
            coords[index,0] = points[k,l,0]
            coords[index,1] = points[k,l,1]
    return coords

def get_calib_mat(camera_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    """
    Compute the calibration matrix relating image coordinates to world coordinates using a
    quadratic model and least squares.

    Parameters:
        camera_points (np.ndarray): Array of shape (rows, cols, 2) of image coordinates.
        world_points (np.ndarray): Array of shape (rows, cols, 3) of world coordinates.

    Returns:
        U (np.ndarray): Calibration matrix of shape (6, 2).
    """
    # Convert grid-form points to vector form
    world_points = grid_to_vec(world_points)
    camera_points = grid_to_vec(camera_points)

    # Extract coordinate vectors
    m_vector = camera_points[:,0]
    n_vector = camera_points[:,1]
    x_vector = world_points[:,0]
    y_vector = world_points[:,1]

    # Compute calibration matrix U
    M = np.stack(arrays = (m_vector ** 2, m_vector * n_vector, n_vector ** 2, m_vector, n_vector, np.ones_like(m_vector)), axis =1 )
    R = np.stack(arrays = (x_vector, y_vector), axis = 1)
    U = np.linalg.inv(M.T @ M) @ M.T @ R

    return U

def get_camera_center(U, ball_img_pts, ball_world_pts):
    """
    Estimate the camera center using known correspondences between image points and 3D world points.

    Parameters:
        U (np.ndarray): Calibration matrix.
        ball_img_pts (np.ndarray): Array of shape (N, 2) of image points.
        ball_world_pts (np.ndarray): Array of shape (N, 3) of corresponding world points.

    Returns:
        C (np.ndarray): Estimated camera center of shape (3,).
    """
    
    def backproject(U, pt):
        m, n = pt
        Ax, Bx, Cx, Dx, Ex, Fx = U[:, 0]
        Ay, By, Cy, Dy, Ey, Fy = U[:, 1]
        x = Ax*m**2 + Bx*m*n + Cx*n**2 + Dx*m + Ex*n + Fx
        y = Ay*m**2 + By*m*n + Cy*n**2 + Dy*m + Ey*n + Fy
        return np.array([x, y, 0.0])

    A = np.zeros((3, 3))
    b = np.zeros(3)

    for (u, v), Pw in zip(ball_img_pts, ball_world_pts):
        P_plane = backproject(U, (u, v))
        
        # Direction from P_plane to Pw (known without C)
        d = Pw - P_plane
        d = d / np.linalg.norm(d)

        # Constraint: C lies on the line passing through Pw with direction d
        # => minimize ||(I - d dᵀ)(C - Pw)||²
        I = np.eye(3)
        M = I - np.outer(d, d)
        A += M
        b += M @ Pw  # = M @ Pw because M @ Pw is the constant term

    C = np.linalg.solve(A, b)
    return C

def triangulate_points(pts1, pts2, C1, C2, calib_mat1, calib_mat2):
    """
    Triangulate 3D points from corresponding 2D image points in two views.

    Parameters:
        pts1 (np.ndarray): Array of shape (N, 2) of image points in camera 1.
        pts2 (np.ndarray): Array of shape (N, 2) of image points in camera 2.
        C1 (np.ndarray): Camera 1 center.
        C2 (np.ndarray): Camera 2 center.
        calib_mat1 (np.ndarray): Calibration matrix for camera 1.
        calib_mat2 (np.ndarray): Calibration matrix for camera 2.

    Returns:
        points_3d (np.ndarray): Array of shape (N, 3) of triangulated 3D points.
    """

    def backproject(calib_mat, pt):
        """
        Convert image point to a direction vector in world space.
        """
        m, n = pt

        Ax, Bx, Cx, Dx, Ex, Fx = calib_mat[:, 0]
        Ay, By, Cy, Dy, Ey, Fy = calib_mat[:, 1]

        x = Ax*m**2 + Bx*m*n + Cx*n**2 + Dx*m + Ex*n + Fx
        y = Ay*m**2 + By*m*n + Cy*n**2 + Dy*m + Ey*n + Fy

        # Direction from camera toward projected point on plane
        dir_vec = np.array([x, y, 0.0])  # Z=0 plane (consistent with calibration)
        return dir_vec

    points_3d = []

    for p1, p2 in zip(pts1, pts2):

        # Compute ray directions
        d1 = backproject(calib_mat1, p1) - C1
        d2 = backproject(calib_mat2, p2) - C2

        d1 /= np.linalg.norm(d1)
        d2 /= np.linalg.norm(d2)

        # Solve for closest points between rays
        A = np.stack([d1, -d2], axis=1)
        b = C2 - C1

        t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        P1 = C1 + t[0] * d1
        P2 = C2 + t[1] * d2

        # Compute midpoint
        P = (P1 + P2) / 2
        points_3d.append(P)

    return np.array(points_3d)

def project_points(world_points, C, calib_mat):
    """
    Project 3D world points into image coordinates.

    Parameters:
        world_points (np.ndarray): Array of shape (N, 3) of world points.
        C (np.ndarray): Camera center.
        calib_mat (np.ndarray): Calibration matrix.

    Returns:
        img_points (np.ndarray): Array of shape (N, 2) of image coordinates.
    """

    def forward_model(m, n, U):
        Ax, Bx, Cx, Dx, Ex, Fx = U[:, 0]
        Ay, By, Cy, Dy, Ey, Fy = U[:, 1]

        x = Ax*m**2 + Bx*m*n + Cx*n**2 + Dx*m + Ex*n + Fx
        y = Ay*m**2 + By*m*n + Cy*n**2 + Dy*m + Ey*n + Fy

        return np.array([x, y, 0.0])

    img_points = []

    for Pw in world_points:
        # Compute direction from camera to point
        d = Pw - C
        d = d / np.linalg.norm(d)

        # Find (m, n) such that forward_model(m, n) lies on this ray
        # Solve using nonlinear least squares (iterative method)

        def residual(p):
            m, n = p
            P_plane = forward_model(m, n, calib_mat)
            ray_dir = P_plane - C
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            return ray_dir - d

        # Initial guess
        p = np.array([0.0, 0.0])

        # Gauss-Newton iterations
        for _ in range(20):
            eps = 1e-6
            r = residual(p)

            J = np.zeros((3, 2))
            for i in range(2):
                dp = np.zeros(2)
                dp[i] = eps
                J[:, i] = (residual(p + dp) - r) / eps

            # Solve J * delta = -r
            delta, _, _, _ = np.linalg.lstsq(J, -r, rcond=None)
            p += delta

        img_points.append(p)

    return np.array(img_points)

def plot_3D_results(cam1: np.ndarray, cam2: np.ndarray, phantom_points: np.ndarray, ball_points_world: np.ndarray):

    def extract_3D_position(point: np.ndarray):
        x = point[:,0]
        y = point[:,1]
        z = point[:,2]
        return x,y,z
    # Retrieve center of cam points
    cam1_x, cam1_y, cam1_z = cam1[0], cam1[1], cam1[2]
    cam2_x, cam2_y, cam2_z = cam2[0], cam2[1], cam2[2]

    # Retrive phantom points
    phantom_pointsx, phantom_pointsy, phantom_pointsz = extract_3D_position(phantom_points)

    # Retrive ball points in world coordinates
    ball_pointsx, ball_pointsy, ball_pointsz = extract_3D_position(ball_points_world)


    fig  = plt.figure()
    ax = fig.add_subplot(projection = "3d")

    # Define grid points
    x = np.array([ (25 * i) + 12.5 for i in range(-12,11)])
    y = np.array([ (25 * i) + 12.5 for i in range(-12,11)])
    zero = np.array([0 for i in range(len(x))])
    x_zero = np.array([y[0] for i in range(len(x))])
    y_zero = np.array([x[0] for i in range(len(y))])

    # Plot grid points
    ax.scatter(x,y_zero,zero, c = "blue", marker = 'o', s = 25)
    ax.scatter(x_zero,y,zero, c = "blue", marker = 'o', s = 25)

    # Plot center of robot
    ax.scatter(0,0,0, c ="green", marker = 'o', s = 25 )
    
    # Plot camera center points
    ax.scatter(cam1_x, cam1_y, cam1_z, c = "red", marker = 'o', s = 25)
    ax.scatter(cam2_x, cam2_y, cam2_z, c = "red", marker = 'o', s = 25)

    # Plot phantom points
    ax.scatter(phantom_pointsx, phantom_pointsy, phantom_pointsz, c = "magenta", marker = 'o', s = 25)

    # Plot ball points
    ax.scatter(ball_pointsx, ball_pointsy, ball_pointsz, c = "orange", marker = 'o', s = 25)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()

def plot_3D_results_cam_calib(cam1: np.ndarray, cam2: np.ndarray):

    def extract_3D_position(point: np.ndarray):
        x = point[:,0]
        y = point[:,1]
        z = point[:,2]
        return x,y,z
    # Retrieve center of cam points
    cam1_x, cam1_y, cam1_z = cam1[0], cam1[1], cam1[2]
    cam2_x, cam2_y, cam2_z = cam2[0], cam2[1], cam2[2]

    fig  = plt.figure()
    ax = fig.add_subplot(projection = "3d")

    # Define grid points
    x = np.array([ (25 * i) + 12.5 for i in range(-12,11)])
    y = np.array([ (25 * i) + 12.5 for i in range(-12,11)])
    zero = np.array([0 for i in range(len(x))])
    x_min = np.array([y[0] for i in range(len(x))])
    x_max = np.array([y[-1] for i in range(len(x))])
    y_min = np.array([x[0] for i in range(len(y))])
    y_max = np.array([x[-1] for i in range(len(y))])

    # Plot grid points
    ax.scatter(x,y_min,zero, c = "blue", marker = 'o', s = 25)
    ax.scatter(x,y_max,zero, c = "blue", marker = 'o', s = 25)
    ax.scatter(x_max,y,zero, c = "blue", marker = 'o', s = 25)
    ax.scatter(x_min,y,zero, c = "blue", marker = 'o', s = 25)

    # Plot center of robot
    ax.scatter(0,0,0, c ="green", marker = 'o', s = 25 )
    
    # Plot camera center points
    ax.scatter(cam1_x, cam1_y, cam1_z, c = "red", marker = 'o', s = 25)
    ax.scatter(cam2_x, cam2_y, cam2_z, c = "red", marker = 'o', s = 25)


    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()