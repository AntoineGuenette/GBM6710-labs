import numpy as np
from scipy.spatial import KDTree

def grid_to_vec(points):
    """
    Convert a grid of 2D points into a vectorized list of coordinates.

    Parameters:
        points (np.ndarray): Array of shape (rows, cols, 2) containing 2D points.

    Returns:
        coords (np.ndarray): Array of shape (rows*cols, 2) containing the flattened list of
        coordinates.
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
    Compute the calibration matrix relating camera coordinates to world coordinates using a
    quadratic model and least squares estimation.

    Parameters:
        camera_points (np.ndarray): Array of shape (rows, cols, 2) containing 2D image coordinates.
        world_points (np.ndarray): Array of shape (rows, cols, 2) containing corresponding 2D world
            coordinates.

    Returns:
        U (np.ndarray): Calibration matrix of shape (6, 2) mapping image coordinates to world
        coordinates.
    """
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

def get_camera_center(U, ball_img_pts, ball_world_pts):
    """
    Estimate camera center: C is the point such that for each ball,
    the ray from C through P_plane (backprojected image point on Z=0)
    passes through Pw (3D world position of the ball).
    
    Constraint: C, P_plane, Pw are collinear
    => (Pw - C) × (P_plane - C) = 0
    Which linearizes to solving a least-squares system.
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
        
        # Direction: de P_plane vers Pw (connue sans C)
        d = Pw - P_plane
        d = d / np.linalg.norm(d)

        # Contrainte: C est sur la droite passant par Pw de direction d
        # => minimiser ||(I - d dᵀ)(C - Pw)||²
        I = np.eye(3)
        M = I - np.outer(d, d)
        A += M
        b += M @ Pw  # = M @ Pw car M @ Pw est le terme constant

    C = np.linalg.solve(A, b)
    return C

def triangulate_points(pts1, pts2, C1, C2, calib_mat1, calib_mat2):
    """
    Triangulate 3D points from corresponding 2D image points.
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

        # direction from camera toward projected point on plane
        dir_vec = np.array([x, y, 0.0])  # plan Z=0 (cohérent calibration)
        return dir_vec

    points_3d = []

    for p1, p2 in zip(pts1, pts2):

        # Directions
        d1 = backproject(calib_mat1, p1) - C1
        d2 = backproject(calib_mat2, p2) - C2

        d1 /= np.linalg.norm(d1)
        d2 /= np.linalg.norm(d2)

        # Solve closest points between rays
        A = np.stack([d1, -d2], axis=1)
        b = C2 - C1

        t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        P1 = C1 + t[0] * d1
        P2 = C2 + t[1] * d2

        # midpoint
        P = (P1 + P2) / 2
        points_3d.append(P)

    return np.array(points_3d)

def project_points(world_points, C, calib_mat):
    """
    Project 3D world points into image coordinates (u, v).

    Parameters:
        world_points (np.ndarray): Array of shape (N, 3) of 3D world points.
        C (np.ndarray): Camera center of shape (3,).
        calib_mat (np.ndarray): Calibration matrix (6x2).

    Returns:
        img_points (np.ndarray): Array of shape (N, 2) of image coordinates (u, v).
    """

    def forward_model(m, n, U):
        Ax, Bx, Cx, Dx, Ex, Fx = U[:, 0]
        Ay, By, Cy, Dy, Ey, Fy = U[:, 1]

        x = Ax*m**2 + Bx*m*n + Cx*n**2 + Dx*m + Ex*n + Fx
        y = Ay*m**2 + By*m*n + Cy*n**2 + Dy*m + Ey*n + Fy

        return np.array([x, y, 0.0])

    img_points = []

    for Pw in world_points:
        # Direction from camera to point
        d = Pw - C
        d = d / np.linalg.norm(d)

        # We find (m, n) such that forward_model(m,n) lies on this ray
        # Solve via nonlinear least squares (simple iterative search)

        def residual(p):
            m, n = p
            P_plane = forward_model(m, n, calib_mat)
            ray_dir = P_plane - C
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            return ray_dir - d

        # Initial guess
        p = np.array([0.0, 0.0])

        # Simple Gauss-Newton iterations
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