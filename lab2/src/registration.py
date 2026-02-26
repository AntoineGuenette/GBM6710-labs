import numpy as np
from scipy.optimize import least_squares

def compute_registration_transform(
        a_1: np.array,
        a_2: np.array, 
        a_3: np.array, 
        b_1: np.array, 
        b_2: np.array, 
        b_3: np.array,
    ) -> np.array:
    """
    Compute the rigid registration transform (rotation and translation) from initial to target
    coordinates using a closed-form SVD solution.

    If the determinant of the computed rotation matric is not 1, a least squares solution is computed
    to find the closest valid rotation matrix.

    Parameters:
        a_1 (np.array): First point in the initial coordinate system (mm).
        a_2 (np.array): Second point in the initial coordinate system (mm).
        a_3 (np.array): Third point in the initial coordinate system (mm).
        b_1 (np.array): First point in the target coordinate system (mm).
        b_2 (np.array): Second point in the target coordinate system (mm).
        b_3 (np.array): Third point in the target coordinate system (mm).

    Returns:
        T (np.array): A 4x4 homogeneous transformation matrix representing the rigid registration
        from the initial to the target coordinate system.
    """

    # Construct matrices A and B from the coordinates of the points
    A = np.stack([a_1, a_2, a_3], axis=1)
    B = np.stack([b_1, b_2, b_3], axis=1)

    # Compute centroids
    a_bar = np.mean(A, axis=1, keepdims=True)
    b_bar = np.mean(B, axis=1, keepdims=True)

    # Compute variation from centroids
    A_tilde = A - a_bar
    B_tilde = B - b_bar

    # Compute the cross-covariance matrix and perform SVD
    H = A_tilde @ B_tilde.T
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T
    # If the determinant of R is negative, correct for a reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Verify that R is a valid rotation matrix
    # If not, compute the closest valid rotation matrix using least squares
    if not np.isclose(np.linalg.det(R), 1.0):
        print("\nWarning: Computed rotation matrix is not valid.\
              Optimizing to find the closest valid rotation matrix.\n")
        R = optimize_rotation_matrix(A_tilde, B_tilde)
        
    # Compute translation vector
    p = (b_bar - R @ a_bar).flatten()

    # Construct the homogeneous transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p

    return T

def optimize_rotation_matrix(A_tilde: np.array, B_tilde: np.array) -> np.array:
    """
    Compute the closest valid rotation matrix to a given matrix using least squares optimization.
    """
    def residuals(r_vec):
        # r_vec is a 9-element vector representing a 3x3 matrix
        R = r_vec.reshape(3, 3)

        # Enforce orthogonality softly via residuals
        ortho_res = (R.T @ R - np.eye(3)).ravel()

        # Enforce det(R) = 1 softly
        det_res = np.array([np.linalg.det(R) - 1.0])

        # Data fitting residuals
        data_res = (R @ A_tilde - B_tilde).ravel()

        return np.concatenate([data_res, ortho_res, det_res])

    # Initial guess: identity rotation
    r0 = np.eye(3).ravel()

    # Least squares optimization
    result = least_squares(residuals, r0)

    # Reshape the optimized vector back to a 3x3 matrix
    R_opt = result.x.reshape(3, 3)

    return R_opt
