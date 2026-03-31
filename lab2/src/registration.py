import numpy as np
from scipy.optimize import least_squares

def compute_registration_transform(
        A: np.array,
        B: np.array,
    ) -> np.array:
    """
    Compute the rigid registration transform (rotation and translation) from initial to target
    coordinates using a variable number of corresponding points. If the determinant of the computed
    rotation matrix is not 1, a least squares solution is computed to find the closest valid
    rotation matrix.

    Parameters:
        A (np.array): Array of shape (N, 3) containing points in the initial coordinate system (mm).
        B (np.array): Array of shape (N, 3) containing corresponding points in the target coordinate
            system (mm).

    Returns:
        T (np.array): A 4x4 homogeneous transformation matrix representing the rigid registration
        from the initial to the target coordinate system.
    """

    # Validate inputs
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape.")
    if A.shape[1] != 3:
        raise ValueError("Input point arrays must have shape (N, 3).")

    # Transpose to shape (3, N) for computation
    A = A.T
    B = B.T

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
    # If the determinant of R is negative, multiply the third column of V by -1 to force det(R)>0
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
