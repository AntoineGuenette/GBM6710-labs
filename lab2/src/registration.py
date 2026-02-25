import numpy as np

def compute_registration_transform(
        bead_1_position_phantom: np.array,
        bead_2_position_phantom: np.array, 
        bead_3_position_phantom: np.array, 
        bead_1_position_world: np.array, 
        bead_2_position_world: np.array, 
        bead_3_position_world: np.array,
    ) -> dict:
    """
    Compute the rigid registration transform (rotation and translation)
    from phantom to world coordinates using a closed-form SVD solution.
    """

    # Stack points into 3xN matrices
    A = np.stack([
        bead_1_position_phantom,
        bead_2_position_phantom,
        bead_3_position_phantom
    ], axis=1)  # Phantom points

    B = np.stack([
        bead_1_position_world,
        bead_2_position_world,
        bead_3_position_world
    ], axis=1)  # World points

    # Compute centroids
    a_bar = np.mean(A, axis=1, keepdims=True)
    b_bar = np.mean(B, axis=1, keepdims=True)

    # Center the points
    A_tilde = A - a_bar
    B_tilde = B - b_bar

    # Compute cross-covariance matrix
    H = A_tilde @ B_tilde.T

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    p = (b_bar - R @ a_bar).flatten()

    # Homogeneous transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p

    return {
        "R": R,
        "p": p,
        "T": T
    }
