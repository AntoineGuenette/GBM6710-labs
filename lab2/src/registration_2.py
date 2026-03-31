import numpy as np
bead_1_position_phantom = np.array([18.99, -133.90, -227.54], dtype=float)
bead_2_position_phantom = np.array([-13.68, -132.96, -226.15], dtype=float)
bead_3_position_phantom = np.array([-14.61, -131.08, -124.10], dtype=float)
bead_1_position_world = np.array([61.784, 193.498, 54.029])
bead_2_position_world = np.array([63.716, 226.818, 53.5140])
bead_3_position_world = np.array([-39.159, 192.056, 54.646])
"""
 noir = bille 1
 noir = bille 2
 rouge = bille 3

"""
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
    def compute_a_tilde(initial_vect: list) -> np.array:
        a_bar = np.mean(initial_vect)
        a_tilde = initial_vect - a_bar
        return a_tilde, a_bar

    
    beads_phantom = [bead_1_position_phantom, bead_2_position_phantom, bead_3_position_phantom]
    beads_world = [bead_1_position_world, bead_2_position_world, bead_3_position_world]
    beads_phantom_tilde,a_bar = compute_a_tilde(beads_phantom)
    beads_world_tilde,b_bar = compute_a_tilde(beads_world)
    H = np.eye(3)
    for i in range(len(beads_phantom_tilde)):
        
    # Stack points into 3xN matrices
        A = np.stack([
            beads_phantom_tilde[i],
            beads_phantom_tilde[i],
            beads_phantom_tilde[i]
        ], axis=1)  # Phantom points

        B = np.stack([
            beads_world_tilde[i],
            beads_world_tilde[i],
            beads_world_tilde[i]
        ], axis=1)  # World points
        H_preliminary = A @ B.T

        H += H_preliminary
    # Compute centroids
    # a_bar = np.mean(A, axis=1, keepdims=True)
    # b_bar = np.mean(B, axis=1, keepdims=True)

    # # Center the points
    # A_tilde = A - a_bar
    # B_tilde = B - b_bar

    # # Compute cross-covariance matrix
    # H = A_tilde @ B_tilde.T

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    if np.linalg.det(R) == 1:
        print("WARNING!!!! det(R) = 1")
    # Compute translation

    p = (b_bar - R@a_bar)

    # Homogeneous transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    print(f"T: {T}")
    print(f"R: {R}")
    print(f"p: {p}")
    return {
        "R": R,
        "p": p,
        "T": T
    }
R, p, T = compute_registration_transform(bead_1_position_phantom, bead_2_position_phantom, bead_3_position_phantom, bead_1_position_world, bead_2_position_world, bead_3_position_world)
T_ = np.linalg.inv(T)

