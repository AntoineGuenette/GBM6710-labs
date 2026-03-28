import numpy as np

import numpy as np

def get_A_matrix(coord_im: np.ndarray, coord_world: np.ndarray) -> np.ndarray:
    """
    Build the system matrix A used in the Direct Linear Transform (DLT) method for camera calibration.

    Parameters:
        coord_im (np.ndarray): Image points of shape (N, 2) containing pixel coordinates (u, v).
        coord_world (np.ndarray): World points of shape (N, 3) containing coordinates (X, Y, Z).

    Returns:
        A (np.ndarray): Matrix of shape (2N, 12) used to solve for the DLT parameters.
    """

    coord_im = coord_im.reshape(-1, 2)
    coord_world = coord_world.reshape(-1, 3)

    N = coord_im.shape[0]
    A = []

    for i in range(N):
        u, v = coord_im[i]
        X, Y, Z = coord_world[i]

        Xi = np.array([X, Y, Z, 1])

        A.append(np.hstack([Xi, np.zeros(4), -u * Xi]))
        A.append(np.hstack([np.zeros(4), Xi, -v * Xi]))

    return np.array(A)

def get_DLT_parameters(A: np.ndarray) -> np.ndarray:
    """
    Compute the DLT parameters by solving the homogeneous system Ap = 0 using Singular Value Decomposition (SVD).

    Parameters:
        A (np.ndarray): System matrix of shape (2N, 12) built from 2D–3D correspondences.

    Returns:
        p (np.ndarray): Vector of shape (12,) containing the normalized DLT parameters.
    """

    _, _, Vt = np.linalg.svd(A)
    p = Vt[-1]

    # Normalization (12th DLT param is 1)
    p = p / p[-1]

    return p

def get_calib_parameters(dlt_params: np.ndarray) -> dict:
    """
    Extract intrinsic and extrinsic camera parameters from the DLT parameter vector.

    Parameters:
        dlt_params (np.ndarray): Vector of shape (12,) containing the DLT parameters.

    Returns:
        out (dict): A dictionary containing:
            - 'u0' (float): Principal point x-coordinate.
            - 'v0' (float): Principal point y-coordinate.
            - 'c_u' (float): Focal length in x (pixels).
            - 'c_v' (float): Focal length in y (pixels).
            - 'R' (np.ndarray): Rotation matrix (3, 3).
            - 'C' (np.ndarray): Camera center in world coordinates (3,).
    """

    out = {}

    # Normalization
    d = -1 / np.sqrt(dlt_params[8]**2 + dlt_params[9]**2 + dlt_params[10]**2)

    # Center of image
    u0 = (dlt_params[0]*dlt_params[8] +
          dlt_params[1]*dlt_params[9] +
          dlt_params[2]*dlt_params[10]) * d**2

    v0 = (dlt_params[4]*dlt_params[8] +
          dlt_params[5]*dlt_params[9] +
          dlt_params[6]*dlt_params[10]) * d**2

    # Focal lengths
    c_u = np.sqrt(d**2 * ((u0*dlt_params[8] - dlt_params[0])**2 +
                          (u0*dlt_params[9] - dlt_params[1])**2 +
                          (u0*dlt_params[10] - dlt_params[2])**2))

    c_v = np.sqrt(d**2 * ((v0*dlt_params[8] - dlt_params[4])**2 +
                          (v0*dlt_params[9] - dlt_params[5])**2 +
                          (v0*dlt_params[10] - dlt_params[6])**2))

    # Rotation matrix
    R = np.array([
        [d/c_u * (u0*dlt_params[8] - dlt_params[0]),
         d/c_u * (u0*dlt_params[9] - dlt_params[1]),
         d/c_u * (u0*dlt_params[10] - dlt_params[2])],

        [d/c_v * (v0*dlt_params[8] - dlt_params[4]),
         d/c_v * (v0*dlt_params[9] - dlt_params[5]),
         d/c_v * (v0*dlt_params[10] - dlt_params[6])],

        [dlt_params[8]*d,
         dlt_params[9]*d,
         dlt_params[10]*d]
    ])

    # Camera coordinates
    M = np.array([
        [dlt_params[0], dlt_params[1], dlt_params[2]],
        [dlt_params[4], dlt_params[5], dlt_params[6]],
        [dlt_params[8], dlt_params[9], dlt_params[10]]
    ])
    p4 = np.array([
        [-dlt_params[3]],
        [-dlt_params[7]],
        [-1]
    ])
    C = np.linalg.inv(M) @ p4

    # Format in a dictionnary
    out['u0'] = u0
    out['v0'] = v0
    out['c_u'] = c_u
    out['c_v'] = c_v
    out['R'] = R
    out['C'] = C.flatten()

    return out

def calibrate_camera(coord_im: np.ndarray, coord_world: np.ndarray) -> dict:
    """
    Perform full camera calibration using the Direct Linear Transform (DLT) method.

    This function builds the system matrix A from 2D–3D correspondences, solves for the
    DLT parameters using SVD, and extracts intrinsic and extrinsic camera parameters.

    Parameters:
        coord_im (np.ndarray): Image points of shape (N, 2) containing pixel coordinates (u, v).
        coord_world (np.ndarray): World points of shape (N, 3) containing coordinates (X, Y, Z).

    Returns:
        calib (dict): A dictionary containing:
            - 'P' (np.ndarray): Projection matrix of shape (3, 4).
            - 'dlt_params' (np.ndarray): Flattened DLT parameters (12,).
            - 'u0' (float): Principal point x-coordinate.
            - 'v0' (float): Principal point y-coordinate.
            - 'c_u' (float): Focal length in x (pixels).
            - 'c_v' (float): Focal length in y (pixels).
            - 'R' (np.ndarray): Rotation matrix (3, 3).
            - 'C' (np.ndarray): Camera center in world coordinates (3,).
    """

    # Step 1: Build A matrix
    A = get_A_matrix(coord_im, coord_world)

    # Step 2: Solve for DLT parameters
    dlt_params = get_DLT_parameters(A)

    # Step 3: Extract calibration parameters
    params = get_calib_parameters(dlt_params)

    # Step 4: Build projection matrix
    P = dlt_params.reshape(3, 4)

    # Combine all outputs
    calib = {
        'P': P,
        'dlt_params': dlt_params,
        'u0': params['u0'],
        'v0': params['v0'],
        'c_u': params['c_u'],
        'c_v': params['c_v'],
        'R': params['R'],
        'C': params['C']
    }

    return calib
