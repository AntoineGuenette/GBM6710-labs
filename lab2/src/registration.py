import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation

from phantom_params import bead_1_position_phantom, bead_2_position_phantom, bead_3_position_phantom,bead_1_position_world, bead_2_position_world, bead_3_position_world

def compute_a_tilde(initial_vect: list) -> np.array:
        a_bar = 0
        for bead in initial_vect:
            a_bar += bead
        a_bar = a_bar/len(initial_vect)
        a_tilde = initial_vect - a_bar
        return a_tilde

def compute_b_hat(b_tilde: np.array, Rk: np.array) -> np.array:
     b_hat = np.linalg.inv(Rk) @ b_tilde
     return b_hat



def residual_error(a_tilde: np.array, b_tilde: np.array, R: np.array) -> np.array:
     error = np.sum((R @ a_tilde - b_tilde)**2)
     return error
def recalage(
        bead_1_position_phantom: np.array,
        bead_2_position_phantom: np.array, 
        bead_3_position_phantom: np.array, 
        bead_1_position_world: np.array, 
        bead_2_position_world: np.array, 
        bead_3_position_world: np.array,
        tolerance: float, 
        max_itteration: int
    ) -> np.array:
    """
    Fait le recalage entre le repère de l'organe et le reprère monde (robot)

    Parameters: 
        bead_1_position_phantom: position de la bille 1 dans le repère du fantôme (organe)
        bead_2_position_phantom: position de la bille 2 dans le repère du fantôme (organe)
        bead_3_position_phantom: position de la bille 3 dans le repère du fantôme (organe)
        bead_1_position_world: position de la bille 1 dans le repère du monde (robot)
        bead_2_position_world: position de la bille 2 dans le repère du monde (robot)
        bead_3_position_world: position de la bille 3 dans le repère du monde (robot)
    Returns:
        T: la matrice de transformation
    """
    phantom_beads = [bead_1_position_phantom, bead_2_position_phantom, bead_3_position_phantom]
    world_beads = [bead_1_position_world, bead_2_position_world, bead_3_position_world]

    a_tilde, b_tilde = compute_a_tilde(phantom_beads), compute_a_tilde(world_beads)

    # Fonction a optimiser
    def func(angles):
        ax,ay,az = angles
        Rx = Rotation.from_euler('x', ax, degrees=True).as_matrix()
        Ry = Rotation.from_euler('y', ay, degrees=True).as_matrix()
        Rz = Rotation.from_euler('z', az, degrees=True).as_matrix()
        delta_R = Rz @ Ry @ Rx
        residu = np.sum((delta_R @ a_tilde - b_hat)**2, axis = 0)
        return residu.flatten()
    
    # -- 0. Initialisation --
    R_0 = np.eye(3)
    R = R_0
    number_itterations = 0

    while (residual_error(a_tilde, b_tilde, R) >= tolerance) or (number_itterations <= max_itteration):
        # --- 1. Compute b_hat ---
        b_hat = compute_b_hat(b_tilde, R)

        # --- 2. Find correction matrix ---
        angles = fsolve(func, (0,0,0))
        ax,ay,az = angles
        Rx = Rotation.from_euler('x', ax, degrees=True).as_matrix()
        Ry = Rotation.from_euler('y', ay, degrees=True).as_matrix()
        Rz = Rotation.from_euler('z', az, degrees=True).as_matrix()
        delta_R = Rz @ Ry @ Rx
        
        # --- 3. Apply correction matrix --- 
        R = R @ delta_R

        number_itterations += 1
    print(R)
    return R
recalage(bead_1_position_phantom, bead_2_position_phantom, bead_3_position_phantom,bead_1_position_world, bead_2_position_world, bead_3_position_world, tolerance = 1e-4, max_itteration = 10 )