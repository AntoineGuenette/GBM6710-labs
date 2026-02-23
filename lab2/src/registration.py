import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation

from phantom_params import *

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

def compute_registration_transform(
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
    Compute the registration transform (rotation and translation) from phantom to world coordinates
    using an iterative optimization approach.

    Parameters:
        bead_1_position_phantom (np.array): Position of bead 1 in phantom coordinates
        bead_2_position_phantom (np.array): Position of bead 2 in phantom coordinates
        bead_3_position_phantom (np.array): Position of bead 3 in phantom coordinates
        bead_1_position_world (np.array): Position of bead 1 in world coordinates
        bead_2_position_world (np.array): Position of bead 2 in world coordinates
        bead_3_position_world (np.array): Position of bead 3 in world coordinates
        tolerance (float): Tolerance for convergence of the optimization
        max_itteration (int): Maximum number of iterations for the optimization

    Returns:
        R (np.array): Rotation matrix representing the registration transform
        t (np.array): Translation vector representing the registration transform
    """
    phantom_beads = [bead_1_position_phantom, bead_2_position_phantom, bead_3_position_phantom]
    world_beads = [bead_1_position_world, bead_2_position_world, bead_3_position_world]

    a_tilde, b_tilde = compute_a_tilde(phantom_beads), compute_a_tilde(world_beads)

    # Function to minimize
    def func(euler_angles):
        ax, ay, az = euler_angles
        Rx = Rotation.from_euler('x', ax, degrees=True).as_matrix()
        Ry = Rotation.from_euler('y', ay, degrees=True).as_matrix()
        Rz = Rotation.from_euler('z', az, degrees=True).as_matrix()
        delta_R = Rz @ Ry @ Rx
        residual = np.sum((delta_R @ a_tilde - b_hat)**2, axis = 0)
        return residual.flatten()
    
    # Initialisation
    R_0 = np.eye(3)
    R = R_0
    number_itterations = 0

    while (residual_error(a_tilde, b_tilde, R) >= tolerance) or (number_itterations <= max_itteration):
        # Compute b_hat
        b_hat = compute_b_hat(b_tilde, R)

        # Find correction matrix
        angles = fsolve(func, (0,0,0))
        ax,ay,az = angles
        Rx = Rotation.from_euler('x', ax, degrees=True).as_matrix()
        Ry = Rotation.from_euler('y', ay, degrees=True).as_matrix()
        Rz = Rotation.from_euler('z', az, degrees=True).as_matrix()
        delta_R = Rz @ Ry @ Rx
        
        # Apply correction matrix
        R = R @ delta_R

        number_itterations += 1

    return R

if __name__ == "__main__":
     
    bead_1_position_world = input("Enter the position of bead 1 in world coordinates (mm) as x,y,z: ")
    bead_2_position_world = input("Enter the position of bead 2 in world coordinates (mm) as x,y,z: ")
    bead_3_position_world = input("Enter the position of bead 3 in world coordinates (mm) as x,y,z: ")

    R = compute_registration_transform(
          bead_1_position_phantom,
          bead_2_position_phantom,
          bead_3_position_phantom,
          bead_1_position_world,
          bead_2_position_world,
          bead_3_position_world,
          tolerance = 1e-4,
          max_itteration = 10 )
     
    print(R)
