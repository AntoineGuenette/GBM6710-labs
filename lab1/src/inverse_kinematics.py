import numpy as np
from scipy.spatial.transform import Rotation as R

from transforms import numerical_jacobian
from utils import enforce_joint_limits

def solve_inverse_kinematics_seed(
    x: float,
    y: float,
    z: float,
    alpha: float,
    beta: float,
    gamma: float,
    initial_guess: np.array,
    max_iters: int = 100,
    position_tolerance: float = 1e-3,
    orientation_tolerance: float = 1.0,
    damping: float = 1e-2,
    gain: float = 0.6,
    verbose: bool = False
) -> tuple:
    """
    Solve the inverse kinematics for a single seed using damped least squares (DLS)
    numerical method.

    Parameters:
        x (float): Target x position in millimeters.
        y (float): Target y position in millimeters.
        z (float): Target z position in millimeters.
        alpha (float): Target alpha angle in degrees (Euler ZYX convention).
        beta (float): Target beta angle in degrees (Euler ZYX convention).
        gamma (float): Target gamma angle in degrees (Euler ZYX convention).
        initial_guess (np.array): Initial joint configuration (degrees).
        max_iters (int): Maximum number of iterations.
        position_tolerance (float): Position error tolerance in millimeters.
        orientation_tolerance (float): Orientation error tolerance (norm of rotation vector).
        damping (float): Damping factor for DLS.
        gain (float): Integration gain for joint updates.
        verbose (bool): Whether to print iteration info to console.

    Returns:
        joint_angles (np.array): Solution joint angles (degrees).
        success (bool): True if solution converged, False otherwise.
        position_error (float): Final position error (mm).
        score (float): Final score (sum of position and orientation errors).
    """
    # Target pose
    target_position = np.array([x, y, z], dtype=float)
    target_rotation = R.from_euler('ZYX', [gamma, beta, alpha], degrees=True).as_matrix()
    joint_angles = initial_guess.astype(float)

    print("Solving IK with initial guess:", joint_angles)
    for iteration in range(max_iters):
        J, error = numerical_jacobian(joint_angles, target_position, target_rotation)
        position_error = np.linalg.norm(error[:3])
        orientation_error = np.linalg.norm(error[3:])
        score = position_error + orientation_error

        if verbose and (iteration % 10 == 0):
            print(f"  it={iteration:3d} | pos_err={position_error:8.3f} | rot_err={orientation_error:8.4e} | score={score:8.3f}")

        if position_error < position_tolerance and orientation_error < orientation_tolerance:
            return enforce_joint_limits(joint_angles), True, score

        # Damped Least Squares step
        A = J @ J.T + (damping ** 2) * np.eye(6)
        delta_q = -J.T @ np.linalg.solve(A, error)

        # Update joint angles (rad -> deg)
        joint_angles += gain * np.rad2deg(delta_q)
        joint_angles = enforce_joint_limits(joint_angles)

    return joint_angles, False, score

def inverse_kinematics(
    x: float,
    y: float,
    z: float,
    alpha: float,
    beta: float,
    gamma: float,
    initial_guesses: list = None,
    max_iters: int = 100,
    position_tolerance: float = 1e-3,
    orientation_tolerance: float = 1.0,
    damping: float = 1e-2,
    gain: float = 0.6,
    verbose: bool = False
) -> list:
    """
    Solve the inverse kinematics of the Meca500 robotic arm using a damped least squares (DLS)
    numerical method. The solver can be initialized from one or multiple initial joint
    configurations to recover multiple valid solutions.

    Parameters:
        x (float): Target x position in millimeters.
        y (float): Target y position in millimeters.
        z (float): Target z position in millimeters.
        alpha (float): Target alpha angle in degrees (Euler ZYX convention).
        beta (float): Target beta angle in degrees (Euler ZYX convention).
        gamma (float): Target gamma angle in degrees (Euler ZYX convention).
        initial_guesses (list): List of initial joint configurations (degrees). If None, a default
            set of seeds is used.
        max_iters (int): Maximum number of iterations per initial guess.
        position_tolerance (float): Position error tolerance in millimeters.
        orientation_tolerance (float): Orientation error tolerance (norm of the rotation vector).
        damping (float): Damping factor for the DLS method.
        gain (float): Integration gain for joint updates.
        verbose (bool): Whether to print iteration info to console.

    Returns:
        best_solution np.array: The best solution (joint angles in degrees) that satisfies the inverse kinematics constraints.
    """
    # Default initial guesses (seeds)
    if initial_guesses is None:
        initial_guesses = [
            np.zeros(6),
            np.array([0, -45, 45, 0, 0, 0]),
            np.array([0, 45, -45, 0, 0, 0]),
            np.array([90, 0, 0, 0, 0, 0]),
            np.array([-90, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 90, 0, 0]),
            np.array([0, 0, 0, -90, 0, 0]),
        ]

    # Initialize best solution
    best_solution = None
    best_score = float('inf')

    for seed in initial_guesses:
        joint_angles, success, score = solve_inverse_kinematics_seed(
            x, y, z, alpha, beta, gamma,
            initial_guess=seed,
            max_iters=max_iters,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            damping=damping,
            gain=gain,
            verbose=verbose
        )

        if success:
            # Keep only the best solution (not already in the list)
            if score < best_score:
                best_solution = joint_angles
                best_score = score
    return best_solution


if __name__ == "__main__":

    print("\n --- Meca500 Inverse Kinematics --- \n")

    x = float(input("Enter target x position (mm): "))
    y = float(input("Enter target y position (mm): "))
    z = float(input("Enter target z position (mm): "))
    alpha = float(input("Enter target alpha angle (degrees): "))
    beta = float(input("Enter target beta angle (degrees): "))
    gamma = float(input("Enter target gamma angle (degrees): "))

    print("\n")

    # Solve IK
    solution = inverse_kinematics(
        x, y, z, alpha, beta, gamma,
        verbose=True
    )

    # Display solutions
    if solution is None:
        print("\nNo solution found.")
    else:
        print(f"\nBest solution found:\n")
        for j, angle in enumerate(solution, 1):
            print(f"    Joint {j} angle: {angle:8.3f} deg")
        print()