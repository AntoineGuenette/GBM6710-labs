import numpy as np
from scipy.spatial.transform import Rotation as R

from transforms import numerical_jacobian
from utils import enforce_joint_limits

def inverse_kinematics(
    x: float,
    y: float,
    z: float,
    alpha: float,
    beta: float,
    gamma: float,
    initial_guesses=None,
    max_iters: int = 100,
    position_tolerance: float = 1e-2,
    orientation_tolerance: float = 1e-2,
    damping: float = 0.1,
    gain: float = 0.65,
    verbose: bool = False
) -> np.array:
    """
    Solve the inverse kinematics of the Meca500 robotic arm using a damped least squares (DLS)
    numerical method. The solver can be initialized from one or multiple initial joint
    configurations to recover multiple valid solutions.

    Parameters:
        x (float): Target x position in millimeters.
        y (float): Target y position in millimeters.
        z (float): Target z position in millimeters.
        alpha (float): Target alpha angle in degrees.
        beta (float): Target beta angle in degrees.
        gamma (float): Target gamma angle in degrees.
        initial_guesses (list): List of initial joint configurations in degrees. If None, a default
            set of seeds is used.
        max_iters (int): Maximum number of iterations per initial guess.
        position_tolerance (float): Position error tolerance in millimeters.
        orientation_tolerance (float): Orientation error tolerance in degrees (Euclidean norm of
        Euler angle error).
        damping (float): Damping factor for the DLS method.
        gain (float): Integration gain for joint updates.
        verbose (bool): Whether to print iteration info to the console.

    Returns:
        best_solution (Optional(np.array)): Best joint configuration (degrees) satisfying the IK constraints,
        or None if no solution is found.
    """
    # Target pose
    target_position = np.array([x, y, z], dtype=float)
    target_rotation = R.from_euler(
        'zyx', [gamma, beta, alpha], degrees=True
    ).as_matrix()

    # Set of initial guesses (seeds)
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

    # Allow single seed as np.array
    if isinstance(initial_guesses, np.ndarray):
        initial_guesses = [initial_guesses]

    # Initialization
    best_solution = None
    best_score = float("inf")

    for seed in initial_guesses:
        joint_angles = seed.astype(float)

        if verbose:
            print("\nSolving IK with initial guess:", joint_angles)

        for iteration in range(max_iters):
            # Compute the jacobian and the errors
            J, error = numerical_jacobian(
                joint_angles, target_position, target_rotation
            )

            # Deduce the score via the position and orientation errors
            pos_err = np.linalg.norm(error[:3])
            rot_err = np.linalg.norm(error[3:])
            score = pos_err + rot_err

            # Show optimization progress
            if verbose and iteration % 10 == 0:
                print(
                    f"  it={iteration:3d} | "
                    f"pos_err={pos_err:8.3f} | "
                    f"rot_err={rot_err:8.4e} | "
                    f"score={score:8.3f}"
                )

            # Save the solution only if it is better than the current best solution
            if pos_err < position_tolerance and rot_err < orientation_tolerance:
                joint_angles = enforce_joint_limits(joint_angles)
                if score < best_score:
                    best_solution = joint_angles.copy()
                    best_score = score
                break

            # Damped Least Squares
            A = J @ J.T + (damping ** 2) * np.eye(6)
            delta_q = -J.T @ np.linalg.solve(A, error)

            # Update joints
            joint_angles += gain * np.rad2deg(delta_q)
            joint_angles = enforce_joint_limits(joint_angles)

    return best_solution


if __name__ == "__main__":

    print("\n --- Meca500 Inverse Kinematics --- \n")

    x = float(input("Enter target x position (mm): "))
    y = float(input("Enter target y position (mm): "))
    z = float(input("Enter target z position (mm): "))
    alpha = float(input("Enter target alpha angle (degrees): "))
    beta = float(input("Enter target beta angle (degrees): "))
    gamma = float(input("Enter target gamma angle (degrees): "))

    # Solve IK
    solution = inverse_kinematics(
        x, y, z, alpha, beta, gamma,
        verbose=True
    )

    # Display solutions
    if solution is None:
        print("\nNo solution found.")
    else:
        print(f"\nBest solution found:")
        for j, angle in enumerate(solution, 1):
            print(f"\tJoint {j} angle: {angle:8.3f} deg")
        print()