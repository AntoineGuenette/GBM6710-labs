import numpy as np
from scipy.spatial.transform import Rotation as R
from transforms import Rx, Ry, Rz, T
from meca500_params import *


def dk_T(q_deg):
    """
    Fonction qui retourne la matrice de transformation T_6_0 (4X4)

    q_deg : angles des 6 joints en degré
    """
    q1, q2, q3, q4, q5, q6 = q_deg

    R_1_0 = Rz(q1)
    R_2_1 = Rx(-90) @ Rz(q2)
    R_3_2 = Rz(-90) @ Rz(q3)
    R_4_3 = Rx(-90) @ Rz(q4)
    R_5_4 = Rx(90)  @ Rz(q5)
    R_6_5 = Rx(-90) @ Ry(180) @ Rz(q6)

    T_6_0 = (
        T(R_1_0, P_1org_0) @
        T(R_2_1, P_2org_1) @
        T(R_3_2, P_3org_2) @
        T(R_4_3, P_4org_3) @
        T(R_5_4, P_5org_4) @
        T(R_6_5, P_6org_5)
    )

    return T_6_0

# Gérer limites articulaires parce que la cinématique inverse peut proposer des angles hors limites
def gerer_limites(q):
    """Force chaque angle q à rester dans les limites JOINT_LIMITS"""
    q = q.copy()
    for i, (low, high) in enumerate(JOINT_LIMITS):
        q[i] = np.clip(q[i], low, high)
    return q

# delta O de l'erreur
def erreur_orientation(R_target, R_current):
    """Erreur orientation sous forme de vecteur rotation"""
    R_err = R_target @ R_current.T
    return R.from_matrix(R_err).as_rotvec()

# Selon mes recherches + conseil IA, la meilleure technique pour faire la cinématique inverse
# est Damped Least Squares qui permet d'éviter que ça explose quand Jacobien est proche d'une singularité
# Ça permet "d'inversé" la jacobienne avec un facteur d'amortissement. 


# Jacobien numérique 
# Jacobien J : e(q+dq) = e(q) + Jdq avec e vecteur d'erreur de position et d'orientation

def jacobien_num(q, p_target, R_target, eps_deg=1e-2):
    """
    Calcule le Jacobien numérique J (6x6) et l'erreur e (6,) pour une pose cible.

    e = [p_target - p_current,  rotvec(R_target * R_current^T)]
    - position en mm
    - orientation en radians
    """
    T0 = dk_T(q)
    p0 = T0[:3, 3]
    R0 = T0[:3, :3]

    e0 = np.hstack([
        p_target - p0,
        erreur_orientation(R_target, R0)
    ])

    J = np.zeros((6, 6))
    step = np.deg2rad(eps_deg)

    for i in range(6):
        q1 = q.copy()
        q1[i] += eps_deg

        T1 = dk_T(q1)
        p1 = T1[:3, 3]
        R1 = T1[:3, :3]

        e1 = np.hstack([
            p_target - p1,
            erreur_orientation(R_target, R1)
        ])

        J[:, i] = (e1 - e0) / step

    return J, e0


def cin_inv_solution(x, y, z,alpha, beta, gamma, q0, max_iters=60, tol_pos=1, tol_rot=1, lam=1e-2, gain=0.6, verbose = False):
    p_target = np.array([x, y, z], dtype=float)

    # Convention Euler ZYX 
    R_target = R.from_euler('ZYX',[gamma, beta, alpha], degrees=True).as_matrix()
    q = q0.astype(float)

    for it in range(max_iters):
        J, e = jacobien_num(q, p_target, R_target)
        pos_err = np.linalg.norm(e[:3])
        rot_err = np.linalg.norm(e[3:])

        if verbose and (it % 10 == 0):
            print(f"  it={it:3d} | pos_err={pos_err:8.3f} | rot_err={rot_err:8.4e}")

        if pos_err < tol_pos and rot_err < tol_rot:
            return gerer_limites(q), True

        # DLS
        A = J @ J.T + (lam ** 2) * np.eye(6)
        dq = -J.T @ np.linalg.solve(A, e)

        # Remettre les angles de rad --> deg
        q += gain * np.rad2deg(dq)
        q = gerer_limites(q)

    return q, False

# On lance plusieurs seeds pour "balayer" plusieurs bassins d’attraction
# Le robot 6R peut avoir plusieurs solutions (coude haut/bas, wrist flip, etc.)
# Un solveur numérique converge souvent vers "la solution la plus proche" de la seed

def cin__inv_toutes_solutions(x, y, z, alpha, beta, gamma):
    """
    Retourne une liste de solutions q (deg) trouvées en lançant la cinématique inverse depuis plusieurs seeds.

    Remarque :
    - Il est possible d'avoir 0 solution si la pose est inatteignable.
    - Il est possible d'avoir plusieurs solutions (symétries / flips).
    """
    seeds = [
        np.zeros(6),
        np.array([0, -45, 45, 0, 0, 0]),
        np.array([0, 45, -45, 0, 0, 0]),
        np.array([90, 0, 0, 0, 0, 0]),
        np.array([-90, 0, 0, 0, 0, 0]),
        np.array([0, 0, 0, 90, 0, 0]),
        np.array([0, 0, 0, -90, 0, 0]),
    ]

    solutions = []

    for k, q0 in enumerate(seeds, 1):
        print(f"\nSeed {k}/{len(seeds)} : {q0}")
        q_sol, ok = cin_inv_solution(x,y,z, alpha, beta, gamma, q0, max_iters=80, verbose=True)
        if not ok:
            continue

        # éliminer doublons
        is_new = True
        for q_prev in solutions:
            if np.linalg.norm(q_prev - q_sol) < 1.0:
                is_new = False
                break

        if is_new:
            solutions.append(q_sol)

    return solutions


def print_solutions(solutions):
    """
    Print plus propre pour les solutions
    """
    if not solutions:
        print("Aucune solution valide trouvée.")
        return

    print(f"{len(solutions)} solution(s) trouvée(s) :\n")

    for i, q in enumerate(solutions, 1):
        print(f"Solution {i}")
        for j, angle in enumerate(q, 1):
            print(f"  q{j}: {angle:8.3f} deg")
        print()
def euler_zyx_from_q(q_deg):
    T0 = dk_T(q_deg)
    R0 = T0[:3, :3]
    gamma, beta, alpha = R.from_matrix(R0).as_euler('ZYX', degrees=True)
    return alpha, beta, gamma

# Test
x, y, z = 190, 0, 308
alpha, beta, gamma = 0, 90, -180

solutions = cin__inv_toutes_solutions(x, y, z, alpha, beta, gamma)
print_solutions(solutions)


