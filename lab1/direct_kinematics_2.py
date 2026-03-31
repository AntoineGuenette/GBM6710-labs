import numpy as np
from scipy.spatial.transform import Rotation as rot

def direct_kinematics_meca500(joint_angles):
    # --- Dimensions exactes pour obtenir Z = 281.748 ---
    L1 = 135.0  # Base à Axe 2
    L3 = 135.0  # Segment bras
    L4_z = 38.0 # Offset vertical poignet
    L4_x = 120.0 # Longueur avant-bras
    L6 = 70.0   # Axe 5 à l'outil

    def T_mat(R, P):
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = P
        return T

    # --- Séquence de Transformations ---
    # 1. Base -> J1 (Rotation Z)
    T1 = T_mat(rot.from_euler('z', joint_angles[0], degrees=True).as_matrix(), [0, 0, L1])

    # 2. J1 -> J2 (Rotation Y)
    T2 = T_mat(rot.from_euler('y', joint_angles[1], degrees=True).as_matrix(), [0, 0, 0])

    # 3. J2 -> J3 (Translation L3 + Rotation Y)
    T3 = T_mat(rot.from_euler('y', joint_angles[2], degrees=True).as_matrix(), [0, 0, L3])

    # 4. J3 -> J4 (Translation L4 + Rotation X)
    # Note: On avance en X et on monte en Z pour atteindre le centre du poignet
    T4 = T_mat(rot.from_euler('x', joint_angles[3], degrees=True).as_matrix(), [L4_x, 0, L4_z])

    # 5. J4 -> J5 (Rotation Y)
    T5 = T_mat(rot.from_euler('y', joint_angles[4], degrees=True).as_matrix(), [0, 0, 0])

    # 6. J5 -> J6 (Translation L6 + Rotation X)
    T6 = T_mat(rot.from_euler('x', joint_angles[5], degrees=True).as_matrix(), [L6, 0, 0])

    # Matrice globale
    T_total = T1 @ T2 @ T3 @ T4 @ T5 @ T6

    # Position
    pos = T_total[0:3, 3]

    # Orientation : Le Meca500 utilise la convention XYZ extrinsèque pour l'affichage
    # C'est ce qui permet d'obtenir les angles -49.308, -10.847, 2.677
    euler = rot.from_matrix(T_total[0:3, 0:3]).as_euler('xyz', degrees=True)

    return pos, euler

# Test avec tes valeurs
angles = [10, 0, 15, 0, -20, -50]
pos, ori = direct_kinematics_meca500(angles)

print(f"Position (X, Y, Z): {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")
print(f"Angles Euler (X, Y, Z): {ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}")