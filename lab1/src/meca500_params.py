import numpy as np

# Link lengths (mm)
L1 = 90
L2 = 45
L3 = 135
L4x = 38
L4y = 60
L5 = 60
L6 = 70

# Joint limits (degrees)
JOINT_LIMITS = [
    (-175, 175),
    (-70, 90),
    (-135, 70),
    (-170, 170),
    (-115, 115),
    (-180, 180),
]

# Joint position vectors
P_1org_0 = np.array([0, 0, L1])     # Joint 1 position in base frame
P_2org_1 = np.array([0, 0, L2])     # Joint 2 position in joint 1 frame
P_3org_2 = np.array([0, -L3, 0])    # Joint 3 position in joint 2 frame
P_4org_3 = np.array([L4x, L4y, 0])  # Joint 4 position in joint 3 frame
P_5org_4 = np.array([0, 0, L5])     # Joint 5 position in joint 4 frame
P_6org_5 = np.array([0, L6, 0])     # Joint 6 position in joint 5 frame