import numpy as np

# All dimensions in mm

starting_distance = 50.0

bead_1_position_phantom = np.array([18.99, -133.90, -227.54], dtype=float)
bead_2_position_phantom = np.array([-13.68, -132.96, -226.15], dtype=float)
bead_3_position_phantom = np.array([-14.61, -131.08, -124.10], dtype=float)

pink_tumor_position_phantom = np.array([0, 0, 0], dtype=float)
orange_tumor_position_phantom = np.array([0, 0, 0], dtype=float)

pink_insertion_point_phantom = np.array([0, 0, 0], dtype=float)
orange_insertion_point_phantom = np.array([0, 0, 0], dtype=float)

pink_starting_point_phantom = pink_insertion_point_phantom - np.array([0, 50, 0], dtype=float)
orange_starting_point_phantom = orange_insertion_point_phantom - np.array([0, 50, 0], dtype=float)
