import numpy as np

# All dimensions are in mm
# The left and right sides are defined by looking at the open side of the phantom

# Box top corners
box_back_right_phantom = np.array([64.98, -179.54, -292.13], dtype=float)
box_back_left_phantom = np.array([-58.65, -179.54, -291.67], dtype=float)
box_front_right_phantom = np.array([67.31, -179.54, -400.03], dtype=float)
box_front_left_phantom = np.array([-59.12, -179.54, -399.11], dtype=float)

# Obstacle front corners
obs_top_right_phantom = np.array([50.98, -156.05, -377.81], dtype=float)
obs_top_left_phantom = np.array([-44.19, -156.05, -377.81], dtype=float)
obs_bottom_right_phantom = np.array([41.65, -146.19, -377.81], dtype=float)
obs_bottom_left_phantom = np.array([-36.26, -146.19, -377.81], dtype=float)

# Target front corners
trg_top_right_phantom = np.array([10.86, -118.00, -378.27], dtype=float)
trg_top_left_phantom = np.array([-4.07, -118.00, -378.27], dtype=float)
trg_bottom_right_phantom = np.array([10.86, -106.26, -377.81], dtype=float)
trg_bottom_left_phantom = np.array([-4.54, -106.26, -377.81], dtype=float)

# Contact point
contact_point_phantom = np.array([3.39, -118.00, -372.00], dtype=float)

# Deflection point (maximum bending of target)
deflection_point_phantom = np.array([3.39, -113.00, -372.00], dtype=float)

# Middle point (between target and obstacle, inside the phantom)
middle_point_phantom = np.array([3.39, -132.00, -372.00], dtype=float)

# Starting point (between target and obstacle, outside the phantom)
starting_point_phantom = np.array([3.39, -132.00, -425.00], dtype=float)
