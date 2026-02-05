import numpy as np
from scipy.spatial.transform import Rotation as R

from meca500_params import *

def enforce_joint_limits(joint_angles):
    """Enforce joint limits defined in JOINT_LIMITS on the given joint angles."""
    q = joint_angles.copy()
    for i, (low, high) in enumerate(JOINT_LIMITS):
        q[i] = np.clip(q[i], low, high)
    return q
