# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import numpy as np
from scipy.spatial.transform import Rotation

def readlines(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            values = [val for val in line.split()]
            assert len(values) == 16, f"Expected length 16, but got {len(values)}"
            lines.append(values)
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def compute_relative_pose(pose1, pose2):
    # Extract translation and quaternion components
    t1, t2 = np.array([pose1[:3]]), np.array([pose2[:3]])
    q1, q2 = np.array([pose1[4], pose1[5], pose1[6], pose1[3]]), np.array([pose2[4], pose2[5], pose2[6], pose2[3]])
    # Compute relative translation
    relative_translation = t2 - t1
    # Compute relative rotation
    r1 = Rotation.from_quat(q1)
    r2 = Rotation.from_quat(q2)
    relative_rotation = r2 * r1.inv()
    # Get relative rotation as a 4x4 matrix
    relative_matrix = np.eye(4)
    relative_matrix[:3, :3] = relative_rotation.as_matrix()
    relative_matrix[:3, 3] = relative_translation.flatten()
    return relative_matrix