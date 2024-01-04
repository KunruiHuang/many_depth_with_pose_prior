import unittest
import numpy as np
from scipy.spatial.transform import Rotation

def compute_relative_pose(pose1, pose2):
    # Extract translation and quaternion components
    t1, t2 = np.array([pose1[:3]]), np.array([pose2[:3]])
    q1, q2 = np.array([pose1[4], pose1[5], pose1[6], pose1[3]]), np.array([pose2[4], pose2[5], pose2[6], pose2[3]])

    # Compute transformation matrices
    T1 = np.eye(4)
    T1[:3, :3] = Rotation.from_quat(q1).as_matrix()
    T1[:3, 3] = t1.flatten()

    T2 = np.eye(4)
    T2[:3, :3] = Rotation.from_quat(q2).as_matrix()
    T2[:3, 3] = t2.flatten()

    # Compute relative transformation
    relative_matrix = np.linalg.inv(T2) @ T1

    return relative_matrix

class TestComputeRelativePose(unittest.TestCase):
    def test_compute_relative_pose_identity(self):
        # Test when pose1 and pose2 are the same (identity transformation)
        pose = [1.0, 2.0, 3.0, 0.7071, 0.0, 0.0, 0.7071]
        relative_matrix = compute_relative_pose(pose, pose)
        expected_matrix = np.eye(4)
        np.testing.assert_allclose(relative_matrix, expected_matrix, atol=1e-6)

    def test_compute_relative_pose_translation(self):
        # Test when there is only translation, no rotation
        pose1 = [1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        pose2 = [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0]
        relative_matrix = compute_relative_pose(pose1, pose2)
        expected_matrix = np.eye(4)
        expected_matrix[:3, 3] = [-3.0, -2.0, -3.0]
        np.testing.assert_allclose(relative_matrix, expected_matrix, atol=1e-6)

    def test_compute_relative_pose_rotation(self):
        # Test when there is only rotation, no translation
        pose1 = [0.0, 0.0, 0.0, 0.7071, 0.0, 0.0, 0.7071]
        pose2 = [0.0, 0.0, 0.0, 0.7071, 0.0, 0.0, -0.7071]
        relative_matrix = compute_relative_pose(pose1, pose2)
        expected_matrix = np.eye(4)
        expected_matrix[:3, :3] = Rotation.from_quat([0, 0, 1, 0]).as_matrix()
        np.testing.assert_allclose(relative_matrix, expected_matrix, atol=1e-6)

if __name__ == '__main__':
    unittest.main()