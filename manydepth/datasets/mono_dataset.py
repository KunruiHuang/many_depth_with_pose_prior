# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import random
import sys
sys.path.append("..") 
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms
cv2.setNumThreads(0)
import numpy as np
from scipy.spatial.transform import Rotation
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
    relative_matrix_torch = torch.tensor(relative_matrix, dtype=torch.float32)
    return relative_matrix_torch
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 ):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales

        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                # check it isn't a blank frame - keep _aug as zeros so we can check for it
                if inputs[(n, im, i)].sum() == 0:
                    inputs[(n + "_aug", im, i)] = inputs[(n, im, i)]
                else:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                    # inputs[(n + "_aug", im, i)] = inputs[(n, im, i)]

    def __len__(self):
        return len(self.filenames)

    def load_intrinsics(self, folder, frame_index):
        return self.K.copy()

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps

        <frame_id> is:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = False
        do_flip = self.is_train and random.random() > 0.5

        folder, frame_index, side,pose_dict = self.index_to_folder_and_frame_idx(index)
        poses = {}
        if type(self).__name__ in ["CityscapesPreprocessedDataset", "CityscapesEvalDataset"]:
            inputs.update(self.get_colors(folder, frame_index, side, do_flip))
        else:
            for i in self.frame_idxs:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(
                        folder, frame_index, other_side, do_flip)
                else:
                    try:
                        inputs[("color", i, -1)] = self.get_color(
                            folder, frame_index + i, side, do_flip)
                        #test
                        if i != 0 and pose_dict is not None:
                            folder, frame_index, side,pose_dict_i = self.index_to_folder_and_frame_idx(index + i)
                            pose0 = [pose_dict['tx'], pose_dict['ty'], pose_dict['tz'], pose_dict['qw'], pose_dict['qx'], pose_dict['qy'], pose_dict['qz']]
                            posei = [pose_dict_i['tx'], pose_dict_i['ty'], pose_dict_i['tz'], pose_dict_i['qw'], pose_dict_i['qx'], pose_dict_i['qy'], pose_dict_i['qz']]
                            relative_matrix = compute_relative_pose(pose0, posei)
                            inputs[("cam_T_cam", 0, i)] = relative_matrix
                        #end of test
                    except FileNotFoundError as e:
                        if i != 0:
                            # fill with dummy values
                            inputs[("color", i, -1)] = \
                                Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
                            poses[i] = None
                        else:
                            raise FileNotFoundError(f'Cannot find frame - make sure your '
                                                    f'--data_path is set correctly, or try adding'
                                                    f' the --png flag. {e}')

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.load_intrinsics(folder, frame_index)

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth and False:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
