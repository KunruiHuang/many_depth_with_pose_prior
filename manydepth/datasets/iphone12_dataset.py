# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class IPHONE12Dataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(IPHONE12Dataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[0.798, 0, 0.5, 0],
                           [0, 1.596, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (768, 384)
        self.side_map = {"2": 2, "3": 3, "l": 3, "r": 3}


    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        print("index:",index)
        line = self.filenames[index - 1].split()
        folder = line[0]
        frame_index = int(line[1])
        side = line[2]
        pose_dict = {
        'tx': 0.0,
        'ty': 0.0,
        'tz': 0.0,
        'qw': 1.0,
        'qx': 0.0,
        'qy': 0.0,
        'qz': 0.0
        } 
        if len(line) == 10:
            pose_dict.update({key: float(value) for key, value in zip(['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz'], line[3:])})
        #end of test

        return folder, frame_index, side, pose_dict

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "images03/", f_str)
        return image_path
    def check_depth(self):
        return False






