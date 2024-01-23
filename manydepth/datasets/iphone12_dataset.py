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
from PIL import Image
import matplotlib.pyplot as plt

from .mono_dataset import MonoDataset
from scipy.spatial.transform import Rotation as R

def load_depth_npy(path, data_format='HW'):
    z = np.load(path).astype(np.float32)
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z

def load_depth(depth_path):
    return load_depth_npy(depth_path, data_format='CHW')

def load_image(path, normalize=True, data_format='HWC'):
    '''
    Loads an RGB image

    Arg(s):
        path : str
            path to RGB image
        normalize : bool
            if set, then normalize image between [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : H x W x C or C x H x W image
    '''

    # Load image
    image = Image.open(path).convert('RGB')

    # Convert to numpy
    image = np.asarray(image, np.uint8)

    if data_format == 'CHW':
        image = np.transpose(image, (2, 0, 1))

    return image


def load_image_triplet(path, normalize=True):
    '''
    Load images from image triplet

    Arg(s):
        path : str
            path to image triplet
        normalize : bool
            if set, normalize by to [0, 1] range
    Return:
        numpy[float32] : image at time t (C x H x W)
        numpy[float32] : image at time t-1 (C x H x W)
        numpy[float32] : image at time t+1 (C x H x W)
    '''

    # Load image triplet and split into images at t-1, t, t+1
    images = load_image(
        path,
        normalize=normalize,
        data_format='HWC')

    # Split along width
    image1, image0, image2 = np.split(images, indices_or_sections=3, axis=1)
    image1 = Image.fromarray(image1)
    image0 = Image.fromarray(image0)
    image2 = Image.fromarray(image2)
    return image1, image0, image2


def qt2T(q_wxyz, t):
   q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
   R_obj = R.from_quat(q_xyzw)
   R_ = R_obj.as_matrix()

   T = np.eye(4)
   T[:3, :3] = R_
   T[:3, 3] = t 
   
   return T

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
        row = self.filenames[index]
        image_triplet_path = os.path.join(self.root_dir,str(row[0]),'trained_images',str(row[1])+ '.png')
        image1, image0, image2 = load_image_triplet(
            image_triplet_path,
            normalize=False)
                
        sparse_depth_npy_path = os.path.join(self.root_dir,str(row[0]),'sparse_depth',str(row[1])+ '.npy') 
        sparse_depth0 = load_depth(sparse_depth_npy_path)
        q_cur2last = np.array([float(row[5]),float(row[6]),float(row[7]),float(row[8])])
        t_cur2last = np.array([float(row[2]),float(row[3]),float(row[4])])
        q_cur2future = np.array([float(row[12]),float(row[13]),float(row[14]),float(row[15])])
        t_cur2future = np.array([float(row[9]),float(row[10]),float(row[11])])
        T_last_cur = qt2T(q_cur2last, t_cur2last)
        T_future_cur = qt2T(q_cur2future, t_cur2future)



        T_last_cur,T_future_cur,sparse_depth0 = [
            T.astype(np.float32)
            for T in [T_last_cur,T_future_cur,sparse_depth0]
        ]
        # image0.show()
        # image1.show()
        # image2.show()
        # print("numpy:",np.max(sparse_depth0))
        return image0, image1, image2,T_last_cur,T_future_cur,sparse_depth0

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_02/", f_str)
        return image_path
    def check_depth(self):
        return False






