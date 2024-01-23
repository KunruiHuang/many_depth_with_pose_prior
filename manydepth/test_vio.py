# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import json
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from manydepth import datasets, networks
import torch
from torchvision import transforms
from manydepth.utils import readlines, sec_to_hm_str
from manydepth.datasets.iphone12_dataset import load_image_triplet,qt2T,load_image,load_depth
from manydepth import networks
from manydepth.layers import transformation_from_parameters,disp_to_depth

import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader


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
    relative_matrix_torch = torch.tensor(relative_matrix, dtype=torch.float32)
    return relative_matrix_torch
def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for ManyDepth models.')

    parser.add_argument('--target_image_path', type=str,
                        help='path to a test image to predict for', required=True)
    parser.add_argument('--source_image_path', type=str,
                        help='path to a previous image in the video sequence', required=True)
    parser.add_argument('--intrinsics_json_path', type=str,
                        help='path to a json file containing a normalised 3x3 intrinsics matrix',
                        required=True)
    parser.add_argument('--model_path', type=str,
                        help='path to a folder of weights to load', required=True)
    parser.add_argument('--mode', type=str, default='multi', choices=('multi', 'mono'),
                        help='"multi" or "mono". If set to "mono" then the network is run without '
                             'the source image, e.g. as described in Table 5 of the paper.',
                        required=False)
    return parser.parse_args()


def load_and_preprocess_image(image_path, resize_width, resize_height):
    image = pil.open(image_path).convert('RGB')
    original_width, original_height = image.size
    image = image.resize((resize_width, resize_height), pil.LANCZOS)
    image = transforms.ToTensor()(image).unsqueeze(0)
    if torch.cuda.is_available():
        return image.cuda(), (original_height, original_width)
    return image, (original_height, original_width)


def load_and_preprocess_intrinsics(intrinsics_path, resize_width, resize_height):
    K = np.array([[0.798, 0, 0.5, 0],
                           [0, 1.596, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
    K[0, :] *= resize_width // 4
    K[1, :] *= resize_height // 4

    invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
    K = torch.Tensor(K).unsqueeze(0)

    if torch.cuda.is_available():
        return K.cuda(), invK.cuda()
    return K, invK


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_path is not None, \
        "You must specify the --model_path parameter"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("-> Loading model from ", args.model_path)

    # Loading pretrained model
    print("   Loading pretrained encoder")
    encoder_dict = torch.load(os.path.join(args.model_path, "encoder.pth"), map_location=device)
    encoder = networks.ResnetEncoderMatching(18, False,
                                             input_width=encoder_dict['width'],
                                             input_height=encoder_dict['height'],
                                             adaptive_bins=True,
                                             min_depth_bin=encoder_dict['min_depth_bin'],
                                             max_depth_bin=encoder_dict['max_depth_bin'],
                                             depth_binning='linear',
                                             num_depth_bins=96)

    filtered_dict_enc = {k: v for k, v in encoder_dict.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(os.path.join(args.model_path, "depth.pth"), map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    # Setting states of networks
    encoder.eval()
    depth_decoder.eval()
    if torch.cuda.is_available():
        encoder.cuda()
        depth_decoder.cuda()
    root_dir = '/home/hkr/Dataset/iphone12_collect_data'
    test_file_txt = '/home/hkr/Code/manydepth/splits/iphone/train_files.txt'
    test_filenames = readlines(test_file_txt)
    for i in range(len(test_filenames)):
        row = test_filenames[i]
        image_triplet_path = os.path.join(root_dir,str(row[0]),'trained_images',str(row[1])+ '.png')
        sparse_depth_npy_path = os.path.join(root_dir,str(row[0]),'sparse_depth',str(row[1])+ '.npy') 
        sparse_depth0 = load_depth(sparse_depth_npy_path)
        sparse_depth0 = torch.tensor(sparse_depth0, dtype=torch.float32)
        flat_tensor = torch.flatten(sparse_depth0)
        percentile_value = torch.max(flat_tensor[(flat_tensor > 0) & (flat_tensor < 100)]).cpu().numpy()
        print("percentile_value:",percentile_value)
        # percentile_index = int(0.99 * len(sorted_tensor))
        # percentile_value = sorted_tensor[percentile_index].cpu().numpy()
        max_depth_from_vio = min(percentile_value * 1.2 ,80)
        image1, image0, image2 = load_image_triplet(
            image_triplet_path,
            normalize=False)
        q_cur2last = np.array([float(row[5]),float(row[6]),float(row[7]),float(row[8])])
        t_cur2last = np.array([float(row[2]),float(row[3]),float(row[4])])
        q_cur2future = np.array([float(row[12]),float(row[13]),float(row[14]),float(row[15])])
        t_cur2future = np.array([float(row[9]),float(row[10]),float(row[11])])
        T_last_cur = qt2T(q_cur2last, t_cur2last)
        T_future_cur = qt2T(q_cur2future, t_cur2future)
        T_last_cur,T_future_cur = [
            T.astype(np.float32)
            for T in [T_last_cur,T_future_cur]
        ]
        image0 = transforms.ToTensor()(image0).unsqueeze(0).cuda()
        image1 = transforms.ToTensor()(image1).unsqueeze(0).cuda()
        image2 = transforms.ToTensor()(image2).unsqueeze(0).cuda()
        T_last_cur = torch.tensor(T_last_cur, dtype=torch.float32).unsqueeze(0).cuda()
        T_future_cur = torch.tensor(T_future_cur, dtype=torch.float32).unsqueeze(0).cuda()
        lookup_frames = torch.cat([image1.unsqueeze(1), image2.unsqueeze(1)], dim=1)
        relative_poses = torch.cat([T_last_cur.unsqueeze(1), T_future_cur.unsqueeze(1)], dim=1)
        K, invK = load_and_preprocess_intrinsics(args.intrinsics_json_path,
                                             resize_width=encoder_dict['width'],
                                             resize_height=encoder_dict['height'])
        with torch.no_grad():
            # Estimate depth
            output, lowest_cost, _ = encoder(current_image=image0,
                                            lookup_images=lookup_frames,
                                            poses=relative_poses,
                                            K=K,
                                            invK=invK,
                                            min_depth_bin=encoder_dict['min_depth_bin'],
                                            max_depth_bin=max_depth_from_vio)

            output = depth_decoder(output)

            sigmoid_output = output[("disp", 0)]
            _, depth = disp_to_depth(sigmoid_output, encoder_dict['min_depth_bin'], max_depth_from_vio)
            depth = depth.cpu().numpy()
            max_value = np.max(depth)
            mean_value = np.mean(depth)

            print("最大值:", max_value)
            print("平均值:", mean_value)
            # outputs[("depth", 0, scale)] = depth
            original_size = (encoder_dict['height'], encoder_dict['width'])
            sigmoid_output_resized = torch.nn.functional.interpolate(
                sigmoid_output, original_size, mode="bilinear", align_corners=False)
            sigmoid_output_resized = sigmoid_output_resized.cpu().numpy()[:, 0]

            # # Saving numpy file
            directory = '/home/hkr/Code/many_depth_vio_debug/mdp/models/test_output'
            output_name = str(row[0]) + '_' + str(row[1])

            name_dest_npy = os.path.join(directory, "{}_depth_{}.npy".format(output_name, args.mode))
            np.save(name_dest_npy, depth)
            np.save('depth_array.npy', depth)
            # # Saving colormapped depth image and cost volume argmin
            for plot_name, toplot in (('costvol_min', lowest_cost), ('disp', sigmoid_output_resized)):
                toplot = toplot.squeeze()
                normalizer = mpl.colors.Normalize(vmin=toplot.min(), vmax=np.percentile(toplot, 95))
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(toplot)[:, :, :3] * 255).astype(np.uint8)
                origin_img_path = os.path.join(root_dir,str(row[0]),'images',str(row[1])+ '.jpg')
                origin_img = load_image(origin_img_path,normalize=False, data_format='HWC')
                print(origin_img.shape)
                print(colormapped_im.shape)
                if origin_img.shape == colormapped_im.shape:
                    colormapped_im = np.concatenate([origin_img,colormapped_im], axis=1)
                    im = pil.fromarray(colormapped_im)
                    name_dest_im = os.path.join(directory,
                                            "{}_{}_{}.jpeg".format(output_name, plot_name, args.mode))
                    im.save(name_dest_im)
                    print("-> Saved output image to {}".format(name_dest_im))
        break
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
