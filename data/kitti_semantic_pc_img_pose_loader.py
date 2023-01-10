import open3d
import torch.utils.data as data
import random
import numbers
import os
import os.path
import numpy as np
import struct
import math
import torch
import torchvision
import cv2
from PIL import Image
from torchvision import transforms
import time

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

sys.path.append(os.getcwd())

from data import augmentation
from util import vis_tools
from kitti import options_clip
from data.kitti_helper import *
import json


def downsample_with_intensity_sn(pointcloud, intensity, sn, label, voxel_grid_downsample_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
    intensity_max = np.max(intensity)

    fake_colors = np.zeros((pointcloud.shape[1], 3))
    fake_colors[:, 0:1] = np.transpose(intensity) / intensity_max
    fake_colors[:, 1:2] = np.transpose(label) / 10

    pcd.colors = open3d.utility.Vector3dVector(fake_colors)
    pcd.normals = open3d.utility.Vector3dVector(np.transpose(sn))

    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = down_pcd_points

    intensity = np.transpose(np.asarray(down_pcd.colors)[:, 0:1]) * intensity_max
    label = np.transpose(np.asarray(down_pcd.colors)[:, 1:2]) * 10
    sn = np.transpose(np.asarray(down_pcd.normals))

    return pointcloud, intensity, sn, label

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def make_kitti_dataset(root_path, mode, opt):
    dataset = []

    if mode == 'train':
        seq_list = [0,5,6,7,8] 
    elif 'val' in mode:
        seq_list = [2]
    else:
        raise Exception('Invalid mode.')

    np_folder = 'voxel0.1-SNr0.6'
    # np_folder = 'stride4-acc50-voxel0.4'
    skip_start_end = 40
    # 这里是载入双目的图像，然后结成对
    for seq in seq_list:
        pc_nwu_folder = os.path.join(root_path, 'data_odometry_semantic_velodyne_clippre', 'sequences', '%02d' % seq, np_folder)
        pc_graph_folder = os.path.join(root_path, 'graph', '%02d' % seq, 'lidar_gn_clip')
        seimg2_folder = os.path.join(root_path, 'data_odometry_semantic_npy_pre', 'sequences', '%02d' % seq, 'image_02')
        seimg3_folder = os.path.join(root_path, 'data_odometry_semantic_npy_pre', 'sequences', '%02d' % seq, 'image_03')
        img_graph_folder = os.path.join(root_path, 'data_odometry_semantic_graph', 'sequences', '%02d' % seq, 'image_02')
        # se2_folder = os.path.join(root_path, 'data_odometry_semantic_npy', 'sequences', '%02d' % seq, 'image_2_e')
        # se3_folder = os.path.join(root_path, 'data_odometry_semantic_npy', 'sequences', '%02d' % seq, 'image_3_e')
        pair_folder = os.path.join(root_path, 'pair_list_1_20', '%02d' % seq,'%02d.txt' % seq)
        pose_folder = os.path.join(root_path, 'poses', '%02d' % seq)

        sample_num = round(len(os.listdir(pose_folder)))

        with open(pair_folder) as f:
            while True:
                line = f.readline()
                if not line: break
                line = line.rstrip('\r\n')
                # print(line)
                lidar_index,image2_index,flag = line.split(' ')
                # pc_nwu_file = os.path.join(pc_nwu_folder,'%06d.npy' %int(lidar_index))
                # image2_file = os.path.join(img2_folder,'%06d.npy' %int(image2_index))
                # image3_file = os.path.join(img3_folder,'%06d.npy' %int(image2_index))
                # dataset_new.append((pc_nwu_file, image2_file, 'P2', flag, sample_num))
                # dataset_new.append((pc_nwu_file, image3_file, 'P3', flag, sample_num))
                dataset.append((pc_nwu_folder,pc_graph_folder,int(lidar_index), seimg2_folder, img_graph_folder, int(image2_index), seq, sample_num, 'P2', flag))
                dataset.append((pc_nwu_folder,pc_graph_folder,int(lidar_index), seimg3_folder, img_graph_folder, int(image2_index), seq, sample_num, 'P3', flag))
                # print(lidar_index,"***", image2_index, "****",flag)

        # for i in range(skip_start_end, sample_num-skip_start_end):
        #     dataset.append((pc_nwu_folder, img2_folder, seq, i, sample_num, 'P2'))
        #     dataset.append((pc_nwu_folder, img3_folder, seq, i, sample_num, 'P3'))

    return dataset

def vlad_make_kitti_dataset(root_path, mode, opt):
    dataset = []

    if mode == 'train':
        seq_list = [2,5,6,7,8] 
    elif 'val' in mode:
        seq_list = [0]
    else:
        raise Exception('Invalid mode.')

    np_folder = 'voxel0.1-SNr0.6'
    # np_folder = 'stride4-acc50-voxel0.4'
    skip_start_end = 40
    # 这里是载入双目的图像，然后结成对
    for seq in seq_list:
        pc_nwu_folder = os.path.join(root_path, 'data_odometry_semantic_velodyne_pre', 'sequences', '%02d' % seq, np_folder)
        pc_graph_folder = os.path.join(root_path, 'graph', '%02d' % seq, 'lidar_gn_clip')
        seimg2_folder = os.path.join(root_path, 'data_odometry_semantic_npy_pre', 'sequences', '%02d' % seq, 'image_02')
        seimg3_folder = os.path.join(root_path, 'data_odometry_semantic_npy_pre', 'sequences', '%02d' % seq, 'image_03')
        img_graph_folder = os.path.join(root_path, 'data_odometry_semantic_graph', 'sequences', '%02d' % seq, 'image_02')
        # se2_folder = os.path.join(root_path, 'data_odometry_semantic_npy', 'sequences', '%02d' % seq, 'image_2_e')
        # se3_folder = os.path.join(root_path, 'data_odometry_semantic_npy', 'sequences', '%02d' % seq, 'image_3_e')
        pair_folder = os.path.join(root_path, 'pair_list_1_20', '%02d' % seq,'%02d.txt' % seq)
        pose_folder = os.path.join(root_path, 'poses', '%02d' % seq)

        sample_num = round(len(os.listdir(pose_folder)))

        with open(pair_folder) as f:
            while True:
                line = f.readline()
                if not line: break
                line = line.rstrip('\r\n')
                # print(line)
                lidar_index,image2_index,flag = line.split(' ')
                # pc_nwu_file = os.path.join(pc_nwu_folder,'%06d.npy' %int(lidar_index))
                # image2_file = os.path.join(img2_folder,'%06d.npy' %int(image2_index))
                # image3_file = os.path.join(img3_folder,'%06d.npy' %int(image2_index))
                # dataset_new.append((pc_nwu_file, image2_file, 'P2', flag, sample_num))
                # dataset_new.append((pc_nwu_file, image3_file, 'P3', flag, sample_num))
                dataset.append((pc_nwu_folder,int(lidar_index), pc_nwu_folder, int(image2_index), seq, sample_num, 'P2', flag))
                # dataset.append((pc_nwu_folder,pc_graph_folder,int(lidar_index), seimg3_folder, img_graph_folder, int(image2_index), seq, sample_num, 'P3', flag))
                # print(lidar_index,"***", image2_index, "****",flag)

        # for i in range(skip_start_end, sample_num-skip_start_end):
        #     dataset.append((pc_nwu_folder, img2_folder, seq, i, sample_num, 'P2'))
        #     dataset.append((pc_nwu_folder, img3_folder, seq, i, sample_num, 'P3'))

    return dataset


def transform_pc_np(P, pc_np):
    """
    :param pc_np: 3xN
    :param P: 4x4
    :return:
    """
    pc_homo_np = np.concatenate((pc_np,
                                 np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                axis=0)
    P_pc_homo_np = np.dot(P, pc_homo_np)
    return P_pc_homo_np[0:3, :]


class SeKittiLoader(data.Dataset):
    def __init__(self, root, mode, opt: options_clip.Options):
        # root表示的是文件的路径
        super(SeKittiLoader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode
        self.global_labels = [i for i in range(12)] # 20
        self.global_labels = {val: index for index, val in enumerate(self.global_labels)}
        self.number_of_labels = len(self.global_labels)

        # farthest point sample
        self.farthest_sampler = FarthestSampler(dim=3)

        # store the calibration matrix for each sequence
        self.calib_helper = KittiCalibHelper(root)
        # print(self.calib_helper.calib_matrix_dict)

        # list of (pc_path, img_path, seq, i, img_key)
        self.dataset = make_kitti_dataset(root, mode, opt)

    def augment_pc(self, pc_np, intensity_np, sn_np):
        """

        :param pc_np: 3xN, np.ndarray
        :param intensity_np: 3xN, np.ndarray
        :return:
        """
        # add Gaussian noise
        pc_np = augmentation.jitter_point_cloud(pc_np, sigma=0.01, clip=0.05)
        sn_np = augmentation.jitter_point_cloud(sn_np, sigma=0.01, clip=0.05)
        return pc_np, intensity_np, sn_np

    def augment_img(self, img_np):
        """

        :param img: HxWx4, np.ndarray
        :return:
        """
        # print('success coming here ')
        # print('the img shape is ',img_np.shape)  #160 * 512 * 3
        # color perturbation
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        # tmp = Image.fromarray(img_np)
        # print('the color augment res is *****************')
        # print(tmp)
        color_aug = transforms.ColorJitter.get_params(brightness, contrast, saturation, hue)
        # print(img_np[:,:,:3])
        # img_color_aug_np = np.array(color_aug(Image.fromarray(img_np[:,:,:3])))
        img_color_aug_np = np.concatenate((np.array(color_aug(Image.fromarray(img_np[:,:,:3]))),img_np[:,:,3:4]), axis = 2)
        # print(img_color_aug_np)

        return img_color_aug_np

    def generate_random_transform(self,
                                  P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                                  P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude):
        """

        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-P_tx_amplitude, P_tx_amplitude),
             random.uniform(-P_ty_amplitude, P_ty_amplitude),
             random.uniform(-P_tz_amplitude, P_tz_amplitude)]
        angles = [random.uniform(-P_Rx_amplitude, P_Rx_amplitude),
                  random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
                  random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

        rotation_mat = augmentation.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random

    def downsample_np(self, pc_np, intensity_np, sn_np, label_np):
        if pc_np.shape[1] >= self.opt.input_pt_num:
            choice_idx = np.random.choice(pc_np.shape[1], self.opt.input_pt_num, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.opt.input_pt_num:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.opt.input_pt_num - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]
        sn_np = sn_np[:, choice_idx]
        label_np = label_np[:, choice_idx]

        return pc_np, intensity_np, sn_np, label_np

    def get_sequence_j(self, seq_sample_num, seq_i, seq_pose_folder,
                       delta_ij_max, translation_max):
        # get the max and min of possible j   也是设定一个上下边界
        seq_j_min = max(seq_i - delta_ij_max, 0)
        seq_j_max = min(seq_i + delta_ij_max, seq_sample_num - 1)

        # pose of i
        Pi = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_i))['pose'].astype(np.float32)  # 4x4


        while True:
            seq_j = random.randint(seq_j_min, seq_j_max)
            # get the pose, if the pose is too large, ignore and re-sample
            Pj = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_j))['pose'].astype(np.float32)  # 4x4
            Pji = np.dot(np.linalg.inv(Pj), Pi)  # 4x4
            t_ji = Pji[0:3, 3]  # 3
            t_ji_norm = np.linalg.norm(t_ji)  # scalar

            if t_ji_norm < translation_max:
                break
            else:
                continue

        return seq_j, Pji, t_ji


    def search_for_accumulation(self, pc_folder, seq_pose_folder,
                                seq_i, seq_sample_num, Pc, P_oi,
                                stride):
        Pc_inv = np.linalg.inv(Pc)
        P_io = np.linalg.inv(P_oi)

        pc_np_list, intensity_np_list, sn_np_list = [], [], []

        # print("come here ????????")

        counter = 0
        while len(pc_np_list) < self.opt.accumulation_frame_num:
            counter += 1
            seq_j = seq_i + stride * counter
            if seq_j < 0 or seq_j >= seq_sample_num:
                break

            npy_data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_j)).astype(np.float32)
            pc_np = npy_data[0:3, :]  # 3xN
            intensity_np = npy_data[3:4, :]  # 1xN
            sn_np = npy_data[4:7, :]  # 3xN

            P_oj = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_j))['pose'].astype(np.float32)  # 4x4
            P_ij = np.dot(P_io, P_oj)

            P_transform = np.dot(Pc_inv, np.dot(P_ij, Pc))
            pc_np = transform_pc_np(P_transform, pc_np)
            P_transform_rot = np.copy(P_transform)
            P_transform_rot[0:3, 3] = 0
            sn_np = transform_pc_np(P_transform_rot, sn_np)

            pc_np_list.append(pc_np)
            intensity_np_list.append(intensity_np)
            sn_np_list.append(sn_np)

        return pc_np_list, intensity_np_list, sn_np_list

    # seq_i代表的是选中的是第几帧
    # 好像就是加了一点数据增强，我真是谢谢了
    def get_accumulated_pc(self, pc_folder, seq_pose_folder, seq_i, seq_sample_num, Pc):
        pc_path = os.path.join(pc_folder, '%06d.npy' % seq_i)
        npy_data = np.load(pc_path).astype(np.float32)
        # shuffle the point cloud data, this is necessary!
        npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
        pc_np = npy_data[0:3, :]  # 3xN
        intensity_np = npy_data[3:4, :]  # 1xN
        # sn_np表示的是每个点的向量
        sn_np = npy_data[4:7, :]  # 3xN
        label_np = npy_data[7:, :]

        # print(sn_np.shape)
        # print(label_np.shape)
        # print(label_np)

        # 堆积帧的数量？这玩意不知道干啥的
        if self.opt.accumulation_frame_num <= 0.5:
            # print("come here **************")
            return pc_np, intensity_np, sn_np, label_np

        pc_np_list = [pc_np]
        intensity_np_list = [intensity_np]
        sn_np_list = [sn_np]

        # pose of i  取出第i帧的位姿  好像这里拆出来就是一个简单的4*4矩阵？
        P_oi = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_i))['pose'].astype(np.float32) # 4x4
        # print("success load the pose file **********************")
        # print(P_oi.shape)

        # 看了半天，感觉这玩意。。。像个数据增强
        # search for previous  这个是往前找accumulation_frame_num帧
        prev_pc_np_list, \
        prev_intensity_np_list, \
        prev_sn_np_list = self.search_for_accumulation(pc_folder,
                                                       seq_pose_folder,
                                                       seq_i,
                                                       seq_sample_num,
                                                       Pc,
                                                       P_oi,
                                                       -self.opt.accumulation_frame_skip)
        # search for next   往后找
        next_pc_np_list, \
        next_intensity_np_list, \
        next_sn_np_list = self.search_for_accumulation(pc_folder,
                                                       seq_pose_folder,
                                                       seq_i,
                                                       seq_sample_num,
                                                       Pc,
                                                       P_oi,
                                                       self.opt.accumulation_frame_skip)

        # print("success the work")

        pc_np_list = pc_np_list + prev_pc_np_list + next_pc_np_list
        intensity_np_list = intensity_np_list + prev_intensity_np_list + next_intensity_np_list
        sn_np_list = sn_np_list + prev_sn_np_list + next_sn_np_list

        pc_np = np.concatenate(pc_np_list, axis=1)
        intensity_np = np.concatenate(intensity_np_list, axis=1)
        sn_np = np.concatenate(sn_np_list, axis=1)

        return pc_np, intensity_np, sn_np, label_np



    def __len__(self):
        return len(self.dataset)

    # 这个接口是遍历的时候会用
    def __getitem__(self, index):
        # 记录一下时间
        begin  = time.time()
        # print("are you ok ????")
        # 首先，将前面的这些信息进行一个解析
        # dataset_new.append((pc_nwu_file, image3_file, 'P3', flag))
        # dataset.append((pc_nwu_folder,lidar_index, img2_folder,image2_index, seq, sample_num, 'P2'))
        pc_folder,pc_graph_folder,lidar_index, seimg_folder,img_graph_folder, img_index , seq, seq_sample_num, img_key, target = self.dataset[index]

        # print("the llidar_index is ", lidar_index, " the img_index is ", img_index, "and the target is ", target)
        # pc_file, img_file,img_key_new, target, seq_sample_num = self.dataset_new[index]


        # print("seq_sample_num is : " ,seq_sample_num)
        # 先提取雷达语义图节点的信息
        graph_file = os.path.join(pc_graph_folder,'%010d.json' % lidar_index)
        graph_data = json.load(open(graph_file))
        # graph_label_np = np.array(graph_data["nodes"])

        # 然后要进行一个语义图节点的补充,补充为-1
        node_num =len(graph_data["nodes"])

        # 如果图节点的数量大于需要的数目：
        if node_num > self.opt.node_num:
            sampled_index = np.random.choice(node_num, self.opt.node_num, replace=False)
            sampled_index.sort()
            graph_data["nodes"] = np.array(graph_data["nodes"])[sampled_index].tolist()
            # 补充节点
        elif node_num < self.opt.node_num:
            graph_data["nodes"] = np.concatenate(
                (np.array(graph_data["nodes"]), -np.ones(self.opt.node_num - node_num))).tolist()   # padding 0

        # graph_label_np = np.array(graph_data["nodes"])

        # 然后进行一个one_hot编码
        features_1 = np.expand_dims(np.array(
            [np.zeros(self.number_of_labels).tolist() if node == -1 else [
                1.0 if self.global_labels[node] == label_index else 0 for label_index in self.global_labels.values()]
                for node in graph_data["nodes"]]), axis=0)

        features = np.squeeze(features_1)

        # print(features_1)
        # 下面来提取视觉图节点信息
        vision_node = np.load(os.path.join(img_graph_folder,'%06d.npy' % img_index))
        node2_num = vision_node.shape[0]
        # 如果图节点的数量大于需要的数目：
        if node2_num > self.opt.node_num:
            sampled_index = np.random.choice(node2_num, self.opt.node_num, replace=False)
            sampled_index.sort()
            vision_node = vision_node[sampled_index]
        # 补充节点
        elif node2_num < self.opt.node_num:
            vision_node = np.concatenate(
                (vision_node, -np.ones(self.opt.node_num - node2_num)))   # padding 0

        # 然后进行一个one_hot编码
        features_2 = np.expand_dims(np.array(
            [np.zeros(self.number_of_labels).tolist() if node == -1 else [
                1.0 if self.global_labels[node] == label_index else 0 for label_index in self.global_labels.values()]
                for node in vision_node]), axis=0)

        # 然后去读取一个坐标信息
        seq_pose_folder = os.path.join(self.root, 'poses', '%02d' % seq)

        # print("seq_pose_folder is here !!!!!!!!!!!!!!!!!!!!!!!!!")

        # load point cloud of seq_i
        # 这里感觉是得到了两者之间的变换
        Pc = np.dot(self.calib_helper.get_matrix(seq, img_key),
                    self.calib_helper.get_matrix(seq, 'Tr'))

        # 获取当前点云和加上了数据增强的点云列表
        pc_np, intensity_np, sn_np, label_np = self.get_accumulated_pc(pc_folder, seq_pose_folder, lidar_index, seq_sample_num, Pc)
        # t1 = time.time()
        # print("get the lidar dataset cost:", begin- t1)

        # print("success complate the accumulated work ****************")

        # 如果点云太多，就使用滤波操作 (已在预处理中处理)
        # if pc_np.shape[1] > 2 * self.opt.input_pt_num:
        #     # point cloud too huge, voxel grid downsample first
        #     pc_np, intensity_np, sn_np, label_np = downsample_with_intensity_sn(pc_np, intensity_np, sn_np, label_np,
        #                                                     voxel_grid_downsample_size=0.3)
        #     pc_np = pc_np.astype(np.float32)
        #     intensity_np = intensity_np.astype(np.float32)
        #     sn_np = sn_np.astype(np.float32)
        #     label_np = label_np.astype(np.float32)
            # random downsample to a specific shape, pc is still in NWU coordinate

        # 这里使点云的数量获得统一 已在预处理中处理
        # pc_np, intensity_np, sn_np, label_np = self.downsample_np(pc_np, intensity_np, sn_np, label_np)
        # print("success load the pc data")
        # print(pc_np.shape)
        # print(intensity_np.shape)
        # print(sn_np.shape)
        # print(label_np.shape)
        pc_end = time.time()
        # print('process the pc data cost: ', pc_end - begin)

        # limit max_z, the pc is in NWU coordinate
        # pc_np_x_square = np.square(pc_np[0, :])
        # pc_np_y_square = np.square(pc_np[1, :])
        # pc_np_range_square = pc_np_x_square + pc_np_y_square
        # pc_mask_range = pc_np_range_square < self.opt.pc_max_range * self.opt.pc_max_range
        # pc_np = pc_np[:, pc_mask_range]
        # intensity_np = intensity_np[:, pc_mask_range]

        # 接下来是处理图像数据
        # load image of seq_j
        if self.opt.translation_max < 0:
            seq_j = img_index
            Pji = np.identity(4)
            t_ji = Pji[0:3, 3]
        else:
            # 这里好像也是做一个图像增强，但是获得了好像只有一个
            seq_j, Pji, t_ji = self.get_sequence_j(seq_sample_num, img_index, seq_pose_folder,
                                                   self.opt.delta_ij_max, self.opt.translation_max)
        t1 = time.time()
        # print("img augment cost:", t1 - pc_end)

        # 这里读取img文件，同样也是np格式
        img_path = os.path.join(seimg_folder, '%06d.npy' % seq_j)
        img = np.load(img_path).astype(np.uint8)
        K = self.calib_helper.get_matrix(seq, img_key + '_K')
        # crop the first few rows, original is 370x1226 now
        # print(img.shape)
        # img = img[self.opt.crop_original_top_rows:, :, :]
        # img_rgb = img[:,:,:3]
        # img_label = img[:,:,3:4]
        # print(img)
        # print(img.shape)
        K = camera_matrix_cropping(K, dx=0, dy=self.opt.crop_original_top_rows)
        # scale  图像缩放处理
        # img = cv2.resize(img,
        #             (int(round(img.shape[1] * self.opt.img_scale)),
        #             int(round((img.shape[0] * self.opt.img_scale)))),
        #             interpolation=cv2.INTER_LINEAR)
        K = camera_matrix_scaling(K, self.opt.img_scale)
        # print(img.shape)
        # img = np.concatenate((img_rgb, img_label), axis = 2)
        # print("the resize is success ")
        # print(img.shape)
        t2 = time.time()
        # print("clip the img cost: ", t2 - t1)

        # random crop into input size
        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.opt.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.opt.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.opt.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.opt.img_H) / 2)
        # crop image
        img = img[img_crop_dy:img_crop_dy + self.opt.img_H,
              img_crop_dx:img_crop_dx + self.opt.img_W, :]
        K = camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)

        img_end = time.time()
        # t3 = time.time()
        # print("other process cost: ", img_end - t2)
        # print('process the img data cost: ', img_end - pc_end)

        # print("success load the img data")

        # 接下来好像是整体的加一些数据增强
        #  ------------- apply random transform on points under the NWU coordinate ------------
        if 'train' == self.mode:
            Pr = self.generate_random_transform(self.opt.P_tx_amplitude, self.opt.P_ty_amplitude, self.opt.P_tz_amplitude,
                                                self.opt.P_Rx_amplitude, self.opt.P_Ry_amplitude, self.opt.P_Rz_amplitude)
            Pr_inv = np.linalg.inv(Pr)

            # print('success create the pr data')

            # -------------- augmentation ----------------------
            pc_np, intensity_np, sn_np = self.augment_pc(pc_np, intensity_np, sn_np)
            # print('success finish the pc augment !')

            # 这里有一些问题
            img = self.augment_img(img)
            # print(img)
            # print(img.shape)
            # print('success finish the img augment !!')
            if random.random() > 0.5:
                img = np.flip(img, 1)
                P_flip = np.asarray([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=pc_np.dtype)
                Pr = np.dot(Pr, P_flip)
            Pr_inv = np.linalg.inv(Pr)
            # print('success complate the data augment')
        elif 'val_random_Ry' == self.mode:
            Pr = self.generate_random_transform(0, 0, 0,
                                                0, math.pi*2, 0)
            Pr_inv = np.linalg.inv(Pr)
        else:
            Pr = np.identity(4, dtype=np.float)
            Pr_inv = np.identity(4, dtype=np.float)

        P_cam_nwu = np.asarray([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=pc_np.dtype)
        P_nwu_cam = np.linalg.inv(P_cam_nwu)

        # now the point cloud is in CAMERA coordinate   将雷达点云转换到视觉坐标系下
        Pr_Pcamnwu = np.dot(Pr, P_cam_nwu)
        pc_np = transform_pc_np(Pr_Pcamnwu, pc_np)
        sn_np = transform_pc_np(Pr_Pcamnwu, sn_np)

        # assemble P. P * pc will get the point cloud in the camera image coordinate
        PcPnwucamPrinv = np.dot(Pc, np.dot(P_nwu_cam, Pr_inv))
        P = np.dot(Pji, PcPnwucamPrinv)  # 4x4


        # ------------ Farthest Point Sampling ------------------  进行一个快速抽样
        # node_a_np = fps_approximate(pc_np, voxel_size=4.0, node_num=self.opt.node_a_num)
        node_a_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                              self.opt.node_a_num * 8,
                                                                              replace=False)],
                                                    k=self.opt.node_a_num)
        node_b_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                            self.opt.node_b_num * 8,
                                                                            replace=False)],
                                                  k=self.opt.node_b_num)

        # visualize nodes
        # ax = vis_tools.plot_pc(pc_np, size=1)
        # ax = vis_tools.plot_pc(node_a_np, size=10, ax=ax)
        # plt.show()

        # -------------- convert to torch tensor --------------------- 转换为torch模式
        pc = torch.from_numpy(pc_np)  # 3xN
        intensity = torch.from_numpy(intensity_np)  # 1xN
        sn = torch.from_numpy(sn_np)  # 3xN
        label_np = torch.from_numpy(label_np)  # 1xN
        node_a = torch.from_numpy(node_a_np)  # 3xMa
        node_b = torch.from_numpy(node_b_np)  # 3xMb
        pc_graph_label = torch.from_numpy(features)
        img_graph_label = torch.from_numpy(features_2).squeeze(0)

        P = torch.from_numpy(P[0:3, :].astype(np.float32))  # 3x4

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()  # 3xHxW
        K = torch.from_numpy(K.astype(np.float32))  # 3x3

        t_ji = torch.from_numpy(t_ji.astype(np.float32))  # 3

        target = torch.from_numpy(np.asarray(int(target)))

        # print(target)

        end = time.time()
        # print("other time cost: ", end - img_end)
        # print('get one item cost: ',end - begin)

        return pc, intensity, sn, label_np, node_a, node_b, pc_graph_label, img_graph_label, \
               P, img, K, \
               t_ji, target

class PointnetVladKittiLoader(data.Dataset):
    def __init__(self, root, mode, opt: options_clip.Options):
        # root表示的是文件的路径
        super(PointnetVladKittiLoader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode
        self.global_labels = [i for i in range(12)] # 20
        self.global_labels = {val: index for index, val in enumerate(self.global_labels)}
        self.number_of_labels = len(self.global_labels)

        # farthest point sample
        self.farthest_sampler = FarthestSampler(dim=3)

        # store the calibration matrix for each sequence
        self.calib_helper = KittiCalibHelper(root)
        # print(self.calib_helper.calib_matrix_dict)

        # list of (pc_path, img_path, seq, i, img_key)
        self.dataset = vlad_make_kitti_dataset(root, mode, opt)

    def augment_pc(self, pc_np, intensity_np, sn_np):
        """

        :param pc_np: 3xN, np.ndarray
        :param intensity_np: 3xN, np.ndarray
        :return:
        """
        # add Gaussian noise
        pc_np = augmentation.jitter_point_cloud(pc_np, sigma=0.01, clip=0.05)
        sn_np = augmentation.jitter_point_cloud(sn_np, sigma=0.01, clip=0.05)
        return pc_np, intensity_np, sn_np

    def augment_img(self, img_np):
        """

        :param img: HxWx4, np.ndarray
        :return:
        """
        # print('success coming here ')
        # print('the img shape is ',img_np.shape)  #160 * 512 * 3
        # color perturbation
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        # tmp = Image.fromarray(img_np)
        # print('the color augment res is *****************')
        # print(tmp)
        color_aug = transforms.ColorJitter.get_params(brightness, contrast, saturation, hue)
        # print(img_np[:,:,:3])
        # img_color_aug_np = np.array(color_aug(Image.fromarray(img_np[:,:,:3])))
        img_color_aug_np = np.concatenate((np.array(color_aug(Image.fromarray(img_np[:,:,:3]))),img_np[:,:,3:4]), axis = 2)
        # print(img_color_aug_np)

        return img_color_aug_np

    def generate_random_transform(self,
                                  P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                                  P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude):
        """

        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-P_tx_amplitude, P_tx_amplitude),
             random.uniform(-P_ty_amplitude, P_ty_amplitude),
             random.uniform(-P_tz_amplitude, P_tz_amplitude)]
        angles = [random.uniform(-P_Rx_amplitude, P_Rx_amplitude),
                  random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
                  random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

        rotation_mat = augmentation.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random

    def downsample_np(self, pc_np, intensity_np, sn_np, label_np):
        if pc_np.shape[1] >= self.opt.input_pt_num:
            choice_idx = np.random.choice(pc_np.shape[1], self.opt.input_pt_num, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.opt.input_pt_num:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.opt.input_pt_num - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]
        sn_np = sn_np[:, choice_idx]
        label_np = label_np[:, choice_idx]

        return pc_np, intensity_np, sn_np, label_np

    def get_sequence_j(self, seq_sample_num, seq_i, seq_pose_folder,
                       delta_ij_max, translation_max):
        # get the max and min of possible j   也是设定一个上下边界
        seq_j_min = max(seq_i - delta_ij_max, 0)
        seq_j_max = min(seq_i + delta_ij_max, seq_sample_num - 1)

        # pose of i
        Pi = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_i))['pose'].astype(np.float32)  # 4x4


        while True:
            seq_j = random.randint(seq_j_min, seq_j_max)
            # get the pose, if the pose is too large, ignore and re-sample
            Pj = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_j))['pose'].astype(np.float32)  # 4x4
            Pji = np.dot(np.linalg.inv(Pj), Pi)  # 4x4
            t_ji = Pji[0:3, 3]  # 3
            t_ji_norm = np.linalg.norm(t_ji)  # scalar

            if t_ji_norm < translation_max:
                break
            else:
                continue

        return seq_j, Pji, t_ji


    def search_for_accumulation(self, pc_folder, seq_pose_folder,
                                seq_i, seq_sample_num, Pc, P_oi,
                                stride):
        Pc_inv = np.linalg.inv(Pc)
        P_io = np.linalg.inv(P_oi)

        pc_np_list, intensity_np_list, sn_np_list = [], [], []

        # print("come here ????????")

        counter = 0
        while len(pc_np_list) < self.opt.accumulation_frame_num:
            counter += 1
            seq_j = seq_i + stride * counter
            if seq_j < 0 or seq_j >= seq_sample_num:
                break

            npy_data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_j)).astype(np.float32)
            pc_np = npy_data[0:3, :]  # 3xN
            intensity_np = npy_data[3:4, :]  # 1xN
            sn_np = npy_data[4:7, :]  # 3xN

            P_oj = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_j))['pose'].astype(np.float32)  # 4x4
            P_ij = np.dot(P_io, P_oj)

            P_transform = np.dot(Pc_inv, np.dot(P_ij, Pc))
            pc_np = transform_pc_np(P_transform, pc_np)
            P_transform_rot = np.copy(P_transform)
            P_transform_rot[0:3, 3] = 0
            sn_np = transform_pc_np(P_transform_rot, sn_np)

            pc_np_list.append(pc_np)
            intensity_np_list.append(intensity_np)
            sn_np_list.append(sn_np)

        return pc_np_list, intensity_np_list, sn_np_list

    # seq_i代表的是选中的是第几帧
    # 好像就是加了一点数据增强，我真是谢谢了
    def get_accumulated_pc(self, pc_folder, seq_pose_folder, seq_i, seq_sample_num, Pc):
        pc_path = os.path.join(pc_folder, '%06d.npy' % seq_i)
        npy_data = np.load(pc_path).astype(np.float32)
        # shuffle the point cloud data, this is necessary!
        npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
        pc_np = npy_data[0:3, :]  # 3xN
        intensity_np = npy_data[3:4, :]  # 1xN
        # sn_np表示的是每个点的向量
        sn_np = npy_data[4:7, :]  # 3xN
        label_np = npy_data[7:, :]

        # print(sn_np.shape)
        # print(label_np.shape)
        # print(label_np)

        # 堆积帧的数量？这玩意不知道干啥的
        if self.opt.accumulation_frame_num <= 0.5:
            # print("come here **************")
            return pc_np, intensity_np, sn_np, label_np

        pc_np_list = [pc_np]
        intensity_np_list = [intensity_np]
        sn_np_list = [sn_np]

        # pose of i  取出第i帧的位姿  好像这里拆出来就是一个简单的4*4矩阵？
        P_oi = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_i))['pose'].astype(np.float32) # 4x4
        # print("success load the pose file **********************")
        # print(P_oi.shape)

        # 看了半天，感觉这玩意。。。像个数据增强
        # search for previous  这个是往前找accumulation_frame_num帧
        prev_pc_np_list, \
        prev_intensity_np_list, \
        prev_sn_np_list = self.search_for_accumulation(pc_folder,
                                                       seq_pose_folder,
                                                       seq_i,
                                                       seq_sample_num,
                                                       Pc,
                                                       P_oi,
                                                       -self.opt.accumulation_frame_skip)
        # search for next   往后找
        next_pc_np_list, \
        next_intensity_np_list, \
        next_sn_np_list = self.search_for_accumulation(pc_folder,
                                                       seq_pose_folder,
                                                       seq_i,
                                                       seq_sample_num,
                                                       Pc,
                                                       P_oi,
                                                       self.opt.accumulation_frame_skip)

        # print("success the work")

        pc_np_list = pc_np_list + prev_pc_np_list + next_pc_np_list
        intensity_np_list = intensity_np_list + prev_intensity_np_list + next_intensity_np_list
        sn_np_list = sn_np_list + prev_sn_np_list + next_sn_np_list

        pc_np = np.concatenate(pc_np_list, axis=1)
        intensity_np = np.concatenate(intensity_np_list, axis=1)
        sn_np = np.concatenate(sn_np_list, axis=1)

        return pc_np, intensity_np, sn_np, label_np



    def __len__(self):
        return len(self.dataset)

    # 这个接口是遍历的时候会用
    def __getitem__(self, index):
        # 记录一下时间
        begin  = time.time()
        # print("are you ok ????")
        # 首先，将前面的这些信息进行一个解析
        # dataset_new.append((pc_nwu_file, image3_file, 'P3', flag))
        # dataset.append((pc_nwu_folder,lidar_index, img2_folder,image2_index, seq, sample_num, 'P2'))
        pc_folder,lidar_index, seimg_folder, img_index , seq, seq_sample_num, img_key, target = self.dataset[index]

        # print("the llidar_index is ", lidar_index, " the img_index is ", img_index, "and the target is ", target)
        # pc_file, img_file,img_key_new, target, seq_sample_num = self.dataset_new[index]

        # 然后去读取一个坐标信息
        seq_pose_folder = os.path.join(self.root, 'poses', '%02d' % seq)

        # print("seq_pose_folder is here !!!!!!!!!!!!!!!!!!!!!!!!!")

        # load point cloud of seq_i
        # 这里感觉是得到了两者之间的变换
        Pc = np.dot(self.calib_helper.get_matrix(seq, img_key),
                    self.calib_helper.get_matrix(seq, 'Tr'))

        # 获取当前点云和加上了数据增强的点云列表
        pc_np, intensity_np, sn_np, label_np = self.get_accumulated_pc(pc_folder, seq_pose_folder, lidar_index, seq_sample_num, Pc)
        pc_np2, intensity_np2, sn_np2, label_np2 = self.get_accumulated_pc(seimg_folder, seq_pose_folder, img_index, seq_sample_num, Pc)

        # 接下来好像是整体的加一些数据增强
        #  ------------- apply random transform on points under the NWU coordinate ------------
        if 'train' == self.mode:
            Pr = self.generate_random_transform(self.opt.P_tx_amplitude, self.opt.P_ty_amplitude, self.opt.P_tz_amplitude,
                                                self.opt.P_Rx_amplitude, self.opt.P_Ry_amplitude, self.opt.P_Rz_amplitude)
            Pr_inv = np.linalg.inv(Pr)

            # print('success create the pr data')

            # -------------- augmentation ----------------------
            pc_np, intensity_np, sn_np = self.augment_pc(pc_np, intensity_np, sn_np)
            pc_np2, intensity_np2, sn_np2 = self.augment_pc(pc_np2, intensity_np2, sn_np2)

        # -------------- convert to torch tensor --------------------- 转换为torch模式
        pc = torch.from_numpy(pc_np)  # 3xN
        pc2 = torch.from_numpy(pc_np2)

        target = torch.from_numpy(np.asarray(int(target)))

        # print(target)

        # end = time.time()
        # print("other time cost: ", end - img_end)
        # print('get one item cost: ',end - begin)

        return pc,pc2,target


if __name__ == '__main__':
    root_path = 'dataset/kitti'
    opt = options_clip.Options()
    kittiloader = PointnetVladKittiLoader(root_path, 'train', opt)

    # print("there is ok")

    for i in range(0, len(kittiloader), 1000):
        print('--- %d ---' % i)
        data = kittiloader[i]
        for item in data:
            print(item.size())
