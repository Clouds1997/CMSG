import numpy as np
import open3d
import os
import sys
from scipy.spatial import cKDTree
import struct
from multiprocessing import Process
import time
import math
import json

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from util import vis_tools
from data.kitti_helper import *

input_pt_num = 5120

# kitti数据的预处理模块
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*4
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content) #读取二进制文件
        for idx, point in enumerate(pc_iter):
            if point[0] >= 5 and point[0] < 30 and point[1] >= -14 and point[1] < 14:
                # print(point[0],' ', point[1],' ',  point[2],' ',  point[3],' ',  point[4])
                pc_list.append([point[0], point[1], point[2], point[3]])
    return np.asarray(pc_list, dtype=np.float32).T

# ASCII
def read_pcd(path):
    pc_list = []
    with open(path, 'r') as f:
        lines = f.readlines()[11:]
        for line in lines:
            line = list(line.strip('\n').split(' '))
            x = float(line[0])
            y = float(line[1])
            z = float(line[2])
            i = float(line[3])
            #  r = math.sqrt(x**2 + y**2 + z**2) * i
            lable = float(line[4])
            # print(x,' ', y,' ', z,' ', i,' ',lable)
            if x >= 5 and x < 30 and  y >= -14 and  y < 14:
                pc_list.append(np.array([x,y,z,i,lable]))
         # points = list(map(lambda line: list(map(lambda x: float(x), line.split(' '))), lines))

    return np.asarray(pc_list, dtype=np.float32).T

def downsample_with_intensity_sn(pointcloud, intensity, sn, label, voxel_grid_downsample_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
    intensity_max = np.max(intensity)

    fake_colors = np.zeros((pointcloud.shape[1], 3))
    fake_colors[:, 0:1] = np.transpose(intensity) / intensity_max
    fake_colors[:, 1:2] = np.transpose(label) / 10

    pcd.colors = open3d.utility.Vector3dVector(fake_colors)
    pcd.normals = open3d.utility.Vector3dVector(np.transpose(sn))

    down_pcd = open3d.geometry.voxel_down_sample(pcd,voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = down_pcd_points

    intensity = np.transpose(np.asarray(down_pcd.colors)[:, 0:1]) * intensity_max
    label = np.transpose(np.asarray(down_pcd.colors)[:, 1:2]) * 10
    sn = np.transpose(np.asarray(down_pcd.normals))

    # print("the func points size is : ", pointcloud.shape)

    return pointcloud, intensity, sn, label

def downsample_np(pc_np, intensity_np, sn_np, label_np):
    if pc_np.shape[1] >= input_pt_num:
        choice_idx = np.random.choice(pc_np.shape[1], input_pt_num, replace=False)
    else:
        fix_idx = np.asarray(range(pc_np.shape[1]))
        while pc_np.shape[1] + fix_idx.shape[0] < input_pt_num:
            fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
        random_idx = np.random.choice(pc_np.shape[1], input_pt_num - fix_idx.shape[0], replace=False)
        choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
    pc_np = pc_np[:, choice_idx]
    intensity_np = intensity_np[:, choice_idx]
    sn_np = sn_np[:, choice_idx]
    label_np = label_np[:, choice_idx]

    return pc_np, intensity_np, sn_np, label_np

def process_kitti(input_root_path,
                  output_root_path,
                  seq_list,
                  downsample_voxel_size,
                  sn_radius,
                  sn_max_nn):
    for seq in seq_list:
        input_folder = os.path.join(input_root_path,'%02d' % seq, 'lidar_gn_clip')
        output_folder = os.path.join(output_root_path, '%02d' % seq, 'lidar_gn')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        sample_num = round(len(os.listdir(input_folder)))
        for i in range(sample_num):
            # show progress
            print('sequence %d: %d/%d' % (seq, i, sample_num))

            data_np = read_pcd(os.path.join(input_folder, '%06d.pcd' % i))
            pc_np = data_np[0:3, :]
            intensity_np = data_np[3:4, :]
            lable_np = data_np[4:, :]

            # convert to Open3D point cloud datastructure
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(pc_np.T)
            downpcd = open3d.geometry.voxel_down_sample(pcd, voxel_size=downsample_voxel_size)

            # surface normal computation
            open3d.geometry.estimate_normals(downpcd,
                                             search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=sn_radius,
                                                                                                  max_nn=sn_max_nn))
            open3d.geometry.orient_normals_to_align_with_direction(downpcd, [0,0,1])
            # open3d.visualization.draw_geometries([downpcd])

            # get numpy array from pcd
            pc_down_np = np.asarray(downpcd.points).T
            pc_down_sn_np = np.asarray(downpcd.normals).T

            # get intensity through 1-NN between downsampled pc and original pc
            kdtree = cKDTree(pc_np.T)
            D, I = kdtree.query(pc_down_np.T, k=1)
            intensity_down_np = intensity_np[:, I]
            label_down_np = lable_np[:, I]
            # print("the points size is: ", pc_down_np.shape)
            # print("the intensity_down_np size is: ", intensity_down_np.shape)
            # print("the pc_down_sn_np size is: ", pc_down_sn_np.shape)
            # print("the label_down_np size is: ", label_down_np.shape)

            # 如果点云太多，就使用滤波操作
            if pc_down_np.shape[1] > 2 * input_pt_num:
                # point cloud too huge, voxel grid downsample first
                pc_down_np, intensity_down_np, pc_down_sn_np, label_down_np = downsample_with_intensity_sn(pc_down_np, intensity_down_np,
                                                                pc_down_sn_np, label_down_np,
                                                                voxel_grid_downsample_size=0.3)
                pc_down_np = pc_down_np.astype(np.float32)
                intensity_down_np = intensity_down_np.astype(np.float32)
                pc_down_sn_np = pc_down_sn_np.astype(np.float32)
                label_down_np = label_down_np.astype(np.float32)
                # random downsample to a specific shape, pc is still in NWU coordinate

            # print("the points size is: ", pc_down_np.shape)
            # print("the intensity_down_np size is: ", intensity_down_np.shape)
            # print("the pc_down_sn_np size is: ", pc_down_sn_np.shape)
            # print("the label_down_np size is: ", label_down_np.shape)

            # 这里使点云的数量获得统一
            pc_down_np, intensity_down_np, pc_down_sn_np, label_down_np = downsample_np(pc_down_np, intensity_down_np,
                                                                            pc_down_sn_np, label_down_np)

            # print("the points size is: ", pc_down_np.shape)
            # print("the intensity_down_np size is: ", intensity_down_np.shape)
            # print("the pc_down_sn_np size is: ", pc_down_sn_np.shape)
            # print("the label_down_np size is: ", label_down_np.shape)

            # save downampled points, intensity, surface normal to npy
            output_np = np.concatenate((pc_down_np, intensity_down_np, pc_down_sn_np, label_down_np), axis=0).astype(np.float32)
            np.save(os.path.join(output_folder, '%06d.npy' % i), output_np)

            # debug
            # vis_tools.plot_pc(pc_down_np, size=1, color=intensity_down_np[0, :])
            # plt.show()
            # break


if __name__ == '__main__':
    input_root_path = 'dataset/download/graph'
    output_root_path = 'dataset/kitti/data_odometry_graph/sequences'
    downsample_voxel_size = 0.1
    sn_radius = 0.6
    sn_max_nn = 30
    seq_list = list(range(11))

    thread_num = [3,8]  # One thread for one folder
    kitti_threads = []
    for i in thread_num:
        thread_seq_list = [i]
        kitti_threads.append(Process(target=process_kitti,
                                     args=(input_root_path,
                                           output_root_path,
                                           thread_seq_list,
                                           downsample_voxel_size,
                                           sn_radius,
                                           sn_max_nn)))

    for thread in kitti_threads:
        thread.start()

    for thread in kitti_threads:
        thread.join()


