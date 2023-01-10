import numpy as np
import open3d
import os
import sys
from scipy.spatial import cKDTree
import struct
from multiprocessing import Process
import time
import math

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from util import vis_tools
from data.kitti_helper import *

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
            pc_list.append(np.array([x,y,z,i,lable]))
         # points = list(map(lambda line: list(map(lambda x: float(x), line.split(' '))), lines))

    return np.asarray(pc_list, dtype=np.float32).T

# 这里是投影rangeimage的地方
def range_projection(current_vertex, fov_up=10.67, fov_down=-30.67, proj_H=32, proj_W=900, max_range=50, cut_z = True, low=0.1, high=6):
    """ Project a pointcloud into a spherical projection, range image.
        Args:
        current_vertex: raw point clouds
        Returns: 
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)

    if cut_z:
        z = current_vertex[:, 2]
        current_vertex = current_vertex[(depth > 0) & (depth < max_range) & (z < high) & (z > low)]  # get rid of [0, 0, 0] points
        depth = depth[(depth > 0) & (depth < max_range) & (z < high) & (z > low)]
    else:
        current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
        depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                        dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                        dtype=np.int32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices

    return proj_range, proj_vertex, proj_idx


def process_kitti(input_root_path,
                  output_root_path,
                  seq_list,
                  downsample_voxel_size,
                  sn_radius,
                  sn_max_nn):
    for seq in seq_list:
        input_folder = os.path.join(input_root_path,'%02d' % seq, 'semantic_lidar')
        output_folder = os.path.join(output_root_path, '%02d' % seq, 'voxel%.1f-SNr%.1f' % (downsample_voxel_size, sn_radius))
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        sample_num = round(len(os.listdir(input_folder)))
        for i in range(sample_num):
            # show progress
            print('sequence %d: %d/%d' % (seq, i, sample_num))

            data_np = read_pcd(os.path.join(input_folder, '%010d.pcd' % i))
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

            # save downampled points, intensity, surface normal to npy
            output_np = np.concatenate((pc_down_np, intensity_down_np, pc_down_sn_np, label_down_np), axis=0).astype(np.float32)
            np.save(os.path.join(output_folder, '%06d.npy' % i), output_np)

            # debug
            # vis_tools.plot_pc(pc_down_np, size=1, color=intensity_down_np[0, :])
            # plt.show()
            # break


if __name__ == '__main__':
    input_root_path = 'dataset/download/raw'
    output_root_path = 'dataset/kitti/data_odometry_semantic_velodyne/sequences'
    downsample_voxel_size = 0.1
    sn_radius = 0.6
    sn_max_nn = 30
    seq_list = list(range(11))

    thread_num = [1]  # One thread for one folder
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


