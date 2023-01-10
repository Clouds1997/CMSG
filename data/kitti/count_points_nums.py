import open3d
import numpy as np
import os
import sys
sys.path.append(os.getcwd())

if __name__ == '__main__':
    input_dir = 'dataset/kitti/data_odometry_velodyne_NWU/sequences'
    # dataset/kitti/data_odometry_velodyne_clip/sequences/00/voxel0.1-SNr0.6
    file_path  = os.path.join(input_dir, '00', 'voxel0.1-SNr0.6','000000.npy')
    data = np.load(file_path)
    print(sum(data[0])/data.shape[1])
    print(sum(data[1])/data.shape[1])
    print(sum(data[2])/data.shape[1])

    print(data.shape)