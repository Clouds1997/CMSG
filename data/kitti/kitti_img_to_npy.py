import numpy as np
import open3d
import os
import sys
from scipy.spatial import cKDTree
import struct
from multiprocessing import Process
import time
import math
import cv2
from PIL import Image

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from util import vis_tools
from data.kitti_helper import *

# kitti数据的预处理模块
def read_images(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*4
    '''
    # img = cv2.imread(path)
    img = Image.open(path)
    img_list = np.asarray(img, dtype=np.uint8)
    # print(img_list)
    # pc_list = []
    # with open(path, 'rb') as f:
    #     content = f.read()
    #     pc_iter = struct.iter_unpack('ffff', content)
    #     print(pc_iter)
        # for idx, point in enumerate(pc_iter):
        #     pc_list.append([point[0], point[1], point[2], point[3]])
    return img_list

def process_kitti(input_root_path,
                  output_root_path,
                  seq_list,
                  img_key):
    for seq in seq_list:
        input_folder = os.path.join(input_root_path, '%02d' % seq,img_key,'data')
        output_folder = os.path.join(output_root_path, '%02d' % seq,img_key[:-2] + img_key[-1])
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        sample_num = round(len(os.listdir(input_folder)))
        print(sample_num)
        for i in range(sample_num):
            print('sequence %d: %d/%d' % (seq, i, sample_num))
            # print(output_folder)
            data_np = read_images(os.path.join(input_folder, '%06d.png' % i))
            # print(data_np.shape)
            np.save(os.path.join(output_folder, '%06d.npy' % i), data_np)



if __name__ == '__main__':
    # input_root_path = '2011_10_03/2011_10_03_drive_0027_sync'
    # output_root_path = 'dataset/kitti/data_odometry_color_npy/sequences'

    input_root_path = 'dataset/download/raw'
    output_root_path = 'dataset/kitti/data_odometry_color_npy/sequences'

    key_list = ['image_02','image_03']

    seq_list = list(range(11))

    thread_num = [3]  # One thread for one folder
    kitti_threads = []
    for i in thread_num:
        thread_seq_list = [i]
        kitti_threads.append(Process(target=process_kitti,
                                     args=(input_root_path,
                                           output_root_path,
                                           thread_seq_list,
                                           key_list[0])))
        kitti_threads.append(Process(target=process_kitti,
                                     args=(input_root_path,
                                           output_root_path,
                                           thread_seq_list,
                                           key_list[1])))

    for thread in kitti_threads:
        thread.start()

    for thread in kitti_threads:
        thread.join()


