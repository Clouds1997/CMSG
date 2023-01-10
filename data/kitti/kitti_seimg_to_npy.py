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

index_map = {
    "100,0,142":0,
    "90,100,142":1,
    "0,0,142":2,
    "128,64,128": 3,
    "244,35,232": 3,
    "70,70,70": 5,
    "70,255,70":6,
    "240,142,35": 7,
    "107,142,35": 7,
    "70,240,70": 10,
    "70,70,240": 10,
    "0,0,0": 19
}



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




def get_lable(path):
    img = Image.open(path)
    img_list = np.asarray(img, dtype=np.uint8)
    se_list = np.full((img_list.shape[0],img_list.shape[1], 1), 19)
    # print(img_list.shape)
    rows = img_list.shape[0]
    cols = img_list.shape[1]
    for row in range(rows):
        for col in range(cols):
            # if (item == np.array([107,142,35])).all():
                # print(type(item))
            # if((img_list[row][col] == np.array([100,0,142])).all()):
            #     se_list[row][col] = 0
            #     continue
            # if((img_list[row][col] == np.array([90,100,142])).all()):
            #     se_list[row][col] = 0
            #     continue
            # if((img_list[row][col] == np.array([0,0,142])).all()):
            #     se_list[row][col] = 0
            #     continue
            if((img_list[row][col] == np.array([128,64,128])).all()):
                se_list[row][col] = 3
                continue
            if((img_list[row][col] == np.array([244,35,232])).all()):
                se_list[row][col] = 3
                continue
            if((img_list[row][col] == np.array([70,70,70])).all()):
                se_list[row][col] = 5
                continue
            if((img_list[row][col] == np.array([70,255,70])).all()):
                se_list[row][col] = 6
                continue
            if((img_list[row][col] == np.array([240,142,35])).all()):
                se_list[row][col] = 7
                continue
            if((img_list[row][col] == np.array([107,142,35])).all()):
                se_list[row][col] = 7
                continue
            if((img_list[row][col] == np.array([70,240,70])).all()):
                se_list[row][col] = 10
                continue
            if((img_list[row][col] == np.array([70,70,240])).all()):
                se_list[row][col] = 10
                continue
            if((img_list[row][col] == np.array([153,153,153])).all()):
                se_list[row][col] = 10
                continue
                # print("233333")
    # print(se_list.shape)
    return se_list

def process_kitti(input_root_path,
                  output_root_path,
                  seq_list,
                  img_key):
    for seq in seq_list:
        img_folder = os.path.join(input_root_path, '%02d' % seq,img_key, 'data')
        se_folder = os.path.join(input_root_path, '%02d' % seq,img_key + "_se")
        output_folder = os.path.join(output_root_path, '%02d' % seq,img_key)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        sample_num = round(len(os.listdir(img_folder)))
        print(sample_num)
        for i in range(sample_num):
            print('sequence %d: %d/%d' % (seq, i, sample_num))
            # print(output_folder)
            data_np = read_images(os.path.join(img_folder, '%010d.png' % i))
            se_np = get_lable(os.path.join(se_folder, '%010d.png' % i))

            img =  np.concatenate((data_np, se_np), axis=2).astype(np.uint8)

            # crop the first few rows, original is 370x1226 now
            # print(img.shape)
            img = img[50:, :, :]

            # scale  图像缩放处理
            img = cv2.resize(img,
                        (int(round(img.shape[1] * 0.5)),
                        int(round((img.shape[0] * 0.5)))),
                        interpolation=cv2.INTER_LINEAR)
            # print(img.shape)
            # img = np.concatenate((img_rgb, img_label), axis = 2)
            # print("the resize is success ")
            # print(img.shape)

            # print('data_shape: ', data_np.shape)
            # print('se_np: ',se_np.shape)

            # print("out_np: ", output_np)
            np.save(os.path.join(output_folder, '%06d.npy' % i), img)



if __name__ == '__main__':
    # input_root_path = '2011_10_03/2011_10_03_drive_0027_sync'
    # output_root_path = 'dataset/kitti/data_odometry_color_npy/sequences'

    input_root_path = 'dataset/download/raw'
    output_root_path = 'dataset/kitti/data_odometry_semantic_npy_pre_Robustness/sequences'

    key_list = ['image_02','image_03']

    seq_list = list(range(11))

    thread_num = [0]  # One thread for one folder
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


