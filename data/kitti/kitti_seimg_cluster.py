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
import copy

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

def get_lable(img):
    # img = Image.open(path)
    img_list = np.asarray(img, dtype=np.uint8)
    se_list = np.full((img_list.shape[0],img_list.shape[1], 1), 19)
    # print(img_list.shape)
    rows = img_list.shape[0]
    cols = img_list.shape[1]
    for row in range(rows):
        for col in range(cols):
            # if (item == np.array([107,142,35])).all():
                # print(type(item))
            if((img_list[row][col] == np.array([100,0,142])).all()):
                #car
                se_list[row][col] = 0
                continue
            if((img_list[row][col] == np.array([90,100,142])).all()):
                #car
                se_list[row][col] = 0
                continue
            if((img_list[row][col] == np.array([0,0,142])).all()):
                #还是car
                se_list[row][col] = 0
                continue
            if((img_list[row][col] == np.array([128,64,128])).all()):
                #side
                se_list[row][col] = 3
                continue
            if((img_list[row][col] == np.array([244,35,232])).all()):
                #side
                se_list[row][col] = 3
                continue
            if((img_list[row][col] == np.array([70,70,70])).all()):
                #
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

def Seed_Filling(depth_np,label_np,mask_np,lab):
    # print(lab)
    depth_clone = np.pad(depth_np,(1,1))
    # label_clone = np.pad(label_np,(1,1))
    mask_clone = np.pad(mask_np,(1,1))
    label = 20
    centers = []
    rows = depth_clone.shape[0] - 1
    cols = depth_clone.shape[1] - 1
    radius = 0.2
    thread = 500
    if lab == 3:
        # print("comming here")
        radius = 1
    if lab == 5:
        radius = 1
    # if lab == 7:
    #     radius = 0.25
    for row in range(1,rows -1):
        for col in range(1, cols - 1):
            neighborPixels = []
            label = label + 1
            if label == 255:
                label = label + 1
            count = 0
            sumX = 0
            sumY = 0
            if mask_clone[row][col] == 255:
                neighborPixels.append([row,col])
                while len(neighborPixels):
                    cur_x,cur_y = neighborPixels.pop()
                    mask_clone[cur_x][cur_y] = label
                    dep0 = depth_clone[cur_x][cur_y]
                    dep1 = depth_clone[cur_x][cur_y - 1]
                    dep2 = depth_clone[cur_x][cur_y + 1]
                    dep3 = depth_clone[cur_x - 1][cur_y]
                    dep4 = depth_clone[cur_x + 1][cur_y]
                    if mask_clone[cur_x][cur_y - 1] == 255 and (abs(dep0 - dep1) < radius):
                        neighborPixels.append([cur_x,cur_y - 1])
                    if mask_clone[cur_x][cur_y + 1] == 255 and (abs(dep0 - dep2) < radius):
                        neighborPixels.append([cur_x,cur_y + 1])
                    if mask_clone[cur_x - 1][cur_y] == 255 and (abs(dep0 - dep3) < radius):
                        neighborPixels.append([cur_x - 1,cur_y])
                    if mask_clone[cur_x + 1][cur_y] == 255 and (abs(dep0 - dep4) < radius):
                        neighborPixels.append([cur_x + 1,cur_y])
                    count = count + 1
                    sumX = sumX + cur_x
                    sumY = sumY + cur_y
                    # print(len(neighborPixels)）
            if count > thread:
                averX = sumX / count
                averY = sumY / count
                centers.append([averX,averY])
    # print(centers)

    return centers

def pointExtraction(data_np, se_np, depth_np, label_np):
    rows = data_np.shape[0]
    cols = data_np.shape[1]

    label_rem = []

    for lab in [0,1,2,3,5,6,7,10]:
        tmp = copy.deepcopy(se_np)
        img = np.squeeze(tmp)
        img = np.uint8(img)
        mask_np = np.zeros((rows,cols))
        any_pix = False
        centers = []
        centers.clear()
        for i in range(data_np.shape[0]):
            for j in range(data_np.shape[1]):
                if  label_np[i][j] == lab:
                    mask_np[i][j] = 255
                    any_pix = True
        if any_pix:
            # print("comming here ")
            centers = Seed_Filling(depth_np, label_np, mask_np, lab)

        for center in centers:
            label_rem.append(lab)
        #     cv2.circle(img,(int(center[1]),int(center[0])),5,(0,255,0),-1)
        # cv2.imwrite("/home/liuhy/workspace/DeepI2P-main/dataset/download/test/mask"+str(lab)+".png",mask_np)
        # cv2.imwrite("/home/liuhy/workspace/DeepI2P-main/dataset/download/test/lab"+str(lab)+".png",img)
        # tmp = np.empty(tmp.shape)
    # print(label_rem)
    labels = np.asarray(label_rem)
    # np.save(os.path.join(output_folder, '%010d.npy' % i),labels)
    return labels

    # return 0

def process_kitti(input_root_path,
                  output_root_path,
                  seq_list,
                  img_key):
    for seq in seq_list:
        img_folder = os.path.join(input_root_path, '%02d' % seq,img_key, 'data')
        se_folder = os.path.join(input_root_path, '%02d' % seq,img_key + "_se")
        depth_folder = os.path.join(input_root_path, '%02d' % seq,"pred_txt")
        output_folder = os.path.join(output_root_path, '%02d' % seq,img_key)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        sample_num = round(len(os.listdir(img_folder)))
        print(sample_num)
        for i in range(sample_num):
            print('sequence %d: %d/%d' % (seq, i, sample_num))
            # print(output_folder)
            data_np = read_images(os.path.join(img_folder, '%010d.png' % i))
            se_np = read_images(os.path.join(se_folder, '%010d.png' % i))
            depth_np = np.loadtxt(os.path.join(depth_folder, '%010d.txt' % i))
            label_np = get_lable(se_np)

            labels = pointExtraction(data_np, se_np, depth_np, label_np)

            np.save(os.path.join(output_folder,'%06d.npy' % i), labels)





if __name__ == '__main__':
    # input_root_path = '2011_10_03/2011_10_03_drive_0027_sync'
    # output_root_path = 'dataset/kitti/data_odometry_color_npy/sequences'

    input_root_path = 'dataset/download/raw'
    output_root_path = 'dataset/kitti/data_odometry_semantic_graph/sequences'

    key_list = ['image_02','image_03']

    seq_list = list(range(11))

    # thread_num = [3,8]  # One thread for one folder
    thread_num = [0,1,2,4,5,6,7,9,10]
    kitti_threads = []
    for i in thread_num:
        thread_seq_list = [i]
        kitti_threads.append(Process(target=process_kitti,
                                     args=(input_root_path,
                                           output_root_path,
                                           thread_seq_list,
                                           key_list[0])))
    for thread in kitti_threads:
        thread.start()

    for thread in kitti_threads:
        thread.join()


