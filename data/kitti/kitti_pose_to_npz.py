import numpy as np
import open3d
import sys
import os
from scipy.spatial import cKDTree
import struct
from multiprocessing import Process
import time
import math

sys.path.append(os.getcwd())

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from util import vis_tools
from data.kitti_helper import *


def process_kitti(input_root_path,
                  output_root_path,
                  seq_list):
    for seq in seq_list:
        input_folder = os.path.join(input_root_path, '%02d.txt' % seq)
        output_folder = os.path.join(output_root_path, '%02d' % seq)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        with open(input_folder) as f:
            i = 0
            for line in f.readlines():
                # print(line)
                lines = line.split(" ")
                lines.append(0)
                lines.append(0)
                lines.append(0)
                lines.append(1)
                # print(lines)
                pose = np.asarray(lines).astype(np.float32).reshape((4,4))
                # print(pose)
                # output_np = np.concatenate((pc_down_np, intensity_down_np, pc_down_sn_np), axis=0).astype(np.float32)
                file_name = os.path.join(output_folder, '%06d.npz' % i)
                np.savez(file_name, pose = pose)
                print("success save the ", file_name)
                i = i + 1



if __name__ == '__main__':
    input_root_path = 'dataset/kitti/pose'
    output_root_path = 'dataset/kitti/poses'

    seq_list = list(range(11))

    thread_num = 11  # One thread for one folder
    kitti_threads = []
    for i in range(thread_num):
        thread_seq_list = [i]
        kitti_threads.append(Process(target=process_kitti,
                                     args=(input_root_path,
                                           output_root_path,
                                           thread_seq_list)))

    for thread in kitti_threads:
        thread.start()

    for thread in kitti_threads:
        thread.join()


