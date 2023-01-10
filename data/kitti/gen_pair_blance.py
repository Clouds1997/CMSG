import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import argparse
import os
# import json
# import pykitti
import random
from tqdm import tqdm

'''
input: graph_dir, random drop out negative samples
output: graph_pairs
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', '-d', type=str, required=False, default="dataset/kitti", help='Dataset path.')
    parser.add_argument('--output_dir', '-o', type=str, required=False, default="dataset/kitti/pair_list_1_20", help='Output path.')
    parser.add_argument('--pos_thre', type=int, required=False, default=1, help='Positive threshold.')
    parser.add_argument('--neg_thre', type=int, required=False, default=20, help='Negative threshold.')
    args, unparsed = parser.parse_known_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sequences = ["07","09","10"]
    # sequences = ["01"]

    for sq in sequences[:]:
        print("*" * 80)
        seqstr = '{0:02d}'.format(int(sq))
        print("parsing seq {}".format(sq))

        # dataset/kitti/data_odometry_color_npy/sequences/00/image_2
        lidar_dir = os.path.join(args.dataset_dir,'data_odometry_velodyne_clip','sequences', sq,'voxel0.1-SNr0.6')
        img2_dir = os.path.join(args.dataset_dir,'data_odometry_color_npy','sequences', sq,'image_2')
        img3_dir = os.path.join(args.dataset_dir,'data_odometry_color_npy','sequences', sq,'image_3')
        pose_dir = os.path.join(args.dataset_dir,'poses',sq)
        lidar_list = os.listdir(lidar_dir)
        lidar_list.sort()
        img2_list = os.listdir(img2_dir)
        img2_list.sort()
        pose_list = os.listdir(pose_dir)
        pose_list.sort()
        output_dir_sq = os.path.join(args.output_dir, sq)
        if not os.path.exists(output_dir_sq):
            os.makedirs(output_dir_sq)

        sum_true = 0
        sum_false = 0

        for i in tqdm(range(len(pose_list))):
            lidar_P = np.load(os.path.join(pose_dir, '%06d.npz' % i))['pose'].astype(np.float32)  # 4x4
            lidar_t = lidar_P[0:3, 3]  # 3

            count = 0
            for j in range(len(pose_list)):
                img_P = np.load(os.path.join(pose_dir, '%06d.npz' % j))['pose'].astype(np.float32)  # 4x4
                img_t = img_P[0:3, 3]  # 3
                dist = np.linalg.norm(lidar_t - img_t)

                choose_prob = False
                flag = 1

                # sampling strategy
                if dist >= args.pos_thre and dist <= args.neg_thre:
                    continue

                if dist <= args.pos_thre: # dist < 3m, choose prob = 1
                    count = count + 1
                    choose_prob = True
                    list_file = os.path.join(output_dir_sq,sq + '.txt')
                    with open(list_file, 'a')as f:
                        f.write(str(i) + " " + str(j) + " " + str(flag) + '\n')

            neg_count = 0
            for j in range(len(pose_list)):
                img_P = np.load(os.path.join(pose_dir, '%06d.npz' % j))['pose'].astype(np.float32)  # 4x4
                img_t = img_P[0:3, 3]  # 3
                dist = np.linalg.norm(lidar_t - img_t)

                choose_prob = False
                flag = 1

                # sampling strategy
                if dist >= args.pos_thre and dist <= args.neg_thre:
                    continue

                rate = count / len(pose_list)
                # print(rate)

                if dist > args.neg_thre: # dist < 3m, choose prob = 1
                    if random.random() <= rate * 1.07:
                        flag = 0
                        choose_prob = True
                        neg_count = neg_count + 1
                    else:
                        choose_prob = False

                if choose_prob == True:
                    # lidar_file = os.path.join(lidar_dir,'%06d.npy' % i)
                    # img2_file = os.path.join(img2_dir,'%06d.npy' % j)
                    # img3_file = os.path.join(img3_dir,'%06d.npy' % j)
                    list_file = os.path.join(output_dir_sq,sq + '.txt')
                    with open(list_file, 'a')as f:
                        f.write(str(i) + " " + str(j) + " " + str(flag) + '\n')
                    # f.write(lidar_file + " " + img3_file + " " + str(flag) + '\n')
            # print(count, " : ", neg_count)
            sum_true = sum_true + count
            sum_false = sum_false + neg_count
        print(sum_true, " : ", sum_false)



