import open3d
import time
import copy
import numpy as np
import math
import os
import shutil
import torch
import sys
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from sklearn import metrics

import matplotlib
matplotlib.use('Agg')

sys.path.append(os.getcwd())

# print(os.getcwd())

from models.multimodal_classifier import MMClassiferSe, MMClassiferSeDime,Classifergraph,MMClassiferBase
from data.kitti_semantic_pc_img_pose_loader import SeKittiLoader
from kitti import options
from kitti import options_clip


def main():
    # opt = options.Options()
    opt = options_clip.Options()
    logdir = './runs/'+str(opt.version)
    if os.path.isdir(logdir):
        shutil.rmtree(logdir)
        # user_answer = input("The log directory %s exists, do you want to delete it? (y or n) : " % logdir)
        # if user_answer == 'y':
        #     # delete log folder
        #     shutil.rmtree(logdir)
        # else:
        #     exit()
    else:
        os.makedirs(logdir)
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=logdir)

    # 载入数据集

    testset = SeKittiLoader(opt.dataroot, 'val', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=opt.dataloader_threads, pin_memory=True)
    print('#testing point clouds = %d' % len(testset))

    # print("the SeKittiLoader is ok")
    # create model, optionally load pre-trained model
    # 创建模型
    # if opt.is_fine_resolution:
    #     model = MMClassifer(opt, writer)
    # else:
    #     model = MMClassiferCoarse(opt, writer)
    # model = Classifergraph(opt, writer)
    model = MMClassiferBase(opt, writer)

    # print(os.path.join(opt.checkpoints_dir,'best.pth'))

    # 加载模型
    model.load_model(os.path.join(opt.checkpoints_dir,'best.pth'))

    print('success load best model')

    sequence = '00'


    pred_db = []
    gt_db = []
        # for i, data in enumerate(testloader):
    for i, data in tqdm(enumerate(testloader), total=len(testloader), desc="testbatch"):
        if(i == len(testloader) - 1):
            break
        # print(len(testloader))
        pc, intensity, sn, label, node_a, node_b, pc_graph, img_graph,\
        P, img, K, t_ij, target = data
        B = pc.size()[0]

        # 无图网络版本
        # model.set_input(pc, intensity, sn, label, node_a, node_b,
        #     P, img, K, target)

        # 图网络版本
        model.set_input(pc, intensity, sn, label, node_a, node_b,
                        P, img, K, target)
        # print("the ", i ," is begin")
        model.test_model()
        # print("the ", i ," is ok")
        _, test_loss_dict = model.get_current_errors()
        pred_b,gt_b = model.get_current_batch()

        # losses += test_loss_dict['loss']
        pred_db.extend(pred_b)
        gt_db.extend(gt_b)

        # test_batch_sum += B
        # test_loss_sum['loss'] += B*test_loss_dict['loss']

    pred_db = np.array(pred_db)
    gt_db = np.array(gt_db)
    # save results

    # print("all the batch is ok")
    # save results
    gt_db_path = os.path.join(opt.output_path,sequence + "_gt_db.npy")
    pred_db_path = os.path.join(opt.output_path,sequence + "_DL_db.npy")
    np.save(gt_db_path, gt_db)
    np.save(pred_db_path, pred_db)
    #####ROC
    fpr, tpr, roc_thresholds = metrics.roc_curve(gt_db, pred_db)
    roc_auc = metrics.auc(fpr, tpr)
    print("fpr: ", fpr)
    print("tpr: ", tpr)
    print("thresholds: ", roc_thresholds)
    print("roc_auc: ", roc_auc)

    # plot ROC Curve
    plt.figure(0)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('DL ROC Curve')
    plt.legend(loc="lower right")
    roc_out = os.path.join(opt.output_path, sequence + "_DL_roc_curve.png")
    plt.savefig(roc_out)


    precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)
    # plot p-r curve
    plt.figure(1)
    lw = 2
    plt.plot(recall, precision, color='darkorange',
                lw=lw, label='P-R curve') 
    plt.axis([0,1,0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('DL Precision-Recall Curve')
    plt.legend(loc="lower right")
    pr_out = os.path.join(opt.output_path, sequence + "_DL_pr_curve.png")
    plt.savefig(pr_out)





if __name__ == "__main__":
    main()