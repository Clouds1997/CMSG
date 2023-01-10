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
from sklearn import metrics

import matplotlib
import argparse
matplotlib.use('Agg')

sys.path.append(os.getcwd())

# print(os.getcwd())

from models.multimodal_classifier import MMClassifer, MMClassiferCoarse
from data.kitti_pc_img_pose_loader import KittiLoader
from kitti import options
from kitti import options_clip
from kitti.model_test import ACE

# kitti的入口
if __name__=='__main__':
        # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--logger_name', default='runs/runX', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--data_path', default='/data', help='path to datasets')
    parser.add_argument('--data_name', default='precomp', help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument("--bert_path", default='./', type=str, help="The BERT model path.")
    parser.add_argument('--max_words', default=32, type=int, help='maximum number of words in a sentence.')
    parser.add_argument('--extra_stc', type=int, default=0, help='Sample (extra_stc * bs) extra sentences.')
    parser.add_argument('--extra_img', type=int, default=0, help='Sample (extra_stc * bs) extra images.')
    # Optimization
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--learning_rate', default=.0002, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int, help='Number of epochs to update the learning rate.')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_true', default=True, help='Use max instead of sum in the rank loss.')
    parser.add_argument('--trade_off', default=0.5, type=float, help='Trade-off parameter for path regularization.')
    # Base
    parser.add_argument('--img_dim', default=2048, type=int, help='Dimensionality of the image embedding.')
    # parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input.')
    parser.add_argument('--embed_size', default=256, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--direction', type=str, default='i2t',help='Version of model, i2t | t2i')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=2000, type=int, help='Number of steps to run validation.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--finetune', action='store_true', help='Fine-tune the image encoder.')
    # parser.add_argument('--cnn_type', default='vgg19', help="""The CNN used for image encoder(e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true', help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine', help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true', help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true', help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true', help='Ensure the training is always done in train mode (Not recommended).')
    # DIME
    parser.add_argument('--num_head_IMRC', type=int, default=16, help='Number of heads in Intra-Modal Reasoning Cell')
    parser.add_argument('--hid_IMRC', type=int, default=512, help='Hidden size of FeedForward in Intra-Modal Reasoning Cell')
    parser.add_argument('--raw_feature_norm_CMRC', default="clipped_l2norm", help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--lambda_softmax_CMRC', default=4., type=float, help='Attention softmax temperature.')
    parser.add_argument('--hid_router', type=int, default=512, help='Hidden size of MLP in routers')
    opt_net = parser.parse_args()
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
    # 这里在枚举的时候有点问题 ***********
    trainset = KittiLoader(opt.dataroot, 'train', opt)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.dataloader_threads, drop_last=True, pin_memory=True)
    print('#training point clouds = %d' % len(trainset))

    testset = KittiLoader(opt.dataroot, 'val', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=opt.dataloader_threads, pin_memory=True)
    print('#testing point clouds = %d' % len(testset))

    # create model, optionally load pre-trained model
    # 创建模型
    model = ACE(opt_net)
    # model.load_model('/ssd/jiaxin/point-img-feature/kitti/save/1.3-odometry/checkpoints/best.pth')

    # print("the models load is okk ***************************")

    best_test_accuracy = 0
    epochs = trange(opt.epochs, leave=True, desc="Epoch")
    for epoch in epochs:
        epoch_iter = 0
        loss_sum = 0
        main_index = 0
        # for i, data in enumerate(trainloader):
        model.train_start()
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader), desc="Batches"):
            model.train_start()
            model.train_emb(epoch, data)
            





