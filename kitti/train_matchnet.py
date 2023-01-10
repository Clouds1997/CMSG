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
matplotlib.use('Agg')

sys.path.append(os.getcwd())

# print(os.getcwd())
from models.PointNetVlad import PointNetVlad
from models.multimodal_classifier import MMClassiferSe, MMClassiferSeDime,Classifergraph,MMClassiferBase,PointvladBase,MatchBase
from data.kitti_semantic_pc_img_pose_loader import SeKittiLoader, PointnetVladKittiLoader
from kitti import options
from kitti import options_matchnet

# kitti的入口
if __name__=='__main__':
    # opt = options.Options()
    opt = options_matchnet.Options()
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
    trainset = SeKittiLoader(opt.dataroot, 'train', opt)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.dataloader_threads, drop_last=True, pin_memory=True)
    print('#training point clouds = %d' % len(trainset))

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
    model = MatchBase(opt, writer)
    # model = MMClassiferSe(opt, writer)
    # model = MMClassiferBase(opt, writer)
    # model.load_model('/ssd/jiaxin/point-img-feature/kitti/save/1.3-odometry/checkpoints/best.pth')

    # print("the models load is okk ***************************")

    best_test_accuracy = 0
    epochs = trange(opt.epochs, leave=True, desc="Epoch")
    for epoch in epochs:
        epoch_iter = 0
        loss_sum = 0
        main_index = 0
        # epoch_begin = time.time()
        # for i, data in enumerate(trainloader):
        start = time.time()
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader), desc="Batches"):
            cur_time = time.time()
            # print("dataloader cost: ", cur_time - start)
            # 这里各个参数表示什么需要打印出来查看
            # print("there is ok")
            # time1 = time.time()
            pc, intensity, sn, label, node_a, node_b, pc_graph, img_graph, \
            P, img, K, t_ij, target = data
            # time2 = time.time()
            # print("time2 - time1 = ", time2- time1)
            # print("the img shape is ", img.shape)
            B = pc.size()[0]

            iter_start_time = time.time()
            epoch_iter += B
            model.global_step_inc(B)
            time3 = time.time()
            # print("time3 - time2 = ", time3 - time2)
            # 无图网络版本
            model.set_input(pc, intensity, sn, label, node_a, node_b,
                            P, img, K, target)

            # 图网络版本
            # model.set_input(pc, pc2, target)
            # time4 = time.time()
            # print("time4 - time3: =", time4 - time3)
            # print("the model data is load ok ***************")
    #         # 前面数据load部分已经处理完毕，现在开始分析特征
            model.optimize()
            # time5 = time.time()
            # print("time5 -time4: ", time5 - time4)
            # print("the optimize once is okk *****************")
            train_loss_dict, _ = model.get_current_errors()
            main_index = main_index + len(trainloader)
            loss_sum = loss_sum + train_loss_dict['loss'] * len(trainloader)
            loss = loss_sum / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

            if i % int(800) == 0 and i > 0:
                # print/plot errors
                t = (time.time() - iter_start_time) / opt.batch_size
                train_loss_dict, test_loss_dict = model.get_current_errors()
                # train_accuracy_dict, test_accuracy_dict = model.get_current_accuracy()
                # model.print_loss_dict(train_loss_dict, train_accuracy_dict, t)

                model.write_loss()

        # epoch done
        test_start_time = time.time()
        test_batch_sum = 0
        test_loss_sum = {'loss': 0}
        losses = 0
        pred_db = []
        gt_db = []
        # for i, data in enumerate(testloader):
        for i, data in tqdm(enumerate(testloader), total=len(testloader), desc="testbatch"):
            if(i == len(testloader) - 1):
                break
            # print(len(testloader))
            pc, intensity, sn, label, node_a, node_b, pc_graph, img_graph, \
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

            test_batch_sum += B
            test_loss_sum['loss'] += B*test_loss_dict['loss']

        print("all the batch is ok")

        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)
        print("the precision is: ",precision," and the recall is", recall)
        # calc F1-score
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        F1_max_score = np.max(F1_score)
        print(" F1_max_score: " + str(F1_max_score) + ".")
        # model_loss = losses / len(testloader)
        # print(" loss: " + str(model_loss) + ".")


        test_loss_sum['loss'] /= test_batch_sum
        test_persample_time = (time.time() - test_start_time) / test_batch_sum
        output = 'Per sample time: %.4f - ' % (test_persample_time)
        for key, value in test_loss_sum.items():
            output += '%s: %.4f, ' % (key, value)
        print(output)

        # print('Test loss and accuracy:')
        # model.print_loss_dict(test_loss_sum, test_accuracy_sum, test_persample_time)
        # set the mean loss/accuracy to the model, so that the tensorboard visualization is correct
        model.test_loss_dict = test_loss_sum
        # model.test_accuracy = test_accuracy_sum

        # record best test loss
        # if test_accuracy_sum['class_accuracy'] > best_test_accuracy:
        #     best_test_accuracy = test_accuracy_sum['coarse_accuracy']
        #     print('--- best test coarse accuracy %f' % best_test_accuracy)

        print('Epoch %d done.' % epoch)

        if epoch % opt.lr_decay_step==0 and epoch > 0:
            model.update_learning_rate(opt.lr_decay_scale)

        # save network
        if epoch >= 0:
            print("Saving network...")
            model.save_network(model.detector, "v%s-gpu%d-epoch%d-%f.pth" % (opt.version,
                                                                    opt.gpu_ids[0],
                                                                    epoch, F1_max_score))





