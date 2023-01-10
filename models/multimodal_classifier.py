import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
from collections import OrderedDict
import os
import random
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
sys.path.append(os.getcwd())

from models import networks_pc
from models import networks_img
from models import networks_united
from models import dime_united
from models import PointNetVlad
from models import matchnet
from models import losses
from data import augmentation
from models import operations
from kitti.options import Options
from util import pytorch_helper
from util import vis_tools
from models import focal_loss
# from loss import ContrastiveLoss
import time

class Classifergraph():
    def __init__(self, opt: Options, writer):
        self.opt = opt
        self.writer = writer
        self.global_step = 0

        # 这个东西是一个关键点的检测？
        self.detector = networks_united.KeypointDetectorG(self.opt).to(self.opt.device)
        # self.coarse_ce_criteria = nn.CrossEntropyLoss()
        self.coarse_ce_criteria = focal_loss.FocalLoss(alpha=0.5, gamma=2, reduction='mean')
        self.fine_ce_criteria = nn.CrossEntropyLoss()

        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.detector = nn.DataParallel(self.detector, device_ids=opt.gpu_ids)
            # self.coarse_ce_criteria = nn.DataParallel(self.coarse_ce_criteria, device_ids=opt.gpu_ids)
            # self.fine_ce_criteria = nn.DataParallel(self.fine_ce_criteria, device_ids=opt.gpu_ids)


        # learning rate_control
        self.old_lr_detector = self.opt.lr
        self.optimizer_detector = torch.optim.Adam(self.detector.parameters(),
                                                   lr=self.old_lr_detector,
                                                   betas=(0.9, 0.999),
                                                   weight_decay=0.0005)

        # place holder for GPU tensors
        self.pc = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.intensity = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.label = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.sn = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.node_a = torch.empty(self.opt.batch_size, 3, self.opt.node_a_num, dtype=torch.float, device=self.opt.device)
        self.node_b = torch.empty(self.opt.batch_size, 3, self.opt.node_b_num, dtype=torch.float, device=self.opt.device)
        self.pc_graph = torch.empty(self.opt.batch_size, 30, 12, dtype=torch.float, device=self.opt.device)
        self.img_graph = torch.empty(self.opt.batch_size, 30, 12, dtype=torch.float, device=self.opt.device)
        self.P = torch.empty(self.opt.batch_size, 3, 4, dtype=torch.float, device=self.opt.device)
        self.img = torch.empty(self.opt.batch_size, 3, self.opt.img_H, self.opt.img_W, dtype=torch.float, device=self.opt.device)
        self.K = torch.empty(self.opt.batch_size, 3, 3, dtype=torch.float, device=self.opt.device)
        self.target = torch.empty(self.opt.batch_size, 1, dtype=torch.float, device=self.opt.device)

        # record the train / test loss and accuracy
        self.train_loss_dict = {}
        self.test_loss_dict = {}

        self.pred_batch = {}
        self.gt_batch = {}

        # pre-cautions for shared memory usage
        # if opt.batch_size * 2 / len(opt.gpu_ids) > operations.CUDA_SHARED_MEM_DIM_X or opt.node_num > operations.CUDA_SHARED_MEM_DIM_Y:
        #     print('--- WARNING: batch_size or node_num larger than pre-defined cuda shared memory array size. '
        #           'Please modify CUDA_SHARED_MEM_DIM_X and CUDA_SHARED_MEM_DIM_Y in models/operations.py')

    def global_step_inc(self, delta):
        self.global_step += delta

    def load_model(self, model_path):
        self.detector.load_state_dict(
            pytorch_helper.model_state_dict_convert_auto(
                torch.load(
                    model_path,
                    map_location='cpu'), self.opt.gpu_ids))

    # 这里进行数据的填装
    def set_input(self,
                  pc, intensity, label, sn, node_a, node_b, pc_graph, img_graph, 
                  P,
                  img, K, target):
        self.pc.resize_(pc.size()).copy_(pc).detach()
        self.intensity.resize_(intensity.size()).copy_(intensity).detach()
        self.label.resize_(label.size()).copy_(label).detach()
        self.sn.resize_(sn.size()).copy_(sn).detach()
        self.node_a.resize_(node_a.size()).copy_(node_a).detach()
        self.node_b.resize_(node_b.size()).copy_(node_b).detach()
        self.pc_graph.resize_(pc_graph.size()).copy_(pc_graph).detach()
        self.img_graph.resize_(img_graph.size()).copy_(img_graph).detach()
        self.P.resize_(P.size()).copy_(P).detach()
        self.img.resize_(img.size()).copy_(img).detach()
        self.K.resize_(K.size()).copy_(K).detach()
        self.target.resize_(target.size()).copy_(target).detach()

    def forward(self,
                pc, intensity, sn, label, node_a, node_b, pc_graph, img_graph, 
                img):
        return self.detector(pc, intensity, sn, label, node_a, node_b, pc_graph, img_graph, img)

    def inference_pass(self):
        N = self.pc.size(2)
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(self.opt.img_W / self.opt.img_fine_resolution_scale)
        img_H_fine_res = int(self.opt.img_H / self.opt.img_fine_resolution_scale)

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # BLx2xN, BLxKxN
        coarse_scores, fine_scores = self.forward(self.pc, self.intensity, self.sn, self.node_a, self.node_b,
                                                  self.img)
        K = fine_scores.size(1)
        assert K == img_W_fine_res * img_H_fine_res
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        _, coarse_prediction = torch.max(coarse_scores, dim=1, keepdim=False)  # BLxN
        _, fine_prediction = torch.max(fine_scores, dim=1, keepdim=False)  # BLxN
        return coarse_prediction, fine_prediction

    ###前向传播
    def foraward_pass(self):
        N = self.pc.size(2)
        # print('n = ', N)
        Ma, Mb = self.opt.node_a_num, self.opt.node_b_num
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(round(W / self.opt.img_fine_resolution_scale))
        img_H_fine_res = int(round(H / self.opt.img_fine_resolution_scale))

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # 重点关注这个模块，是特征提取模块   将视觉特征和图像都送进去处理了
        # Bx2xN, BxLxN
        prediction = self.forward(self.pc, self.intensity, self.sn, self.label, self.node_a, self.node_b, self.pc_graph, self.img_graph, 
                                                  self.img)
        # L = fine_scores.size(1)
        # print(coarse_scores)
        # print('the coarse_scores shape is ', coarse_scores.shape) # ([8, 2, 20480])
        # print('the fine_scores shape is ', fine_scores.shape) # ([8, 80, 20480])
        # print('the score shape is ', prediction.shape) # ([8, 1])
        # assert L == img_W_fine_res * img_H_fine_res
        # build loss -------------------------------------------------------------------------------------
        # 'cuda:'+str(1)+','str(2)+str(3)
        loss = torch.mean(torch.nn.functional.binary_cross_entropy(prediction, self.target))
        # loss = coarse_loss + fine_loss
        # build loss -------------------------------------------------------------------------------------

        # get accuracy
        # get predictions for visualization
        # class_accuracy = torch.sum(torch.eq(self.target, prediction).to(dtype=torch.float)) /  B
        pred_batch = prediction.cpu().detach().numpy().reshape(-1)
        gt_batch = self.target.cpu().detach().numpy().reshape(-1)

        loss_dict = {'loss': loss}
        # accuracy_dict = {'class_accuracy': class_accuracy}

        # debug
        if self.opt.is_debug:
            print(loss_dict)
            # print(accuracy_dict)
        # print(loss_dict)

        return loss_dict, pred_batch, gt_batch

    def optimize(self):
        self.detector.train()
        self.detector.zero_grad()
        time1 = time.time()
        self.train_loss_dict,_,_ = self.foraward_pass()
        time2 = time.time()
        # print("the foraward cost : ", time2 - time1)
        self.train_loss_dict['loss'].backward()
        time3 = time.time()
        # print("the backward cost : ", time3 - time2)
        self.optimizer_detector.step()

    def test_model(self):
        self.detector.eval()
        with torch.no_grad():
            self.test_loss_dict, self.pred_batch, self.gt_batch = self.foraward_pass()

    def freeze_model(self):
        print("!!! WARNING: Model freezed.")
        for p in self.detector.parameters():
            p.requires_grad = False

    def run_model(self, pc, intensity, sn, node, img):
        self.detector.eval()
        with torch.no_grad():
            raise Exception("Not implemented.")

    def tensor_dict_to_float_dict(self, tensor_dict):
        float_dict = {}
        for key, value in tensor_dict.items():
            if type(value) == torch.Tensor:
                float_dict[key] = value.item()
            else:
                float_dict[key] = value
        return float_dict

    def get_current_errors(self):
        return self.tensor_dict_to_float_dict(self.train_loss_dict), \
               self.tensor_dict_to_float_dict(self.test_loss_dict)

    def get_current_accuracy(self):
        return self.tensor_dict_to_float_dict(self.train_accuracy), \
               self.tensor_dict_to_float_dict(self.test_accuracy)

    def get_current_batch(self):
        return self.pred_batch, self.gt_batch

    @staticmethod
    def print_loss_dict(loss_dict, accuracy_dict=None, duration=-1):
        output = 'Per sample time: %.4f - ' % (duration)
        for key, value in loss_dict.items():
            output += '%s: %.4f, ' % (key, value)
        if accuracy_dict is not None:
            for key, value in accuracy_dict.items():
                output += '%s: %.4f, ' % (key, value)
        print(output)

    def save_network(self, network, save_filename):
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # detector
        lr_detector = self.old_lr_detector * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in self.optimizer_detector.param_groups:
            param_group['lr'] = lr_detector
        print('update detector learning rate: %f -> %f' % (self.old_lr_detector, lr_detector))
        self.old_lr_detector = lr_detector

    # visualization with tensorboard, pytorch 1.2
    def write_loss(self):
        train_loss_dict, test_loss_dict = self.get_current_errors()
        self.writer.add_scalars('train_loss',
                                train_loss_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_loss',
                                test_loss_dict,
                                global_step=self.global_step)

    def write_accuracy(self):
        train_acc_dict, test_acc_dict = self.get_current_accuracy()
        self.writer.add_scalars('train_accuracy',
                                train_acc_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_accuracy',
                                test_acc_dict,
                                global_step=self.global_step)

    def write_pc_label(self, pc, labels, title):
        """
        :param pc: Bx3xN
        :param labels: BxN
        :param title: string
        :return:
        """
        pc_np = pc.detach().cpu().numpy()  # Bx3xN
        labels_np = labels.detach().cpu().numpy()  # BxN

        B = pc_np.shape[0]

        visualization_fig_np_list = []
        vis_number = min(self.opt.vis_max_batch, B)
        for b in range(vis_number):
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(pc_np[b, 0, :], pc_np[b, 2, :],
                        c=labels_np[b, :],
                        cmap='jet',
                        marker='o',
                        s=1)
            fig.tight_layout()
            plt.axis('equal')
            fig_img_np = vis_tools.fig_to_np(fig)
            plt.close(fig)
            visualization_fig_np_list.append(fig_img_np)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image(title, imgs_vis_grid, global_step=self.global_step)

    def write_img(self):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW

        vis_number = min(self.opt.vis_max_batch, B)
        imgs_vis = imgs[0:vis_number, ...]  # Bx3xHxW
        imgs_vis_grid = torchvision.utils.make_grid(imgs_vis, nrow=2)
        self.writer.add_image('image', imgs_vis_grid, global_step=self.global_step)

    def write_classification_visualization(self,
                                           pc_pxpy,
                                           coarse_predictions, fine_predictions,
                                           coarse_labels, fine_labels,
                                           t_ij):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW
        imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # BxHxWx3

        pc_pxpy_np = pc_pxpy.cpu().numpy()  # Bx2xN
        coarse_predictions_np = coarse_predictions.detach().cpu().numpy()  # BxN
        fine_predictions_np = fine_predictions.detach().cpu().numpy()  # BxN
        coarse_labels_np = coarse_labels.detach().cpu().numpy()  # BxN
        fine_labels_np = fine_labels.detach().cpu().numpy()  # BxN

        t_ij_np = t_ij.cpu().numpy()  # Bx3

        vis_number = min(self.opt.vis_max_batch, B)
        visualization_fig_np_list = []
        for b in range(vis_number):
            img_b = imgs_np[b, ...]  # HxWx3
            pc_pxpy_b = pc_pxpy_np[b, ...]  # 2xN
            coarse_prediction_b = coarse_predictions_np[b, :]  # N
            fine_predictions_b = fine_predictions_np[b, :]  # N
            coarse_labels_b = coarse_labels_np[b, :]  # N
            fine_labels_b = fine_labels_np[b, :]  # N

            vis_img = vis_tools.get_classification_visualization(pc_pxpy_b,
                                                                 coarse_prediction_b, fine_predictions_b,
                                                                 coarse_labels_b, fine_labels_b,
                                                                 img_b,
                                                                 img_fine_resolution_scale=self.opt.img_fine_resolution_scale,
                                                                 H_delta=100, W_delta=100,
                                                                 circle_size=1,
                                                                 t_ij_np=t_ij_np[b, :])
            visualization_fig_np_list.append(vis_img)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image("Classification Visualization", imgs_vis_grid, global_step=self.global_step)

class MMClassiferSe():
    def __init__(self, opt: Options, writer):
        self.opt = opt
        self.writer = writer
        self.global_step = 0

        # 这个东西是一个关键点的检测？
        self.detector = networks_united.KeypointDetector(self.opt).to(self.opt.device)
        # self.coarse_ce_criteria = nn.CrossEntropyLoss()
        self.coarse_ce_criteria = focal_loss.FocalLoss(alpha=0.5, gamma=2, reduction='mean')
        self.fine_ce_criteria = nn.CrossEntropyLoss()

        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.detector = nn.DataParallel(self.detector, device_ids=opt.gpu_ids)
            # self.coarse_ce_criteria = nn.DataParallel(self.coarse_ce_criteria, device_ids=opt.gpu_ids)
            # self.fine_ce_criteria = nn.DataParallel(self.fine_ce_criteria, device_ids=opt.gpu_ids)


        # learning rate_control
        self.old_lr_detector = self.opt.lr
        self.optimizer_detector = torch.optim.Adam(self.detector.parameters(),
                                                   lr=self.old_lr_detector,
                                                   betas=(0.9, 0.999),
                                                   weight_decay=0.0005)

        # place holder for GPU tensors
        self.pc = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.intensity = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.label = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.sn = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.node_a = torch.empty(self.opt.batch_size, 3, self.opt.node_a_num, dtype=torch.float, device=self.opt.device)
        self.node_b = torch.empty(self.opt.batch_size, 3, self.opt.node_b_num, dtype=torch.float, device=self.opt.device)
        self.P = torch.empty(self.opt.batch_size, 3, 4, dtype=torch.float, device=self.opt.device)
        self.img = torch.empty(self.opt.batch_size, 3, self.opt.img_H, self.opt.img_W, dtype=torch.float, device=self.opt.device)
        self.K = torch.empty(self.opt.batch_size, 3, 3, dtype=torch.float, device=self.opt.device)
        self.target = torch.empty(self.opt.batch_size, 1, dtype=torch.float, device=self.opt.device)

        # record the train / test loss and accuracy
        self.train_loss_dict = {}
        self.test_loss_dict = {}

        self.pred_batch = {}
        self.gt_batch = {}

        # pre-cautions for shared memory usage
        # if opt.batch_size * 2 / len(opt.gpu_ids) > operations.CUDA_SHARED_MEM_DIM_X or opt.node_num > operations.CUDA_SHARED_MEM_DIM_Y:
        #     print('--- WARNING: batch_size or node_num larger than pre-defined cuda shared memory array size. '
        #           'Please modify CUDA_SHARED_MEM_DIM_X and CUDA_SHARED_MEM_DIM_Y in models/operations.py')

    def global_step_inc(self, delta):
        self.global_step += delta

    def load_model(self, model_path):
        self.detector.load_state_dict(
            pytorch_helper.model_state_dict_convert_auto(
                torch.load(
                    model_path,
                    map_location='cpu'), self.opt.gpu_ids))

    # 这里进行数据的填装
    def set_input(self,
                  pc, intensity, label, sn, node_a, node_b,
                  P,
                  img, K, target):
        self.pc.resize_(pc.size()).copy_(pc).detach()
        self.intensity.resize_(intensity.size()).copy_(intensity).detach()
        self.label.resize_(label.size()).copy_(label).detach()
        self.sn.resize_(sn.size()).copy_(sn).detach()
        self.node_a.resize_(node_a.size()).copy_(node_a).detach()
        self.node_b.resize_(node_b.size()).copy_(node_b).detach()
        self.P.resize_(P.size()).copy_(P).detach()
        self.img.resize_(img.size()).copy_(img).detach()
        self.K.resize_(K.size()).copy_(K).detach()
        self.target.resize_(target.size()).copy_(target).detach()

    def forward(self,
                pc, intensity, sn, label, node_a, node_b,
                img):
        return self.detector(pc, intensity, sn, label, node_a, node_b, img)

    def inference_pass(self):
        N = self.pc.size(2)
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(self.opt.img_W / self.opt.img_fine_resolution_scale)
        img_H_fine_res = int(self.opt.img_H / self.opt.img_fine_resolution_scale)

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # BLx2xN, BLxKxN
        coarse_scores, fine_scores = self.forward(self.pc, self.intensity, self.sn, self.node_a, self.node_b,
                                                  self.img)
        K = fine_scores.size(1)
        assert K == img_W_fine_res * img_H_fine_res
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        _, coarse_prediction = torch.max(coarse_scores, dim=1, keepdim=False)  # BLxN
        _, fine_prediction = torch.max(fine_scores, dim=1, keepdim=False)  # BLxN
        return coarse_prediction, fine_prediction

    ###前向传播
    def foraward_pass(self):
        N = self.pc.size(2)
        # print('n = ', N)
        Ma, Mb = self.opt.node_a_num, self.opt.node_b_num
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(round(W / self.opt.img_fine_resolution_scale))
        img_H_fine_res = int(round(H / self.opt.img_fine_resolution_scale))

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # 重点关注这个模块，是特征提取模块   将视觉特征和图像都送进去处理了
        # Bx2xN, BxLxN
        prediction = self.forward(self.pc, self.intensity, self.sn, self.label, self.node_a, self.node_b,
                                                  self.img)
        # L = fine_scores.size(1)
        # print(coarse_scores)
        # print('the coarse_scores shape is ', coarse_scores.shape) # ([8, 2, 20480])
        # print('the fine_scores shape is ', fine_scores.shape) # ([8, 80, 20480])
        # print('the score shape is ', prediction.shape) # ([8, 1])
        # assert L == img_W_fine_res * img_H_fine_res
        # build loss -------------------------------------------------------------------------------------
        # 'cuda:'+str(1)+','str(2)+str(3)
        loss = torch.mean(torch.nn.functional.binary_cross_entropy(prediction, self.target))
        # loss = coarse_loss + fine_loss
        # build loss -------------------------------------------------------------------------------------

        # get accuracy
        # get predictions for visualization
        # class_accuracy = torch.sum(torch.eq(self.target, prediction).to(dtype=torch.float)) /  B
        pred_batch = prediction.cpu().detach().numpy().reshape(-1)
        gt_batch = self.target.cpu().detach().numpy().reshape(-1)

        loss_dict = {'loss': loss}
        # accuracy_dict = {'class_accuracy': class_accuracy}

        # debug
        if self.opt.is_debug:
            print(loss_dict)
            # print(accuracy_dict)
        # print(loss_dict)

        return loss_dict, pred_batch, gt_batch

    def optimize(self):
        self.detector.train()
        self.detector.zero_grad()
        time1 = time.time()
        self.train_loss_dict,_,_ = self.foraward_pass()
        time2 = time.time()
        # print("the foraward cost : ", time2 - time1)
        self.train_loss_dict['loss'].backward()
        time3 = time.time()
        # print("the backward cost : ", time3 - time2)
        self.optimizer_detector.step()

    def test_model(self):
        self.detector.eval()
        with torch.no_grad():
            self.test_loss_dict, self.pred_batch, self.gt_batch = self.foraward_pass()

    def freeze_model(self):
        print("!!! WARNING: Model freezed.")
        for p in self.detector.parameters():
            p.requires_grad = False

    def run_model(self, pc, intensity, sn, node, img):
        self.detector.eval()
        with torch.no_grad():
            raise Exception("Not implemented.")

    def tensor_dict_to_float_dict(self, tensor_dict):
        float_dict = {}
        for key, value in tensor_dict.items():
            if type(value) == torch.Tensor:
                float_dict[key] = value.item()
            else:
                float_dict[key] = value
        return float_dict

    def get_current_errors(self):
        return self.tensor_dict_to_float_dict(self.train_loss_dict), \
               self.tensor_dict_to_float_dict(self.test_loss_dict)

    def get_current_accuracy(self):
        return self.tensor_dict_to_float_dict(self.train_accuracy), \
               self.tensor_dict_to_float_dict(self.test_accuracy)

    def get_current_batch(self):
        return self.pred_batch, self.gt_batch

    @staticmethod
    def print_loss_dict(loss_dict, accuracy_dict=None, duration=-1):
        output = 'Per sample time: %.4f - ' % (duration)
        for key, value in loss_dict.items():
            output += '%s: %.4f, ' % (key, value)
        if accuracy_dict is not None:
            for key, value in accuracy_dict.items():
                output += '%s: %.4f, ' % (key, value)
        print(output)

    def save_network(self, network, save_filename):
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # detector
        lr_detector = self.old_lr_detector * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in self.optimizer_detector.param_groups:
            param_group['lr'] = lr_detector
        print('update detector learning rate: %f -> %f' % (self.old_lr_detector, lr_detector))
        self.old_lr_detector = lr_detector

    # visualization with tensorboard, pytorch 1.2
    def write_loss(self):
        train_loss_dict, test_loss_dict = self.get_current_errors()
        self.writer.add_scalars('train_loss',
                                train_loss_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_loss',
                                test_loss_dict,
                                global_step=self.global_step)

    def write_accuracy(self):
        train_acc_dict, test_acc_dict = self.get_current_accuracy()
        self.writer.add_scalars('train_accuracy',
                                train_acc_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_accuracy',
                                test_acc_dict,
                                global_step=self.global_step)

    def write_pc_label(self, pc, labels, title):
        """
        :param pc: Bx3xN
        :param labels: BxN
        :param title: string
        :return:
        """
        pc_np = pc.detach().cpu().numpy()  # Bx3xN
        labels_np = labels.detach().cpu().numpy()  # BxN

        B = pc_np.shape[0]

        visualization_fig_np_list = []
        vis_number = min(self.opt.vis_max_batch, B)
        for b in range(vis_number):
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(pc_np[b, 0, :], pc_np[b, 2, :],
                        c=labels_np[b, :],
                        cmap='jet',
                        marker='o',
                        s=1)
            fig.tight_layout()
            plt.axis('equal')
            fig_img_np = vis_tools.fig_to_np(fig)
            plt.close(fig)
            visualization_fig_np_list.append(fig_img_np)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image(title, imgs_vis_grid, global_step=self.global_step)

    def write_img(self):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW

        vis_number = min(self.opt.vis_max_batch, B)
        imgs_vis = imgs[0:vis_number, ...]  # Bx3xHxW
        imgs_vis_grid = torchvision.utils.make_grid(imgs_vis, nrow=2)
        self.writer.add_image('image', imgs_vis_grid, global_step=self.global_step)

    def write_classification_visualization(self,
                                           pc_pxpy,
                                           coarse_predictions, fine_predictions,
                                           coarse_labels, fine_labels,
                                           t_ij):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW
        imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # BxHxWx3

        pc_pxpy_np = pc_pxpy.cpu().numpy()  # Bx2xN
        coarse_predictions_np = coarse_predictions.detach().cpu().numpy()  # BxN
        fine_predictions_np = fine_predictions.detach().cpu().numpy()  # BxN
        coarse_labels_np = coarse_labels.detach().cpu().numpy()  # BxN
        fine_labels_np = fine_labels.detach().cpu().numpy()  # BxN

        t_ij_np = t_ij.cpu().numpy()  # Bx3

        vis_number = min(self.opt.vis_max_batch, B)
        visualization_fig_np_list = []
        for b in range(vis_number):
            img_b = imgs_np[b, ...]  # HxWx3
            pc_pxpy_b = pc_pxpy_np[b, ...]  # 2xN
            coarse_prediction_b = coarse_predictions_np[b, :]  # N
            fine_predictions_b = fine_predictions_np[b, :]  # N
            coarse_labels_b = coarse_labels_np[b, :]  # N
            fine_labels_b = fine_labels_np[b, :]  # N

            vis_img = vis_tools.get_classification_visualization(pc_pxpy_b,
                                                                 coarse_prediction_b, fine_predictions_b,
                                                                 coarse_labels_b, fine_labels_b,
                                                                 img_b,
                                                                 img_fine_resolution_scale=self.opt.img_fine_resolution_scale,
                                                                 H_delta=100, W_delta=100,
                                                                 circle_size=1,
                                                                 t_ij_np=t_ij_np[b, :])
            visualization_fig_np_list.append(vis_img)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image("Classification Visualization", imgs_vis_grid, global_step=self.global_step)

class MMClassiferSeDime():
    def __init__(self, opt: Options, writer):
        self.opt = opt
        self.writer = writer
        self.global_step = 0

        # 这个东西是一个关键点的检测？
        self.detector = dime_united.KeypointDetector(self.opt).to(self.opt.device)
        # self.coarse_ce_criteria = nn.CrossEntropyLoss()
        self.coarse_ce_criteria = focal_loss.FocalLoss(alpha=0.5, gamma=2, reduction='mean')
        self.fine_ce_criteria = nn.CrossEntropyLoss()


        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.detector = nn.DataParallel(self.detector, device_ids=opt.gpu_ids)
            # self.coarse_ce_criteria = nn.DataParallel(self.coarse_ce_criteria, device_ids=opt.gpu_ids)
            # self.fine_ce_criteria = nn.DataParallel(self.fine_ce_criteria, device_ids=opt.gpu_ids)


        # learning rate_control
        self.old_lr_detector = self.opt.lr
        self.optimizer_detector = torch.optim.Adam(self.detector.parameters(),
                                                   lr=self.old_lr_detector,
                                                   betas=(0.9, 0.999),
                                                   weight_decay=0.0005)
        # self.criterion = ContrastiveLoss(opt, margin=opt.margin,
        #                                     measure=opt.measure,
        #                                     max_violation=opt.max_violation)

        # place holder for GPU tensors
        self.pc = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.intensity = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.label = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.sn = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.node_a = torch.empty(self.opt.batch_size, 3, self.opt.node_a_num, dtype=torch.float, device=self.opt.device)
        self.node_b = torch.empty(self.opt.batch_size, 3, self.opt.node_b_num, dtype=torch.float, device=self.opt.device)
        self.P = torch.empty(self.opt.batch_size, 3, 4, dtype=torch.float, device=self.opt.device)
        self.img = torch.empty(self.opt.batch_size, 3, self.opt.img_H, self.opt.img_W, dtype=torch.float, device=self.opt.device)
        self.K = torch.empty(self.opt.batch_size, 3, 3, dtype=torch.float, device=self.opt.device)
        self.target = torch.empty(self.opt.batch_size, 1, dtype=torch.float, device=self.opt.device)

        # record the train / test loss and accuracy
        self.train_loss_dict = {}
        self.test_loss_dict = {}

        self.pred_batch = {}
        self.gt_batch = {}

        # pre-cautions for shared memory usage
        # if opt.batch_size * 2 / len(opt.gpu_ids) > operations.CUDA_SHARED_MEM_DIM_X or opt.node_num > operations.CUDA_SHARED_MEM_DIM_Y:
        #     print('--- WARNING: batch_size or node_num larger than pre-defined cuda shared memory array size. '
        #           'Please modify CUDA_SHARED_MEM_DIM_X and CUDA_SHARED_MEM_DIM_Y in models/operations.py')

    def global_step_inc(self, delta):
        self.global_step += delta

    def load_model(self, model_path):
        self.detector.load_state_dict(
            pytorch_helper.model_state_dict_convert_auto(
                torch.load(
                    model_path,
                    map_location='cpu'), self.opt.gpu_ids))

    # 这里进行数据的填装
    def set_input(self,
                  pc, intensity, label, sn, node_a, node_b,
                  P,
                  img, K, target):
        self.pc.resize_(pc.size()).copy_(pc).detach()
        self.intensity.resize_(intensity.size()).copy_(intensity).detach()
        self.label.resize_(label.size()).copy_(label).detach()
        self.sn.resize_(sn.size()).copy_(sn).detach()
        self.node_a.resize_(node_a.size()).copy_(node_a).detach()
        self.node_b.resize_(node_b.size()).copy_(node_b).detach()
        self.P.resize_(P.size()).copy_(P).detach()
        self.img.resize_(img.size()).copy_(img).detach()
        self.K.resize_(K.size()).copy_(K).detach()
        self.target.resize_(target.size()).copy_(target).detach()

    def forward(self,
                pc, intensity, sn, label, node_a, node_b,
                img):
        return self.detector(pc, intensity, sn, label, node_a, node_b, img)

    def inference_pass(self):
        N = self.pc.size(2)
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(self.opt.img_W / self.opt.img_fine_resolution_scale)
        img_H_fine_res = int(self.opt.img_H / self.opt.img_fine_resolution_scale)

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # BLx2xN, BLxKxN
        coarse_scores, fine_scores = self.forward(self.pc, self.intensity, self.sn, self.node_a, self.node_b,
                                                  self.img)
        K = fine_scores.size(1)
        assert K == img_W_fine_res * img_H_fine_res
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        _, coarse_prediction = torch.max(coarse_scores, dim=1, keepdim=False)  # BLxN
        _, fine_prediction = torch.max(fine_scores, dim=1, keepdim=False)  # BLxN
        return coarse_prediction, fine_prediction

    ###前向传播
    def foraward_pass(self):
        # print("comming here success")
        N = self.pc.size(2)
        # print('n = ', N)
        Ma, Mb = self.opt.node_a_num, self.opt.node_b_num
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(round(W / self.opt.img_fine_resolution_scale))
        img_H_fine_res = int(round(H / self.opt.img_fine_resolution_scale))

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # 重点关注这个模块，是特征提取模块   将视觉特征和图像都送进去处理了
        # Bx2xN, BxLxN
        sim_mat, sim_paths = self.forward(self.pc, self.intensity, self.sn, self.label, self.node_a, self.node_b,
                                                  self.img)
        # print(sim_mat.shape)
        # 计算最大值和最小值
        min_value = torch.min(sim_mat)
        max_value = torch.max(sim_mat)

        # 进行归一化处理
        sim_mat_hat = (sim_mat - min_value) / (max_value - min_value)
        #这里sim_mat就是图像提取出来的特征和雷达的全局特征之间的矩阵相似度，首先直接用这个来尝试构建一下loss
        loss = torch.mean(torch.nn.functional.binary_cross_entropy(sim_mat_hat, self.target))
        # loss = coarse_loss + fine_loss
        # build loss -------------------------------------------------------------------------------------

        # get accuracy
        # get predictions for visualization
        # class_accuracy = torch.sum(torch.eq(self.target, prediction).to(dtype=torch.float)) /  B
        pred_batch = sim_mat.cpu().detach().numpy().reshape(-1)
        gt_batch = self.target.cpu().detach().numpy().reshape(-1)

        loss_dict = {'loss': loss}

        return loss_dict, pred_batch, gt_batch

    def optimize(self):
        self.detector.train()
        self.detector.zero_grad()
        # time1 = time.time()
        self.train_loss_dict,_,_ = self.foraward_pass()
        # time2 = time.time()
        # print("the foraward cost : ", time2 - time1)
        # self.loss.backward()
        self.train_loss_dict['loss'].backward()
        # time3 = time.time()
        # print("the backward cost : ", time3 - time2)
        self.optimizer_detector.step()

    def test_model(self):
        self.detector.eval()
        with torch.no_grad():
            self.test_loss_dict, self.pred_batch, self.gt_batch = self.foraward_pass()

    def freeze_model(self):
        print("!!! WARNING: Model freezed.")
        for p in self.detector.parameters():
            p.requires_grad = False

    def run_model(self, pc, intensity, sn, node, img):
        self.detector.eval()
        with torch.no_grad():
            raise Exception("Not implemented.")

    def tensor_dict_to_float_dict(self, tensor_dict):
        float_dict = {}
        for key, value in tensor_dict.items():
            if type(value) == torch.Tensor:
                float_dict[key] = value.item()
            else:
                float_dict[key] = value
        return float_dict

    def get_current_errors(self):
        return self.tensor_dict_to_float_dict(self.train_loss_dict), \
               self.tensor_dict_to_float_dict(self.test_loss_dict)

    def get_current_accuracy(self):
        return self.tensor_dict_to_float_dict(self.train_accuracy), \
               self.tensor_dict_to_float_dict(self.test_accuracy)

    def get_current_batch(self):
        return self.pred_batch, self.gt_batch

    @staticmethod
    def print_loss_dict(loss_dict, accuracy_dict=None, duration=-1):
        output = 'Per sample time: %.4f - ' % (duration)
        for key, value in loss_dict.items():
            output += '%s: %.4f, ' % (key, value)
        if accuracy_dict is not None:
            for key, value in accuracy_dict.items():
                output += '%s: %.4f, ' % (key, value)
        print(output)

    def save_network(self, network, save_filename):
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # detector
        lr_detector = self.old_lr_detector * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in self.optimizer_detector.param_groups:
            param_group['lr'] = lr_detector
        print('update detector learning rate: %f -> %f' % (self.old_lr_detector, lr_detector))
        self.old_lr_detector = lr_detector

    # visualization with tensorboard, pytorch 1.2
    def write_loss(self):
        train_loss_dict, test_loss_dict = self.get_current_errors()
        self.writer.add_scalars('train_loss',
                                train_loss_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_loss',
                                test_loss_dict,
                                global_step=self.global_step)

    def write_accuracy(self):
        train_acc_dict, test_acc_dict = self.get_current_accuracy()
        self.writer.add_scalars('train_accuracy',
                                train_acc_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_accuracy',
                                test_acc_dict,
                                global_step=self.global_step)

    def write_pc_label(self, pc, labels, title):
        """
        :param pc: Bx3xN
        :param labels: BxN
        :param title: string
        :return:
        """
        pc_np = pc.detach().cpu().numpy()  # Bx3xN
        labels_np = labels.detach().cpu().numpy()  # BxN

        B = pc_np.shape[0]

        visualization_fig_np_list = []
        vis_number = min(self.opt.vis_max_batch, B)
        for b in range(vis_number):
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(pc_np[b, 0, :], pc_np[b, 2, :],
                        c=labels_np[b, :],
                        cmap='jet',
                        marker='o',
                        s=1)
            fig.tight_layout()
            plt.axis('equal')
            fig_img_np = vis_tools.fig_to_np(fig)
            plt.close(fig)
            visualization_fig_np_list.append(fig_img_np)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image(title, imgs_vis_grid, global_step=self.global_step)

    def write_img(self):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW

        vis_number = min(self.opt.vis_max_batch, B)
        imgs_vis = imgs[0:vis_number, ...]  # Bx3xHxW
        imgs_vis_grid = torchvision.utils.make_grid(imgs_vis, nrow=2)
        self.writer.add_image('image', imgs_vis_grid, global_step=self.global_step)

    def write_classification_visualization(self,
                                           pc_pxpy,
                                           coarse_predictions, fine_predictions,
                                           coarse_labels, fine_labels,
                                           t_ij):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW
        imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # BxHxWx3

        pc_pxpy_np = pc_pxpy.cpu().numpy()  # Bx2xN
        coarse_predictions_np = coarse_predictions.detach().cpu().numpy()  # BxN
        fine_predictions_np = fine_predictions.detach().cpu().numpy()  # BxN
        coarse_labels_np = coarse_labels.detach().cpu().numpy()  # BxN
        fine_labels_np = fine_labels.detach().cpu().numpy()  # BxN

        t_ij_np = t_ij.cpu().numpy()  # Bx3

        vis_number = min(self.opt.vis_max_batch, B)
        visualization_fig_np_list = []
        for b in range(vis_number):
            img_b = imgs_np[b, ...]  # HxWx3
            pc_pxpy_b = pc_pxpy_np[b, ...]  # 2xN
            coarse_prediction_b = coarse_predictions_np[b, :]  # N
            fine_predictions_b = fine_predictions_np[b, :]  # N
            coarse_labels_b = coarse_labels_np[b, :]  # N
            fine_labels_b = fine_labels_np[b, :]  # N

            vis_img = vis_tools.get_classification_visualization(pc_pxpy_b,
                                                                 coarse_prediction_b, fine_predictions_b,
                                                                 coarse_labels_b, fine_labels_b,
                                                                 img_b,
                                                                 img_fine_resolution_scale=self.opt.img_fine_resolution_scale,
                                                                 H_delta=100, W_delta=100,
                                                                 circle_size=1,
                                                                 t_ij_np=t_ij_np[b, :])
            visualization_fig_np_list.append(vis_img)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image("Classification Visualization", imgs_vis_grid, global_step=self.global_step)

class MMClassiferBase():
    def __init__(self, opt: Options, writer):
        self.opt = opt
        self.writer = writer
        self.global_step = 0

        # 这个东西是一个关键点的检测？
        self.detector = networks_united.KeypointDetector_Base(self.opt).to(self.opt.device)
        # self.coarse_ce_criteria = nn.CrossEntropyLoss()
        self.coarse_ce_criteria = focal_loss.FocalLoss(alpha=0.5, gamma=2, reduction='mean')
        self.fine_ce_criteria = nn.CrossEntropyLoss()

        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.detector = nn.DataParallel(self.detector, device_ids=opt.gpu_ids)
            # self.coarse_ce_criteria = nn.DataParallel(self.coarse_ce_criteria, device_ids=opt.gpu_ids)
            # self.fine_ce_criteria = nn.DataParallel(self.fine_ce_criteria, device_ids=opt.gpu_ids)


        # learning rate_control
        self.old_lr_detector = self.opt.lr
        self.optimizer_detector = torch.optim.Adam(self.detector.parameters(),
                                                   lr=self.old_lr_detector,
                                                   betas=(0.9, 0.999),
                                                   weight_decay=0.0005)

        # place holder for GPU tensors
        self.pc = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.intensity = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.label = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.sn = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.node_a = torch.empty(self.opt.batch_size, 3, self.opt.node_a_num, dtype=torch.float, device=self.opt.device)
        self.node_b = torch.empty(self.opt.batch_size, 3, self.opt.node_b_num, dtype=torch.float, device=self.opt.device)
        self.P = torch.empty(self.opt.batch_size, 3, 4, dtype=torch.float, device=self.opt.device)
        self.img = torch.empty(self.opt.batch_size, 3, self.opt.img_H, self.opt.img_W, dtype=torch.float, device=self.opt.device)
        self.K = torch.empty(self.opt.batch_size, 3, 3, dtype=torch.float, device=self.opt.device)
        self.target = torch.empty(self.opt.batch_size, 1, dtype=torch.float, device=self.opt.device)

        # record the train / test loss and accuracy
        self.train_loss_dict = {}
        self.test_loss_dict = {}

        self.pred_batch = {}
        self.gt_batch = {}

        # pre-cautions for shared memory usage
        # if opt.batch_size * 2 / len(opt.gpu_ids) > operations.CUDA_SHARED_MEM_DIM_X or opt.node_num > operations.CUDA_SHARED_MEM_DIM_Y:
        #     print('--- WARNING: batch_size or node_num larger than pre-defined cuda shared memory array size. '
        #           'Please modify CUDA_SHARED_MEM_DIM_X and CUDA_SHARED_MEM_DIM_Y in models/operations.py')

    def global_step_inc(self, delta):
        self.global_step += delta

    def load_model(self, model_path):
        self.detector.load_state_dict(
            pytorch_helper.model_state_dict_convert_auto(
                torch.load(
                    model_path,
                    map_location='cpu'), self.opt.gpu_ids))

    # 这里进行数据的填装
    def set_input(self,
                  pc, intensity,sn,label,node_a, node_b,
                  P,
                  img, K, target):
        self.pc.resize_(pc.size()).copy_(pc).detach()
        self.intensity.resize_(intensity.size()).copy_(intensity).detach()
        self.label.resize_(label.size()).copy_(label).detach()
        self.sn.resize_(sn.size()).copy_(sn).detach()
        self.node_a.resize_(node_a.size()).copy_(node_a).detach()
        self.node_b.resize_(node_b.size()).copy_(node_b).detach()
        self.P.resize_(P.size()).copy_(P).detach()
        self.img.resize_(img.size()).copy_(img).detach()
        self.K.resize_(K.size()).copy_(K).detach()
        self.target.resize_(target.size()).copy_(target).detach()

    def forward(self,
                pc, intensity, sn, label, node_a, node_b,
                img):
        return self.detector(pc, intensity, sn, label, node_a, node_b, img)

    def inference_pass(self):
        N = self.pc.size(2)
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(self.opt.img_W / self.opt.img_fine_resolution_scale)
        img_H_fine_res = int(self.opt.img_H / self.opt.img_fine_resolution_scale)

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # BLx2xN, BLxKxN
        coarse_scores, fine_scores = self.forward(self.pc, self.intensity, self.sn, self.node_a, self.node_b,
                                                  self.img)
        K = fine_scores.size(1)
        assert K == img_W_fine_res * img_H_fine_res
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        _, coarse_prediction = torch.max(coarse_scores, dim=1, keepdim=False)  # BLxN
        _, fine_prediction = torch.max(fine_scores, dim=1, keepdim=False)  # BLxN
        return coarse_prediction, fine_prediction

    ###前向传播
    def foraward_pass(self):
        N = self.pc.size(2)
        # print('n = ', N)
        Ma, Mb = self.opt.node_a_num, self.opt.node_b_num
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(round(W / self.opt.img_fine_resolution_scale))
        img_H_fine_res = int(round(H / self.opt.img_fine_resolution_scale))

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # 重点关注这个模块，是特征提取模块   将视觉特征和图像都送进去处理了
        # Bx2xN, BxLxN
        # print("the pc shape is: ",self.pc.shape)
        # print("the instesity shape is: ",self.intensity.shape)
        # print("the sn shape is: ",self.sn.shape)
        prediction = self.forward(self.pc, self.intensity, self.sn, self.label, self.node_a, self.node_b,
                                                  self.img)
        # L = fine_scores.size(1)
        # print(coarse_scores)
        # print('the coarse_scores shape is ', coarse_scores.shape) # ([8, 2, 20480])
        # print('the fine_scores shape is ', fine_scores.shape) # ([8, 80, 20480])
        # print('the score shape is ', prediction.shape) # ([8, 1])
        # assert L == img_W_fine_res * img_H_fine_res
        # build loss -------------------------------------------------------------------------------------
        # 'cuda:'+str(1)+','str(2)+str(3)
        loss = torch.mean(torch.nn.functional.binary_cross_entropy(prediction, self.target))
        # loss = coarse_loss + fine_loss
        # build loss -------------------------------------------------------------------------------------

        # get accuracy
        # get predictions for visualization
        # class_accuracy = torch.sum(torch.eq(self.target, prediction).to(dtype=torch.float)) /  B
        pred_batch = prediction.cpu().detach().numpy().reshape(-1)
        gt_batch = self.target.cpu().detach().numpy().reshape(-1)

        loss_dict = {'loss': loss}
        # accuracy_dict = {'class_accuracy': class_accuracy}

        # debug
        if self.opt.is_debug:
            print(loss_dict)
            # print(accuracy_dict)
        # print(loss_dict)

        return loss_dict, pred_batch, gt_batch

    def optimize(self):
        self.detector.train()
        self.detector.zero_grad()
        time1 = time.time()
        self.train_loss_dict,_,_ = self.foraward_pass()
        time2 = time.time()
        # print("the foraward cost : ", time2 - time1)
        self.train_loss_dict['loss'].backward()
        time3 = time.time()
        # print("the backward cost : ", time3 - time2)
        self.optimizer_detector.step()

    def test_model(self):
        self.detector.eval()
        with torch.no_grad():
            self.test_loss_dict, self.pred_batch, self.gt_batch = self.foraward_pass()

    def freeze_model(self):
        print("!!! WARNING: Model freezed.")
        for p in self.detector.parameters():
            p.requires_grad = False

    def run_model(self, pc, intensity, sn, node, img):
        self.detector.eval()
        with torch.no_grad():
            raise Exception("Not implemented.")

    def tensor_dict_to_float_dict(self, tensor_dict):
        float_dict = {}
        for key, value in tensor_dict.items():
            if type(value) == torch.Tensor:
                float_dict[key] = value.item()
            else:
                float_dict[key] = value
        return float_dict

    def get_current_errors(self):
        return self.tensor_dict_to_float_dict(self.train_loss_dict), \
               self.tensor_dict_to_float_dict(self.test_loss_dict)

    def get_current_accuracy(self):
        return self.tensor_dict_to_float_dict(self.train_accuracy), \
               self.tensor_dict_to_float_dict(self.test_accuracy)

    def get_current_batch(self):
        return self.pred_batch, self.gt_batch

    @staticmethod
    def print_loss_dict(loss_dict, accuracy_dict=None, duration=-1):
        output = 'Per sample time: %.4f - ' % (duration)
        for key, value in loss_dict.items():
            output += '%s: %.4f, ' % (key, value)
        if accuracy_dict is not None:
            for key, value in accuracy_dict.items():
                output += '%s: %.4f, ' % (key, value)
        print(output)

    def save_network(self, network, save_filename):
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # detector
        lr_detector = self.old_lr_detector * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in self.optimizer_detector.param_groups:
            param_group['lr'] = lr_detector
        print('update detector learning rate: %f -> %f' % (self.old_lr_detector, lr_detector))
        self.old_lr_detector = lr_detector

    # visualization with tensorboard, pytorch 1.2
    def write_loss(self):
        train_loss_dict, test_loss_dict = self.get_current_errors()
        self.writer.add_scalars('train_loss',
                                train_loss_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_loss',
                                test_loss_dict,
                                global_step=self.global_step)

    def write_accuracy(self):
        train_acc_dict, test_acc_dict = self.get_current_accuracy()
        self.writer.add_scalars('train_accuracy',
                                train_acc_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_accuracy',
                                test_acc_dict,
                                global_step=self.global_step)

    def write_pc_label(self, pc, labels, title):
        """
        :param pc: Bx3xN
        :param labels: BxN
        :param title: string
        :return:
        """
        pc_np = pc.detach().cpu().numpy()  # Bx3xN
        labels_np = labels.detach().cpu().numpy()  # BxN

        B = pc_np.shape[0]

        visualization_fig_np_list = []
        vis_number = min(self.opt.vis_max_batch, B)
        for b in range(vis_number):
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(pc_np[b, 0, :], pc_np[b, 2, :],
                        c=labels_np[b, :],
                        cmap='jet',
                        marker='o',
                        s=1)
            fig.tight_layout()
            plt.axis('equal')
            fig_img_np = vis_tools.fig_to_np(fig)
            plt.close(fig)
            visualization_fig_np_list.append(fig_img_np)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image(title, imgs_vis_grid, global_step=self.global_step)

    def write_img(self):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW

        vis_number = min(self.opt.vis_max_batch, B)
        imgs_vis = imgs[0:vis_number, ...]  # Bx3xHxW
        imgs_vis_grid = torchvision.utils.make_grid(imgs_vis, nrow=2)
        self.writer.add_image('image', imgs_vis_grid, global_step=self.global_step)

    def write_classification_visualization(self,
                                           pc_pxpy,
                                           coarse_predictions, fine_predictions,
                                           coarse_labels, fine_labels,
                                           t_ij):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW
        imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # BxHxWx3

        pc_pxpy_np = pc_pxpy.cpu().numpy()  # Bx2xN
        coarse_predictions_np = coarse_predictions.detach().cpu().numpy()  # BxN
        fine_predictions_np = fine_predictions.detach().cpu().numpy()  # BxN
        coarse_labels_np = coarse_labels.detach().cpu().numpy()  # BxN
        fine_labels_np = fine_labels.detach().cpu().numpy()  # BxN

        t_ij_np = t_ij.cpu().numpy()  # Bx3

        vis_number = min(self.opt.vis_max_batch, B)
        visualization_fig_np_list = []
        for b in range(vis_number):
            img_b = imgs_np[b, ...]  # HxWx3
            pc_pxpy_b = pc_pxpy_np[b, ...]  # 2xN
            coarse_prediction_b = coarse_predictions_np[b, :]  # N
            fine_predictions_b = fine_predictions_np[b, :]  # N
            coarse_labels_b = coarse_labels_np[b, :]  # N
            fine_labels_b = fine_labels_np[b, :]  # N

            vis_img = vis_tools.get_classification_visualization(pc_pxpy_b,
                                                                 coarse_prediction_b, fine_predictions_b,
                                                                 coarse_labels_b, fine_labels_b,
                                                                 img_b,
                                                                 img_fine_resolution_scale=self.opt.img_fine_resolution_scale,
                                                                 H_delta=100, W_delta=100,
                                                                 circle_size=1,
                                                                 t_ij_np=t_ij_np[b, :])
            visualization_fig_np_list.append(vis_img)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image("Classification Visualization", imgs_vis_grid, global_step=self.global_step)

class PointvladBase():
    def __init__(self, opt: Options, writer):
        self.opt = opt
        self.writer = writer
        self.global_step = 0

        # 这个东西是一个关键点的检测？
        self.detector = PointNetVlad.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                                    output_dim=256, num_points=5120).to(self.opt.device)
        # self.coarse_ce_criteria = nn.CrossEntropyLoss()
        self.coarse_ce_criteria = focal_loss.FocalLoss(alpha=0.5, gamma=2, reduction='mean')
        self.fine_ce_criteria = nn.CrossEntropyLoss()

        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.detector = nn.DataParallel(self.detector, device_ids=opt.gpu_ids)
            # self.coarse_ce_criteria = nn.DataParallel(self.coarse_ce_criteria, device_ids=opt.gpu_ids)
            # self.fine_ce_criteria = nn.DataParallel(self.fine_ce_criteria, device_ids=opt.gpu_ids)


        # learning rate_control
        self.old_lr_detector = self.opt.lr
        self.optimizer_detector = torch.optim.Adam(self.detector.parameters(),
                                                   lr=self.old_lr_detector,
                                                   betas=(0.9, 0.999),
                                                   weight_decay=0.0005)

        # place holder for GPU tensors
        self.pc = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.pc2 = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.target = torch.empty(self.opt.batch_size, 1, dtype=torch.float, device=self.opt.device)

        # record the train / test loss and accuracy
        self.train_loss_dict = {}
        self.test_loss_dict = {}

        self.pred_batch = {}
        self.gt_batch = {}

        # pre-cautions for shared memory usage
        # if opt.batch_size * 2 / len(opt.gpu_ids) > operations.CUDA_SHARED_MEM_DIM_X or opt.node_num > operations.CUDA_SHARED_MEM_DIM_Y:
        #     print('--- WARNING: batch_size or node_num larger than pre-defined cuda shared memory array size. '
        #           'Please modify CUDA_SHARED_MEM_DIM_X and CUDA_SHARED_MEM_DIM_Y in models/operations.py')

    def global_step_inc(self, delta):
        self.global_step += delta

    def load_model(self, model_path):
        self.detector.load_state_dict(
            pytorch_helper.model_state_dict_convert_auto(
                torch.load(
                    model_path,
                    map_location='cpu'), self.opt.gpu_ids))

    # 这里进行数据的填装
    def set_input(self,
                  pc, pc2, target):
        self.pc.resize_(pc.size()).copy_(pc).detach()
        self.pc2.resize_(pc2.size()).copy_(pc2).detach()
        self.target.resize_(target.size()).copy_(target).detach()

    def forward(self,
                pc, pc2):
        return self.detector(pc,pc2)

    def inference_pass(self):
        N = self.pc.size(2)
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(self.opt.img_W / self.opt.img_fine_resolution_scale)
        img_H_fine_res = int(self.opt.img_H / self.opt.img_fine_resolution_scale)

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # BLx2xN, BLxKxN
        coarse_scores, fine_scores = self.forward(self.pc, self.pc2)
        K = fine_scores.size(1)
        assert K == img_W_fine_res * img_H_fine_res
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        _, coarse_prediction = torch.max(coarse_scores, dim=1, keepdim=False)  # BLxN
        _, fine_prediction = torch.max(fine_scores, dim=1, keepdim=False)  # BLxN
        return coarse_prediction, fine_prediction

    ###前向传播
    def foraward_pass(self):
        N = self.pc.size(2)
        # print('n = ', N)
        # Ma, Mb = self.opt.node_a_num, self.opt.node_b_num
        # B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        # img_W_fine_res = int(round(W / self.opt.img_fine_resolution_scale))
        # img_H_fine_res = int(round(H / self.opt.img_fine_resolution_scale))

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # 重点关注这个模块，是特征提取模块   将视觉特征和图像都送进去处理了
        # Bx2xN, BxLxN
        # print("the pc shape is: ",self.pc.shape)
        # print("the instesity shape is: ",self.intensity.shape)
        # print("the sn shape is: ",self.sn.shape)
        prediction = self.forward(self.pc, self.pc2)
        # L = fine_scores.size(1)
        # print(coarse_scores)
        # print('the coarse_scores shape is ', coarse_scores.shape) # ([8, 2, 20480])
        # print('the fine_scores shape is ', fine_scores.shape) # ([8, 80, 20480])
        # print('the score shape is ', prediction.shape) # ([8, 1])
        # assert L == img_W_fine_res * img_H_fine_res
        # build loss -------------------------------------------------------------------------------------
        # 'cuda:'+str(1)+','str(2)+str(3)
        loss = torch.mean(torch.nn.functional.binary_cross_entropy(prediction, self.target))
        # loss = coarse_loss + fine_loss
        # build loss -------------------------------------------------------------------------------------

        # get accuracy
        # get predictions for visualization
        # class_accuracy = torch.sum(torch.eq(self.target, prediction).to(dtype=torch.float)) /  B
        pred_batch = prediction.cpu().detach().numpy().reshape(-1)
        gt_batch = self.target.cpu().detach().numpy().reshape(-1)

        loss_dict = {'loss': loss}
        # accuracy_dict = {'class_accuracy': class_accuracy}

        # debug
        if self.opt.is_debug:
            print(loss_dict)
            # print(accuracy_dict)
        # print(loss_dict)

        return loss_dict, pred_batch, gt_batch

    def optimize(self):
        self.detector.train()
        self.detector.zero_grad()
        # time1 = time.time()
        self.train_loss_dict,_,_ = self.foraward_pass()
        # time2 = time.time()
        # print("the foraward cost : ", time2 - time1)
        self.train_loss_dict['loss'].backward()
        # time3 = time.time()
        # print("the backward cost : ", time3 - time2)
        self.optimizer_detector.step()

    def test_model(self):
        self.detector.eval()
        with torch.no_grad():
            self.test_loss_dict, self.pred_batch, self.gt_batch = self.foraward_pass()

    def freeze_model(self):
        print("!!! WARNING: Model freezed.")
        for p in self.detector.parameters():
            p.requires_grad = False

    def run_model(self, pc, intensity, sn, node, img):
        self.detector.eval()
        with torch.no_grad():
            raise Exception("Not implemented.")

    def tensor_dict_to_float_dict(self, tensor_dict):
        float_dict = {}
        for key, value in tensor_dict.items():
            if type(value) == torch.Tensor:
                float_dict[key] = value.item()
            else:
                float_dict[key] = value
        return float_dict

    def get_current_errors(self):
        return self.tensor_dict_to_float_dict(self.train_loss_dict), \
               self.tensor_dict_to_float_dict(self.test_loss_dict)

    def get_current_accuracy(self):
        return self.tensor_dict_to_float_dict(self.train_accuracy), \
               self.tensor_dict_to_float_dict(self.test_accuracy)

    def get_current_batch(self):
        return self.pred_batch, self.gt_batch

    @staticmethod
    def print_loss_dict(loss_dict, accuracy_dict=None, duration=-1):
        output = 'Per sample time: %.4f - ' % (duration)
        for key, value in loss_dict.items():
            output += '%s: %.4f, ' % (key, value)
        if accuracy_dict is not None:
            for key, value in accuracy_dict.items():
                output += '%s: %.4f, ' % (key, value)
        print(output)

    def save_network(self, network, save_filename):
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # detector
        lr_detector = self.old_lr_detector * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in self.optimizer_detector.param_groups:
            param_group['lr'] = lr_detector
        print('update detector learning rate: %f -> %f' % (self.old_lr_detector, lr_detector))
        self.old_lr_detector = lr_detector

    # visualization with tensorboard, pytorch 1.2
    def write_loss(self):
        train_loss_dict, test_loss_dict = self.get_current_errors()
        self.writer.add_scalars('train_loss',
                                train_loss_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_loss',
                                test_loss_dict,
                                global_step=self.global_step)

    def write_accuracy(self):
        train_acc_dict, test_acc_dict = self.get_current_accuracy()
        self.writer.add_scalars('train_accuracy',
                                train_acc_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_accuracy',
                                test_acc_dict,
                                global_step=self.global_step)

    def write_pc_label(self, pc, labels, title):
        """
        :param pc: Bx3xN
        :param labels: BxN
        :param title: string
        :return:
        """
        pc_np = pc.detach().cpu().numpy()  # Bx3xN
        labels_np = labels.detach().cpu().numpy()  # BxN

        B = pc_np.shape[0]

        visualization_fig_np_list = []
        vis_number = min(self.opt.vis_max_batch, B)
        for b in range(vis_number):
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(pc_np[b, 0, :], pc_np[b, 2, :],
                        c=labels_np[b, :],
                        cmap='jet',
                        marker='o',
                        s=1)
            fig.tight_layout()
            plt.axis('equal')
            fig_img_np = vis_tools.fig_to_np(fig)
            plt.close(fig)
            visualization_fig_np_list.append(fig_img_np)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image(title, imgs_vis_grid, global_step=self.global_step)

    def write_img(self):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW

        vis_number = min(self.opt.vis_max_batch, B)
        imgs_vis = imgs[0:vis_number, ...]  # Bx3xHxW
        imgs_vis_grid = torchvision.utils.make_grid(imgs_vis, nrow=2)
        self.writer.add_image('image', imgs_vis_grid, global_step=self.global_step)

    def write_classification_visualization(self,
                                           pc_pxpy,
                                           coarse_predictions, fine_predictions,
                                           coarse_labels, fine_labels,
                                           t_ij):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW
        imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # BxHxWx3

        pc_pxpy_np = pc_pxpy.cpu().numpy()  # Bx2xN
        coarse_predictions_np = coarse_predictions.detach().cpu().numpy()  # BxN
        fine_predictions_np = fine_predictions.detach().cpu().numpy()  # BxN
        coarse_labels_np = coarse_labels.detach().cpu().numpy()  # BxN
        fine_labels_np = fine_labels.detach().cpu().numpy()  # BxN

        t_ij_np = t_ij.cpu().numpy()  # Bx3

        vis_number = min(self.opt.vis_max_batch, B)
        visualization_fig_np_list = []
        for b in range(vis_number):
            img_b = imgs_np[b, ...]  # HxWx3
            pc_pxpy_b = pc_pxpy_np[b, ...]  # 2xN
            coarse_prediction_b = coarse_predictions_np[b, :]  # N
            fine_predictions_b = fine_predictions_np[b, :]  # N
            coarse_labels_b = coarse_labels_np[b, :]  # N
            fine_labels_b = fine_labels_np[b, :]  # N

            vis_img = vis_tools.get_classification_visualization(pc_pxpy_b,
                                                                 coarse_prediction_b, fine_predictions_b,
                                                                 coarse_labels_b, fine_labels_b,
                                                                 img_b,
                                                                 img_fine_resolution_scale=self.opt.img_fine_resolution_scale,
                                                                 H_delta=100, W_delta=100,
                                                                 circle_size=1,
                                                                 t_ij_np=t_ij_np[b, :])
            visualization_fig_np_list.append(vis_img)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image("Classification Visualization", imgs_vis_grid, global_step=self.global_step)

class MatchBase():
    def __init__(self, opt: Options, writer):
        self.opt = opt
        self.writer = writer
        self.global_step = 0

        # 这个东西是一个关键点的检测？
        self.detector = matchnet.matchnet(num_points=5120).to(self.opt.device)
        # self.coarse_ce_criteria = nn.CrossEntropyLoss()
        self.coarse_ce_criteria = focal_loss.FocalLoss(alpha=0.5, gamma=2, reduction='mean')
        self.fine_ce_criteria = nn.CrossEntropyLoss()

        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.detector = nn.DataParallel(self.detector, device_ids=opt.gpu_ids)
            # self.coarse_ce_criteria = nn.DataParallel(self.coarse_ce_criteria, device_ids=opt.gpu_ids)
            # self.fine_ce_criteria = nn.DataParallel(self.fine_ce_criteria, device_ids=opt.gpu_ids)


        # learning rate_control
        self.old_lr_detector = self.opt.lr
        self.optimizer_detector = torch.optim.Adam(self.detector.parameters(),
                                                   lr=self.old_lr_detector,
                                                   betas=(0.9, 0.999),
                                                   weight_decay=0.0005)

        # place holder for GPU tensors
        self.pc = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.intensity = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.label = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.sn = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.node_a = torch.empty(self.opt.batch_size, 3, self.opt.node_a_num, dtype=torch.float, device=self.opt.device)
        self.node_b = torch.empty(self.opt.batch_size, 3, self.opt.node_b_num, dtype=torch.float, device=self.opt.device)
        self.P = torch.empty(self.opt.batch_size, 3, 4, dtype=torch.float, device=self.opt.device)
        self.img = torch.empty(self.opt.batch_size, 3, self.opt.img_H, self.opt.img_W, dtype=torch.float, device=self.opt.device)
        self.K = torch.empty(self.opt.batch_size, 3, 3, dtype=torch.float, device=self.opt.device)
        self.target = torch.empty(self.opt.batch_size, 1, dtype=torch.float, device=self.opt.device)

        # record the train / test loss and accuracy
        self.train_loss_dict = {}
        self.test_loss_dict = {}

        self.pred_batch = {}
        self.gt_batch = {}

        # pre-cautions for shared memory usage
        # if opt.batch_size * 2 / len(opt.gpu_ids) > operations.CUDA_SHARED_MEM_DIM_X or opt.node_num > operations.CUDA_SHARED_MEM_DIM_Y:
        #     print('--- WARNING: batch_size or node_num larger than pre-defined cuda shared memory array size. '
        #           'Please modify CUDA_SHARED_MEM_DIM_X and CUDA_SHARED_MEM_DIM_Y in models/operations.py')

    def global_step_inc(self, delta):
        self.global_step += delta

    def load_model(self, model_path):
        self.detector.load_state_dict(
            pytorch_helper.model_state_dict_convert_auto(
                torch.load(
                    model_path,
                    map_location='cpu'), self.opt.gpu_ids))

    # 这里进行数据的填装
    def set_input(self,
                  pc, intensity,sn,label,node_a, node_b,
                  P,
                  img, K, target):
        self.pc.resize_(pc.size()).copy_(pc).detach()
        self.intensity.resize_(intensity.size()).copy_(intensity).detach()
        self.label.resize_(label.size()).copy_(label).detach()
        self.sn.resize_(sn.size()).copy_(sn).detach()
        self.node_a.resize_(node_a.size()).copy_(node_a).detach()
        self.node_b.resize_(node_b.size()).copy_(node_b).detach()
        self.P.resize_(P.size()).copy_(P).detach()
        self.img.resize_(img.size()).copy_(img).detach()
        self.K.resize_(K.size()).copy_(K).detach()
        self.target.resize_(target.size()).copy_(target).detach()

    def forward(self,
                pc, img):
        return self.detector(pc,img)

    def inference_pass(self):
        N = self.pc.size(2)
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(self.opt.img_W / self.opt.img_fine_resolution_scale)
        img_H_fine_res = int(self.opt.img_H / self.opt.img_fine_resolution_scale)

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # BLx2xN, BLxKxN
        coarse_scores, fine_scores = self.forward(self.pc, self.img)
        K = fine_scores.size(1)
        assert K == img_W_fine_res * img_H_fine_res
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        _, coarse_prediction = torch.max(coarse_scores, dim=1, keepdim=False)  # BLxN
        _, fine_prediction = torch.max(fine_scores, dim=1, keepdim=False)  # BLxN
        return coarse_prediction, fine_prediction

    ###前向传播
    def foraward_pass(self):
        N = self.pc.size(2)
        # print('n = ', N)
        Ma, Mb = self.opt.node_a_num, self.opt.node_b_num
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(round(W / self.opt.img_fine_resolution_scale))
        img_H_fine_res = int(round(H / self.opt.img_fine_resolution_scale))

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # 重点关注这个模块，是特征提取模块   将视觉特征和图像都送进去处理了
        # Bx2xN, BxLxN
        # print("the pc shape is: ",self.pc.shape)
        # print("the instesity shape is: ",self.intensity.shape)
        # print("the sn shape is: ",self.sn.shape)
        prediction = self.forward(self.pc, self.img[:,:3,:,:])
        # L = fine_scores.size(1)
        # print(coarse_scores)
        # print('the coarse_scores shape is ', coarse_scores.shape) # ([8, 2, 20480])
        # print('the fine_scores shape is ', fine_scores.shape) # ([8, 80, 20480])
        # print('the score shape is ', prediction.shape) # ([8, 1])
        # assert L == img_W_fine_res * img_H_fine_res
        # build loss -------------------------------------------------------------------------------------
        # 'cuda:'+str(1)+','str(2)+str(3)
        loss = torch.mean(torch.nn.functional.binary_cross_entropy(prediction, self.target))
        # loss = coarse_loss + fine_loss
        # build loss -------------------------------------------------------------------------------------

        # get accuracy
        # get predictions for visualization
        # class_accuracy = torch.sum(torch.eq(self.target, prediction).to(dtype=torch.float)) /  B
        pred_batch = prediction.cpu().detach().numpy().reshape(-1)
        gt_batch = self.target.cpu().detach().numpy().reshape(-1)

        loss_dict = {'loss': loss}
        # accuracy_dict = {'class_accuracy': class_accuracy}

        # debug
        if self.opt.is_debug:
            print(loss_dict)
            # print(accuracy_dict)
        # print(loss_dict)

        return loss_dict, pred_batch, gt_batch

    def optimize(self):
        self.detector.train()
        self.detector.zero_grad()
        time1 = time.time()
        self.train_loss_dict,_,_ = self.foraward_pass()
        time2 = time.time()
        # print("the foraward cost : ", time2 - time1)
        self.train_loss_dict['loss'].backward()
        time3 = time.time()
        # print("the backward cost : ", time3 - time2)
        self.optimizer_detector.step()

    def test_model(self):
        self.detector.eval()
        with torch.no_grad():
            self.test_loss_dict, self.pred_batch, self.gt_batch = self.foraward_pass()

    def freeze_model(self):
        print("!!! WARNING: Model freezed.")
        for p in self.detector.parameters():
            p.requires_grad = False

    def run_model(self, pc, intensity, sn, node, img):
        self.detector.eval()
        with torch.no_grad():
            raise Exception("Not implemented.")

    def tensor_dict_to_float_dict(self, tensor_dict):
        float_dict = {}
        for key, value in tensor_dict.items():
            if type(value) == torch.Tensor:
                float_dict[key] = value.item()
            else:
                float_dict[key] = value
        return float_dict

    def get_current_errors(self):
        return self.tensor_dict_to_float_dict(self.train_loss_dict), \
               self.tensor_dict_to_float_dict(self.test_loss_dict)

    def get_current_accuracy(self):
        return self.tensor_dict_to_float_dict(self.train_accuracy), \
               self.tensor_dict_to_float_dict(self.test_accuracy)

    def get_current_batch(self):
        return self.pred_batch, self.gt_batch

    @staticmethod
    def print_loss_dict(loss_dict, accuracy_dict=None, duration=-1):
        output = 'Per sample time: %.4f - ' % (duration)
        for key, value in loss_dict.items():
            output += '%s: %.4f, ' % (key, value)
        if accuracy_dict is not None:
            for key, value in accuracy_dict.items():
                output += '%s: %.4f, ' % (key, value)
        print(output)

    def save_network(self, network, save_filename):
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # detector
        lr_detector = self.old_lr_detector * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in self.optimizer_detector.param_groups:
            param_group['lr'] = lr_detector
        print('update detector learning rate: %f -> %f' % (self.old_lr_detector, lr_detector))
        self.old_lr_detector = lr_detector

    # visualization with tensorboard, pytorch 1.2
    def write_loss(self):
        train_loss_dict, test_loss_dict = self.get_current_errors()
        self.writer.add_scalars('train_loss',
                                train_loss_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_loss',
                                test_loss_dict,
                                global_step=self.global_step)

    def write_accuracy(self):
        train_acc_dict, test_acc_dict = self.get_current_accuracy()
        self.writer.add_scalars('train_accuracy',
                                train_acc_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_accuracy',
                                test_acc_dict,
                                global_step=self.global_step)

    def write_pc_label(self, pc, labels, title):
        """
        :param pc: Bx3xN
        :param labels: BxN
        :param title: string
        :return:
        """
        pc_np = pc.detach().cpu().numpy()  # Bx3xN
        labels_np = labels.detach().cpu().numpy()  # BxN

        B = pc_np.shape[0]

        visualization_fig_np_list = []
        vis_number = min(self.opt.vis_max_batch, B)
        for b in range(vis_number):
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(pc_np[b, 0, :], pc_np[b, 2, :],
                        c=labels_np[b, :],
                        cmap='jet',
                        marker='o',
                        s=1)
            fig.tight_layout()
            plt.axis('equal')
            fig_img_np = vis_tools.fig_to_np(fig)
            plt.close(fig)
            visualization_fig_np_list.append(fig_img_np)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image(title, imgs_vis_grid, global_step=self.global_step)

    def write_img(self):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW

        vis_number = min(self.opt.vis_max_batch, B)
        imgs_vis = imgs[0:vis_number, ...]  # Bx3xHxW
        imgs_vis_grid = torchvision.utils.make_grid(imgs_vis, nrow=2)
        self.writer.add_image('image', imgs_vis_grid, global_step=self.global_step)

    def write_classification_visualization(self,
                                           pc_pxpy,
                                           coarse_predictions, fine_predictions,
                                           coarse_labels, fine_labels,
                                           t_ij):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW
        imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # BxHxWx3

        pc_pxpy_np = pc_pxpy.cpu().numpy()  # Bx2xN
        coarse_predictions_np = coarse_predictions.detach().cpu().numpy()  # BxN
        fine_predictions_np = fine_predictions.detach().cpu().numpy()  # BxN
        coarse_labels_np = coarse_labels.detach().cpu().numpy()  # BxN
        fine_labels_np = fine_labels.detach().cpu().numpy()  # BxN

        t_ij_np = t_ij.cpu().numpy()  # Bx3

        vis_number = min(self.opt.vis_max_batch, B)
        visualization_fig_np_list = []
        for b in range(vis_number):
            img_b = imgs_np[b, ...]  # HxWx3
            pc_pxpy_b = pc_pxpy_np[b, ...]  # 2xN
            coarse_prediction_b = coarse_predictions_np[b, :]  # N
            fine_predictions_b = fine_predictions_np[b, :]  # N
            coarse_labels_b = coarse_labels_np[b, :]  # N
            fine_labels_b = fine_labels_np[b, :]  # N

            vis_img = vis_tools.get_classification_visualization(pc_pxpy_b,
                                                                 coarse_prediction_b, fine_predictions_b,
                                                                 coarse_labels_b, fine_labels_b,
                                                                 img_b,
                                                                 img_fine_resolution_scale=self.opt.img_fine_resolution_scale,
                                                                 H_delta=100, W_delta=100,
                                                                 circle_size=1,
                                                                 t_ij_np=t_ij_np[b, :])
            visualization_fig_np_list.append(vis_img)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image("Classification Visualization", imgs_vis_grid, global_step=self.global_step)


class MMClassifer():
    def __init__(self, opt: Options, writer):
        self.opt = opt
        self.writer = writer
        self.global_step = 0

        # 这个东西是一个关键点的检测？
        self.detector = networks_united.KeypointDetector_Base(self.opt).to(self.opt.device)
        # self.coarse_ce_criteria = nn.CrossEntropyLoss()
        self.coarse_ce_criteria = focal_loss.FocalLoss(alpha=0.5, gamma=2, reduction='mean')
        self.fine_ce_criteria = nn.CrossEntropyLoss()

        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.detector = nn.DataParallel(self.detector, device_ids=opt.gpu_ids)
            # self.coarse_ce_criteria = nn.DataParallel(self.coarse_ce_criteria, device_ids=opt.gpu_ids)
            # self.fine_ce_criteria = nn.DataParallel(self.fine_ce_criteria, device_ids=opt.gpu_ids)


        # learning rate_control
        self.old_lr_detector = self.opt.lr
        self.optimizer_detector = torch.optim.Adam(self.detector.parameters(),
                                                   lr=self.old_lr_detector,
                                                   betas=(0.9, 0.999),
                                                   weight_decay=0.0005)

        # place holder for GPU tensors
        self.pc = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.intensity = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.sn = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.node_a = torch.empty(self.opt.batch_size, 3, self.opt.node_a_num, dtype=torch.float, device=self.opt.device)
        self.node_b = torch.empty(self.opt.batch_size, 3, self.opt.node_b_num, dtype=torch.float, device=self.opt.device)
        self.P = torch.empty(self.opt.batch_size, 3, 4, dtype=torch.float, device=self.opt.device)
        self.img = torch.empty(self.opt.batch_size, 3, self.opt.img_H, self.opt.img_W, dtype=torch.float, device=self.opt.device)
        self.K = torch.empty(self.opt.batch_size, 3, 3, dtype=torch.float, device=self.opt.device)
        self.target = torch.empty(self.opt.batch_size, 1, dtype=torch.float, device=self.opt.device)

        # record the train / test loss and accuracy
        self.train_loss_dict = {}
        self.test_loss_dict = {}

        self.pred_batch = {}
        self.gt_batch = {}

        # pre-cautions for shared memory usage
        # if opt.batch_size * 2 / len(opt.gpu_ids) > operations.CUDA_SHARED_MEM_DIM_X or opt.node_num > operations.CUDA_SHARED_MEM_DIM_Y:
        #     print('--- WARNING: batch_size or node_num larger than pre-defined cuda shared memory array size. '
        #           'Please modify CUDA_SHARED_MEM_DIM_X and CUDA_SHARED_MEM_DIM_Y in models/operations.py')

    def global_step_inc(self, delta):
        self.global_step += delta

    def load_model(self, model_path):
        self.detector.load_state_dict(
            pytorch_helper.model_state_dict_convert_auto(
                torch.load(
                    model_path,
                    map_location='cpu'), self.opt.gpu_ids))

    # 这里进行数据的填装
    def set_input(self,
                  pc, intensity, sn, node_a, node_b,
                  P,
                  img, K, target):
        self.pc.resize_(pc.size()).copy_(pc).detach()
        self.intensity.resize_(intensity.size()).copy_(intensity).detach()
        self.sn.resize_(sn.size()).copy_(sn).detach()
        self.node_a.resize_(node_a.size()).copy_(node_a).detach()
        self.node_b.resize_(node_b.size()).copy_(node_b).detach()
        self.P.resize_(P.size()).copy_(P).detach()
        self.img.resize_(img.size()).copy_(img).detach()
        self.K.resize_(K.size()).copy_(K).detach()
        self.target.resize_(target.size()).copy_(target).detach()

    def forward(self,
                pc, intensity, sn, node_a, node_b,
                img):
        return self.detector(pc, intensity, sn, node_a, node_b, img)

    def inference_pass(self):
        N = self.pc.size(2)
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(self.opt.img_W / self.opt.img_fine_resolution_scale)
        img_H_fine_res = int(self.opt.img_H / self.opt.img_fine_resolution_scale)

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # BLx2xN, BLxKxN
        coarse_scores, fine_scores = self.forward(self.pc, self.intensity, self.sn, self.node_a, self.node_b,
                                                  self.img)
        K = fine_scores.size(1)
        assert K == img_W_fine_res * img_H_fine_res
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        _, coarse_prediction = torch.max(coarse_scores, dim=1, keepdim=False)  # BLxN
        _, fine_prediction = torch.max(fine_scores, dim=1, keepdim=False)  # BLxN
        return coarse_prediction, fine_prediction

    ###前向传播
    def foraward_pass(self):
        N = self.pc.size(2)
        # print('n = ', N)
        Ma, Mb = self.opt.node_a_num, self.opt.node_b_num
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(round(W / self.opt.img_fine_resolution_scale))
        img_H_fine_res = int(round(H / self.opt.img_fine_resolution_scale))

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # 重点关注这个模块，是特征提取模块   将视觉特征和图像都送进去处理了
        # Bx2xN, BxLxN
        prediction = self.forward(self.pc, self.intensity, self.sn, self.node_a, self.node_b,
                                                  self.img)
        # L = fine_scores.size(1)
        # print(coarse_scores)
        # print('the coarse_scores shape is ', coarse_scores.shape) # ([8, 2, 20480])
        # print('the fine_scores shape is ', fine_scores.shape) # ([8, 80, 20480])
        # print('the score shape is ', prediction.shape) # ([8, 1])
        # assert L == img_W_fine_res * img_H_fine_res
        # build loss -------------------------------------------------------------------------------------
        # 'cuda:'+str(1)+','str(2)+str(3)
        loss = torch.mean(torch.nn.functional.binary_cross_entropy(prediction, self.target))
        # loss = coarse_loss + fine_loss
        # build loss -------------------------------------------------------------------------------------

        # get accuracy
        # get predictions for visualization
        # class_accuracy = torch.sum(torch.eq(self.target, prediction).to(dtype=torch.float)) /  B
        pred_batch = prediction.cpu().detach().numpy().reshape(-1)
        gt_batch = self.target.cpu().detach().numpy().reshape(-1)

        loss_dict = {'loss': loss}
        # accuracy_dict = {'class_accuracy': class_accuracy}

        # debug
        if self.opt.is_debug:
            print(loss_dict)
            # print(accuracy_dict)
        # print(loss_dict)

        return loss_dict, pred_batch, gt_batch

    def optimize(self):
        self.detector.train()
        self.detector.zero_grad()
        self.train_loss_dict,_,_ = self.foraward_pass()
        self.train_loss_dict['loss'].backward()
        self.optimizer_detector.step()

    def test_model(self):
        self.detector.eval()
        with torch.no_grad():
            self.test_loss_dict, self.pred_batch, self.gt_batch = self.foraward_pass()

    def freeze_model(self):
        print("!!! WARNING: Model freezed.")
        for p in self.detector.parameters():
            p.requires_grad = False

    def run_model(self, pc, intensity, sn, node, img):
        self.detector.eval()
        with torch.no_grad():
            raise Exception("Not implemented.")

    def tensor_dict_to_float_dict(self, tensor_dict):
        float_dict = {}
        for key, value in tensor_dict.items():
            if type(value) == torch.Tensor:
                float_dict[key] = value.item()
            else:
                float_dict[key] = value
        return float_dict

    def get_current_errors(self):
        return self.tensor_dict_to_float_dict(self.train_loss_dict), \
               self.tensor_dict_to_float_dict(self.test_loss_dict)

    def get_current_accuracy(self):
        return self.tensor_dict_to_float_dict(self.train_accuracy), \
               self.tensor_dict_to_float_dict(self.test_accuracy)

    def get_current_batch(self):
        return self.pred_batch, self.gt_batch

    @staticmethod
    def print_loss_dict(loss_dict, accuracy_dict=None, duration=-1):
        output = 'Per sample time: %.4f - ' % (duration)
        for key, value in loss_dict.items():
            output += '%s: %.4f, ' % (key, value)
        if accuracy_dict is not None:
            for key, value in accuracy_dict.items():
                output += '%s: %.4f, ' % (key, value)
        print(output)

    def save_network(self, network, save_filename):
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # detector
        lr_detector = self.old_lr_detector * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in self.optimizer_detector.param_groups:
            param_group['lr'] = lr_detector
        print('update detector learning rate: %f -> %f' % (self.old_lr_detector, lr_detector))
        self.old_lr_detector = lr_detector

    # visualization with tensorboard, pytorch 1.2
    def write_loss(self):
        train_loss_dict, test_loss_dict = self.get_current_errors()
        self.writer.add_scalars('train_loss',
                                train_loss_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_loss',
                                test_loss_dict,
                                global_step=self.global_step)

    def write_accuracy(self):
        train_acc_dict, test_acc_dict = self.get_current_accuracy()
        self.writer.add_scalars('train_accuracy',
                                train_acc_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_accuracy',
                                test_acc_dict,
                                global_step=self.global_step)

    def write_pc_label(self, pc, labels, title):
        """
        :param pc: Bx3xN
        :param labels: BxN
        :param title: string
        :return:
        """
        pc_np = pc.detach().cpu().numpy()  # Bx3xN
        labels_np = labels.detach().cpu().numpy()  # BxN

        B = pc_np.shape[0]

        visualization_fig_np_list = []
        vis_number = min(self.opt.vis_max_batch, B)
        for b in range(vis_number):
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(pc_np[b, 0, :], pc_np[b, 2, :],
                        c=labels_np[b, :],
                        cmap='jet',
                        marker='o',
                        s=1)
            fig.tight_layout()
            plt.axis('equal')
            fig_img_np = vis_tools.fig_to_np(fig)
            plt.close(fig)
            visualization_fig_np_list.append(fig_img_np)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image(title, imgs_vis_grid, global_step=self.global_step)

    def write_img(self):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW

        vis_number = min(self.opt.vis_max_batch, B)
        imgs_vis = imgs[0:vis_number, ...]  # Bx3xHxW
        imgs_vis_grid = torchvision.utils.make_grid(imgs_vis, nrow=2)
        self.writer.add_image('image', imgs_vis_grid, global_step=self.global_step)

    def write_classification_visualization(self,
                                           pc_pxpy,
                                           coarse_predictions, fine_predictions,
                                           coarse_labels, fine_labels,
                                           t_ij):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW
        imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # BxHxWx3

        pc_pxpy_np = pc_pxpy.cpu().numpy()  # Bx2xN
        coarse_predictions_np = coarse_predictions.detach().cpu().numpy()  # BxN
        fine_predictions_np = fine_predictions.detach().cpu().numpy()  # BxN
        coarse_labels_np = coarse_labels.detach().cpu().numpy()  # BxN
        fine_labels_np = fine_labels.detach().cpu().numpy()  # BxN

        t_ij_np = t_ij.cpu().numpy()  # Bx3

        vis_number = min(self.opt.vis_max_batch, B)
        visualization_fig_np_list = []
        for b in range(vis_number):
            img_b = imgs_np[b, ...]  # HxWx3
            pc_pxpy_b = pc_pxpy_np[b, ...]  # 2xN
            coarse_prediction_b = coarse_predictions_np[b, :]  # N
            fine_predictions_b = fine_predictions_np[b, :]  # N
            coarse_labels_b = coarse_labels_np[b, :]  # N
            fine_labels_b = fine_labels_np[b, :]  # N

            vis_img = vis_tools.get_classification_visualization(pc_pxpy_b,
                                                                 coarse_prediction_b, fine_predictions_b,
                                                                 coarse_labels_b, fine_labels_b,
                                                                 img_b,
                                                                 img_fine_resolution_scale=self.opt.img_fine_resolution_scale,
                                                                 H_delta=100, W_delta=100,
                                                                 circle_size=1,
                                                                 t_ij_np=t_ij_np[b, :])
            visualization_fig_np_list.append(vis_img)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image("Classification Visualization", imgs_vis_grid, global_step=self.global_step)

class MMClassiferCoarse():
    def __init__(self, opt: Options, writer):
        self.opt = opt
        self.writer = writer
        self.global_step = 0

        self.detector = networks_united.KeypointDetector(self.opt).to(self.opt.device)
        # self.coarse_ce_criteria = nn.CrossEntropyLoss()
        self.coarse_ce_criteria = focal_loss.FocalLoss(alpha=0.5, gamma=2, reduction='mean')
        # self.fine_ce_criteria = nn.CrossEntropyLoss()

        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.detector = nn.DataParallel(self.detector, device_ids=opt.gpu_ids)
            # self.coarse_ce_criteria = nn.DataParallel(self.coarse_ce_criteria, device_ids=opt.gpu_ids)
            # self.fine_ce_criteria = nn.DataParallel(self.fine_ce_criteria, device_ids=opt.gpu_ids)

        # learning rate_control
        self.old_lr_detector = self.opt.lr
        self.optimizer_detector = torch.optim.Adam(self.detector.parameters(),
                                                   lr=self.old_lr_detector,
                                                   betas=(0.9, 0.999),
                                                   weight_decay=0)

        # place holder for GPU tensors
        self.pc = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.intensity = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float,
                                     device=self.opt.device)
        self.sn = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.node_a = torch.empty(self.opt.batch_size, 3, self.opt.node_a_num, dtype=torch.float,
                                  device=self.opt.device)
        self.node_b = torch.empty(self.opt.batch_size, 3, self.opt.node_b_num, dtype=torch.float,
                                  device=self.opt.device)
        self.P = torch.empty(self.opt.batch_size, 3, 4, dtype=torch.float, device=self.opt.device)
        self.img = torch.empty(self.opt.batch_size, 3, self.opt.img_H, self.opt.img_W, dtype=torch.float,
                               device=self.opt.device)
        self.K = torch.empty(self.opt.batch_size, 3, 3, dtype=torch.float, device=self.opt.device)

        # record the train / test loss and accuracy
        self.train_loss_dict = {}
        self.test_loss_dict = {}

        self.train_accuracy = {}
        self.test_accuracy = {}

        # pre-cautions for shared memory usage
        # if opt.batch_size * 2 / len(opt.gpu_ids) > operations.CUDA_SHARED_MEM_DIM_X or opt.node_num > operations.CUDA_SHARED_MEM_DIM_Y:
        #     print('--- WARNING: batch_size or node_num larger than pre-defined cuda shared memory array size. '
        #           'Please modify CUDA_SHARED_MEM_DIM_X and CUDA_SHARED_MEM_DIM_Y in models/operations.py')

    def global_step_inc(self, delta):
        self.global_step += delta

    def load_model(self, model_path):
        self.detector.load_state_dict(
            pytorch_helper.model_state_dict_convert_auto(
                torch.load(
                    model_path,
                    map_location='cpu'), self.opt.gpu_ids))

    def set_input(self,
                  pc, intensity, sn, node_a, node_b,
                  P,
                  img, K):
        self.pc.resize_(pc.size()).copy_(pc).detach()
        self.intensity.resize_(intensity.size()).copy_(intensity).detach()
        self.sn.resize_(sn.size()).copy_(sn).detach()
        self.node_a.resize_(node_a.size()).copy_(node_a).detach()
        self.node_b.resize_(node_b.size()).copy_(node_b).detach()
        self.P.resize_(P.size()).copy_(P).detach()
        self.img.resize_(img.size()).copy_(img).detach()
        self.K.resize_(K.size()).copy_(K).detach()

    def forward(self,
                pc, intensity, sn, node_a, node_b,
                img):
        return self.detector(pc, intensity, sn, node_a, node_b, img)

    def inference_pass(self):
        N = self.pc.size(2)
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # BLx2xN, BLxKxN
        coarse_scores = self.forward(self.pc, self.intensity, self.sn, self.node_a, self.node_b,
                                                  self.img)
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        _, coarse_prediction = torch.max(coarse_scores, dim=1, keepdim=False)  # BLxN
        return coarse_prediction

    def foraward_pass(self):
        N = self.pc.size(2)
        Ma, Mb = self.opt.node_a_num, self.opt.node_b_num
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # Bx2xN, BxLxN
        coarse_scores = self.forward(self.pc, self.intensity, self.sn, self.node_a, self.node_b,
                                                  self.img)
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        # project points onto image to get ground truth labels for both coarse and fine resolution
        pc_homo = torch.cat((self.pc,
                             torch.ones((B, 1, N), dtype=torch.float32, device=self.pc.device)),
                            dim=1)  # Bx4xN
        P_pc_homo = torch.matmul(self.P, pc_homo)  # Bx3xN
        KP_pc_homo = torch.matmul(self.K, P_pc_homo)  # Bx3xN
        KP_pc_pxpy = KP_pc_homo[:, 0:2, :] / KP_pc_homo[:, 2:3, :]  # Bx2xN

        x_inside_mask = (KP_pc_pxpy[:, 0:1, :] >= 0) \
                        & (KP_pc_pxpy[:, 0:1, :] <= W - 1)  # Bx1xN
        y_inside_mask = (KP_pc_pxpy[:, 1:2, :] >= 0) \
                        & (KP_pc_pxpy[:, 1:2, :] <= H - 1)  # Bx1xN
        z_inside_mask = KP_pc_homo[:, 2:3, :] > 0.1  # Bx1xN
        inside_mask = (x_inside_mask & y_inside_mask & z_inside_mask).squeeze(1)  # BxN

        # get coarse labels
        coarse_labels = inside_mask.to(dtype=torch.long)  # BxN

        # build loss -------------------------------------------------------------------------------------
        coarse_loss = self.coarse_ce_criteria(coarse_scores, coarse_labels) * self.opt.coarse_loss_alpha
        loss = coarse_loss
        # build loss -------------------------------------------------------------------------------------

        # get accuracy
        # get predictions for visualization
        _, coarse_predictions = torch.max(coarse_scores, dim=1, keepdim=False)
        coarse_accuracy = torch.sum(torch.eq(coarse_labels, coarse_predictions).to(dtype=torch.float)) / (B * N)

        loss_dict = {'loss': loss,
                     'coarse': coarse_loss}
        vis_dict = {'pc': P_pc_homo,
                    'coarse_labels': coarse_labels,
                    'coarse_predictions': coarse_predictions,
                    'KP_pc_pxpy': KP_pc_pxpy}
        accuracy_dict = {'coarse_accuracy': coarse_accuracy}

        # debug
        if self.opt.is_debug:
            print(loss_dict)

        return loss_dict, vis_dict, accuracy_dict

    def optimize(self):
        self.detector.train()
        self.detector.zero_grad()
        self.train_loss_dict, self.train_visualization, self.train_accuracy = self.foraward_pass()
        self.train_loss_dict['loss'].backward()
        self.optimizer_detector.step()

    def test_model(self):
        self.detector.eval()
        with torch.no_grad():
            self.test_loss_dict, self.test_visualization, self.test_accuracy = self.foraward_pass()

    def freeze_model(self):
        print("!!! WARNING: Model freezed.")
        for p in self.detector.parameters():
            p.requires_grad = False

    def run_model(self, pc, intensity, sn, node, img):
        self.detector.eval()
        with torch.no_grad():
            raise Exception("Not implemented.")

    def tensor_dict_to_float_dict(self, tensor_dict):
        float_dict = {}
        for key, value in tensor_dict.items():
            if type(value) == torch.Tensor:
                float_dict[key] = value.item()
            else:
                float_dict[key] = value
        return float_dict

    def get_current_errors(self):
        return self.tensor_dict_to_float_dict(self.train_loss_dict), \
               self.tensor_dict_to_float_dict(self.test_loss_dict)

    def get_current_accuracy(self):
        return self.tensor_dict_to_float_dict(self.train_accuracy), \
               self.tensor_dict_to_float_dict(self.test_accuracy)

    @staticmethod
    def print_loss_dict(loss_dict, accuracy_dict=None, duration=-1):
        output = 'Per sample time: %.3f - ' % (duration)
        for key, value in loss_dict.items():
            output += '%s: %.2f, ' % (key, value)
        if accuracy_dict is not None:
            for key, value in accuracy_dict.items():
                output += '%s: %.2f, ' % (key, value)
        print(output)

    def save_network(self, network, save_filename):
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # detector
        lr_detector = self.old_lr_detector * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in self.optimizer_detector.param_groups:
            param_group['lr'] = lr_detector
        print('update detector learning rate: %f -> %f' % (self.old_lr_detector, lr_detector))
        self.old_lr_detector = lr_detector

    # visualization with tensorboard, pytorch 1.2
    def write_loss(self):
        train_loss_dict, test_loss_dict = self.get_current_errors()
        self.writer.add_scalars('train_loss',
                                train_loss_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_loss',
                                test_loss_dict,
                                global_step=self.global_step)

    def write_accuracy(self):
        train_acc_dict, test_acc_dict = self.get_current_accuracy()
        self.writer.add_scalars('train_accuracy',
                                train_acc_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_accuracy',
                                test_acc_dict,
                                global_step=self.global_step)

    def write_pc_label(self, pc, labels, title):
        """
        :param pc: Bx3xN
        :param labels: BxN
        :param title: string
        :return:
        """
        pc_np = pc.detach().cpu().numpy()  # Bx3xN
        labels_np = labels.detach().cpu().numpy()  # BxN

        B = pc_np.shape[0]

        visualization_fig_np_list = []
        vis_number = min(self.opt.vis_max_batch, B)
        for b in range(vis_number):
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(pc_np[b, 0, :], pc_np[b, 2, :],
                        c=labels_np[b, :],
                        cmap='jet',
                        marker='o',
                        s=1)
            fig.tight_layout()
            plt.axis('equal')
            fig_img_np = vis_tools.fig_to_np(fig)
            plt.close(fig)
            visualization_fig_np_list.append(fig_img_np)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image(title, imgs_vis_grid, global_step=self.global_step)

    def write_img(self):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW

        vis_number = min(self.opt.vis_max_batch, B)
        imgs_vis = imgs[0:vis_number, ...]  # Bx3xHxW
        imgs_vis_grid = torchvision.utils.make_grid(imgs_vis, nrow=2)
        self.writer.add_image('image', imgs_vis_grid, global_step=self.global_step)

    def write_classification_visualization(self,
                                           pc_pxpy,
                                           coarse_predictions,
                                           coarse_labels,
                                           t_ij):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW
        imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # BxHxWx3

        pc_pxpy_np = pc_pxpy.cpu().numpy()  # Bx2xN
        coarse_predictions_np = coarse_predictions.detach().cpu().numpy()  # BxN
        coarse_labels_np = coarse_labels.detach().cpu().numpy()  # BxN

        t_ij_np = t_ij.cpu().numpy()  # Bx3

        vis_number = min(self.opt.vis_max_batch, B)
        visualization_fig_np_list = []
        for b in range(vis_number):
            img_b = imgs_np[b, ...]  # HxWx3
            pc_pxpy_b = pc_pxpy_np[b, ...]  # 2xN
            coarse_prediction_b = coarse_predictions_np[b, :]  # N
            coarse_labels_b = coarse_labels_np[b, :] # N

            vis_img = vis_tools.get_classification_visualization_coarse(pc_pxpy_b,
                                                                 coarse_prediction_b,
                                                                 coarse_labels_b,
                                                                 img_b,
                                                                 H_delta=100, W_delta=100,
                                                                 circle_size=1,
                                                                 t_ij_np=t_ij_np[b, :])
            visualization_fig_np_list.append(vis_img)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image("Classification Visualization", imgs_vis_grid, global_step=self.global_step)
