import numpy as np
import math
import torch


class Options:
    def __init__(self):
        self.dataroot = 'dataset/kitti'
        self.issemantic = True
        self.output_path = 'eval/1_10_00'
        # self.dataroot = '/data/personal/jiaxin/datasets/kitti'
        # self.checkpoints_dir = 'checkpoints'
        self.checkpoints_dir = 'checkpoints_match_2'
        self.version = '1.10'
        self.is_debug = False #True
        self.is_fine_resolution = True
        self.is_remove_ground = False
        # self.accumulation_frame_num = 3
        self.accumulation_frame_num = 0
        self.accumulation_frame_skip = 6

        self.delta_ij_max = 40
        self.translation_max = 10

        self.crop_original_top_rows = 50
        self.img_scale = 0.5
        self.img_H = 160  # 320 * 0.5
        self.img_W = 512  # 1224 * 0.5
        # the fine resolution is img_H/scale x img_W/scale
        self.img_fine_resolution_scale = 32

        self.input_pt_num = 5120
        self.pc_min_range = -1.0
        self.pc_max_range = 80.0
        # 这里的node_a 和 node_b表示的是什么意思
        self.node_a_num = 30
        self.node_b_num = 30
        self.k_ab = 16
        self.k_interp_ab = 3
        self.k_interp_point_a = 3
        self.k_interp_point_b = 3
        self.node_num = 35

        # CAM coordinate
        self.P_tx_amplitude = 0
        self.P_ty_amplitude = 0
        self.P_tz_amplitude = 0
        self.P_Rx_amplitude = 0.0 * math.pi / 12.0
        self.P_Ry_amplitude = 2.0 * math.pi 
        self.P_Rz_amplitude = 0.0 * math.pi / 12.0
        self.dataloader_threads = 4

        self.epochs = 30
        self.batch_size = 64
        self.gpu_ids = [2,3]
        self.device = torch.device('cuda', self.gpu_ids[0])
        # self.device = torch.device('cuda', 0)
        self.normalization = 'batch'
        self.norm_momentum = 0.1
        self.activation = 'relu'
        self.lr = 0.0002
        self.lr_decay_step = 10
        self.lr_decay_scale = 0.5
        self.vis_max_batch = 4
        if self.is_fine_resolution:
            self.coarse_loss_alpha = 50
        else:
            self.coarse_loss_alpha = 1

        self.img_dim = 512
        self.embed_size = 512
        self.num_head_IMRC = 16
        self.hid_IMRC = 512
        self.raw_feature_norm_CMRC = "clipped_l2norm"
        self.lambda_softmax_CMRC = 4.
        self.hid_router = 512
        self.direction = 'i2t'
        self.filters_1 = 64
        self.filters_2 = 64
        self.filters_3 = 32
        self.tensor_neurons = 16
        self.bottle_neck_neurons = 16
        self.K = 5





