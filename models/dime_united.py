import torch
import torch.nn as nn

from models import networks_img
from models.mmcv.conv_module import ConvModule
from models import resnet
from models import networks_pc
from models import layers_common
from models import layers_pc
from models import InteractionModule
from kitti.options_clip import Options
from util import pytorch_helper


# 特征点检测器
class KeypointDetector(nn.Module):
    def __init__(self, opt: Options):
        super(KeypointDetector, self).__init__()
        self.opt = opt

        self.pc_encoder = networks_pc.PCEncoder(opt, Ca=64, Cb=512, Cg=512).to(self.opt.device)
        self.img_encoder = networks_img.ImageEncoder(self.opt).to(self.opt.device)
        self.itr_module = InteractionModule.InteractionModule(self.opt).to(self.opt.device)

        self.H_fine_res = int(round(self.opt.img_H / self.opt.img_fine_resolution_scale))
        self.W_fine_res = int(round(self.opt.img_W / self.opt.img_fine_resolution_scale))

        self.node_b_attention_pn = layers_pc.PointNet(256+512,
                                               [256, self.H_fine_res*self.W_fine_res],
                                               activation=self.opt.activation,
                                               normalization=self.opt.normalization,
                                               norm_momentum=opt.norm_momentum,
                                               norm_act_at_last=False)

        # in_channels: node_b_features + global_feature + image_s32_feature + image_global_feature
        self.node_b_pn = layers_pc.PointNet(256+512+512+512,
                                            [1024, 512, 512],
                                            activation=self.opt.activation,
                                            normalization=self.opt.normalization,
                                            norm_momentum=opt.norm_momentum,
                                            norm_act_at_last=False)

        self.node_a_attention_pn = layers_pc.PointNet(64 + 512,
                                                      [256, int(self.H_fine_res * self.W_fine_res * 4)],
                                                      activation=self.opt.activation,
                                                      normalization=self.opt.normalization,
                                                      norm_momentum=opt.norm_momentum,
                                                      norm_act_at_last=False)

        # in_channels: node_a_features + interpolated node_b_features
        self.node_a_pn = layers_pc.PointNet(64+256+512,
                                            [512, 128, 128],
                                            activation=self.opt.activation,
                                            normalization=self.opt.normalization,
                                            norm_momentum=opt.norm_momentum,
                                            norm_act_at_last=False)

        # final network for per-point labeling
        # in_channels: second_pn_out + interpolated node_a_features
        per_point_pn_in_channels = 32 + 64 + 128 + 512
        # per_point_pn_in_channels = 32 + 64 + 512 + 512
        if self.opt.is_fine_resolution:
            self.per_point_pn = layers_pc.PointNet(per_point_pn_in_channels,
                                                   [256, 256, 2 + self.H_fine_res * self.W_fine_res],
                                                   activation=self.opt.activation,
                                                   normalization=self.opt.normalization,
                                                   norm_momentum=opt.norm_momentum,
                                                   norm_act_at_last=False,
                                                   dropout_list=[0.5, 0.5, 0])
        else:
            self.per_point_pn = layers_pc.PointNet(per_point_pn_in_channels,
                                                   [128, 128, 2],
                                                   activation=self.opt.activation,
                                                   normalization=self.opt.normalization,
                                                   norm_momentum=opt.norm_momentum,
                                                   norm_act_at_last=False,
                                                   dropout_list=[0.5, 0.5, 0])
        # assemble_channels = 128 + 512
        assemble_channels = 512
        self.assemble_pn = layers_pc.PointNet(assemble_channels,
                                                   [256, 128, 32],
                                                   activation=self.opt.activation,
                                                   normalization=self.opt.normalization,
                                                   norm_momentum=opt.norm_momentum,
                                                   norm_act_at_last=False,
                                                   dropout_list=[0.5, 0.5, 0])
        self.attention = layers_common.AttentionModule(32)
        self.fully_connected_first = torch.nn.Linear(32, 16)
        self.scoring_layer = torch.nn.Linear(16, 1)

    def gather_topk_features(self, min_k_idx, features):
        """

        :param min_k_idx: BxNxk
        :param features: BxCxM
        :return:
        """
        B, N, k = min_k_idx.size(0), min_k_idx.size(1), min_k_idx.size(2)
        C, M = features.size(1), features.size(2)

        return torch.gather(features.unsqueeze(3).expand(B, C, M, k),
                            index=min_k_idx.unsqueeze(1).expand(B, C, N, k),
                            dim=2)  # BxCxNxk

    def upsample_by_interpolation(self,
                                  interp_ab_topk_idx,
                                  node_a,
                                  node_b,
                                  up_node_b_features):
        # print("interp_ab_topk_idx : ", interp_ab_topk_idx.shape) #([8, 20480, 3])
        # print("node_a : ", node_a.shape) #([8, 3, 20480])
        # print("node_b : ", node_b.shape) #([8, 3, 128])
        # print("up_node_b_features : ", up_node_b_features.shape) #([8, 512, 128])
        interp_ab_topk_node_b = self.gather_topk_features(interp_ab_topk_idx, node_b)  # Bx3xMaxk
        # print("interp_ab_topk_node_b : ", interp_ab_topk_node_b.shape) #([8, 3, 20480, 3])
        # Bx3xMa -> Bx3xMaxk -> BxMaxk
        interp_ab_node_diff = torch.norm(node_a.unsqueeze(3) - interp_ab_topk_node_b, dim=1, p=2, keepdim=False)
        # print("interp_ab_node_diff : ", interp_ab_node_diff.shape) #([8, 20480, 3])
        interp_ab_weight = 1 - interp_ab_node_diff / torch.sum(interp_ab_node_diff, dim=2, keepdim=True)  # BxMaxk
        # print("interp_ab_weight : ", interp_ab_weight.shape) #([8, 20480, 3])
        interp_ab_topk_node_b_features = self.gather_topk_features(interp_ab_topk_idx, up_node_b_features)  # BxCxMaxk
        # print("interp_ab_topk_node_b_features : ", interp_ab_topk_node_b_features.shape) #e([8, 512, 20480, 3])
        # BxCxMaxk -> BxCxMa
        interp_ab_weighted_node_b_features = torch.sum(interp_ab_weight.unsqueeze(1) * interp_ab_topk_node_b_features,
                                                       dim=3)
        return interp_ab_weighted_node_b_features

    def forward(self,
                pc, intensity, sn, label, node_a, node_b,
                img):
        """

        :param pc: Bx3xN
        :param intensity: Bx1xN
        :param sn: Bx3xN
        :param label: Bx1xN
        :param node: Bx3xM
        :param img: BLx3xHxW
        :return:
        """
        B, N, Ma, Mb = pc.size(0), pc.size(2), node_a.size(2), node_b.size(2)

        # print("node_a size = ", node_a.shape) #([8, 3, 128])
        # print("node_b size = ", node_b.shape) #([8, 3, 128])

        # point cloud detector ----------------------------------------------------
        # BxC_pointxN  这里是点云特征的提取
        pc_center,\
        cluster_mean, \
        node_a_min_k_idx, \
        first_pn_out, \
        second_pn_out, \
        node_a_features, \
        node_b_features, \
        global_feature = self.pc_encoder(pc,
                                          intensity,
                                          sn,
                                          label,
                                          node_a,
                                          node_b)
        C_global = global_feature.size(1)

        # print('the first_pn_out size is ', first_pn_out.shape) #([8, 32, 20480])
        # print('the second_pn_out size is ', second_pn_out.shape) #([8, 64, 20480])
        # print('the node_a_features size is ', node_a_features.shape) #([8, 64, 128])
        # print('the node_b_features size is ', node_b_features.shape) #([8, 256, 128])
        # print('the pc f size is ', global_feature.shape) #([8, 512, 1])  8代表的是batch吧

        # image detector ----------------------------------------------------------
        # BxC_imgxHxW, BxC_imgx1x1 这里就包含了img信息的提取,还有这里读的到底是左目还是右目呀  这里是左目训练一次，右目训练一下
        img_s16_feature_map, img_s32_feature_map, img_global_feature = self.img_encoder(img)
        # print('the img_s16 f size is ', img_s16_feature_map.shape) #([8, 256, 10, 32])
        # print('the img_s32 f size is ', img_s32_feature_map.shape) #([8, 512, 5, 16])
        # print('the img_global f size is ', img_global_feature.shape) #([8, 512, 1, 1])
        C_img = img_global_feature.size(1)
        img_s16_feature_map_BCHw = img_s16_feature_map.view(B, img_s16_feature_map.size(1), -1)  # BxC_imgx(H*W) ([8, 256, 320])
        img_s32_feature_map_BCHw = img_s32_feature_map.view(B, img_s32_feature_map.size(1), -1)  # BxC_imgx(H*W) ([8, 512, 80])
        img_global_feature_BCMa = img_global_feature.squeeze(3).expand(B, C_img, Ma)  # BxC_img -> BxC_imgxMa ([8, 512, 128])
        img_global_feature_BCMb = img_global_feature.squeeze(3).expand(B, C_img, Mb)  # BxC_img -> BxC_imgxMb ([8, 512, 128])

        # print('the img_s16 f size is ', img_s16_feature_map_BCHw.shape)
        # print('the img_s32 f size is ', img_s32_feature_map_BCHw.shape)
        # print('the img_BCMa f size is ', img_global_feature_BCMa.shape)
        # print('the img_BCMb f size is ', img_global_feature_BCMb.shape)

        # 到这里为止，就是获得了两者的全局特征，后面要接入DIME结构   #([8, 512, 1, 1]) and #([8, 512, 1])
        # 这里node b features作为局部特征
        sim_mat, sim_paths = self.itr_module(img_s32_feature_map_BCHw, global_feature, node_b_features)
        # print("comming here success")



        # 下面是两个attention模块
        # assemble node_a_features, node_b_features, global_feature, img_feature, img_s32_features
        # ----------------------------------------
        # use attention method to select resnet features for each node_b_feature
        # node_b_attention_score = self.node_b_attention_pn(torch.cat((node_b_features,
        #                                                              img_global_feature_BCMb), dim=1))  # Bx(H*W)xMb  5 * 16
        # # print('node_b_attention_score size is ', node_b_attention_score.shape)  #([8, 80, 128])
        # node_b_weighted_img_s32_feature_map = torch.mean(img_s32_feature_map_BCHw.unsqueeze(3) * node_b_attention_score.unsqueeze(1),
        #                                           dim=2)  # BxC_imgx(H*W)xMb -> BxC_imgxMb
        # # print('node_b_weighted_img_s32_feature_map size is ', node_b_weighted_img_s32_feature_map.shape) #([8, 512, 128])
        # # 这里就是attetion权重加权后的结果

        # #尝试将这里作为合并特征一
        # up_node_b_features = self.node_b_pn(torch.cat((node_b_features,
        #                                                global_feature.expand(B, C_global, Mb),
        #                                                node_b_weighted_img_s32_feature_map,
        #                                                img_global_feature_BCMb), dim=1))  # BxCxMb
        # # print('up_node_b_features size is ', up_node_b_features.shape) #([8, 512, 128])  通过pn再提取一遍特征  (这里组合了很多的特征)

        # # interpolation of pc over node_b
        # # print('pc shape is ', pc.shape) #([8, 3, 20480])
        # # print('node_b shale is ', node_b.shape) #([8, 3, 128])
        # # pc_node_b_diff = torch.norm(pc.unsqueeze(3) - node_b.unsqueeze(2), p=2, dim=1, keepdim=False)  # BxNxMb
        # # print('pc_node_diff size is ', pc_node_b_diff.shape ) #([8, 20480, 128])
        # # BxNxk
        # # _, interp_pc_node_b_topk_idx = torch.topk(pc_node_b_diff, k=self.opt.k_interp_point_b,
        # #                                           dim=2, largest=False, sorted=True)
        # # print('the size of interp_pc_node_b_topk_idx is', interp_pc_node_b_topk_idx.shape)   #([8, 20480, 3])
        # # 感觉像找到关联性最大的三个特征
        # #然后使用了一个上采样操作
        # # interp_pb_weighted_node_b_features = self.upsample_by_interpolation(interp_pc_node_b_topk_idx,
        # #                                                                     pc,
        # #                                                                     node_b,
        # #                                                                     up_node_b_features)
        # # print('the size of interp_pb_weighted_node_b_features is', interp_pb_weighted_node_b_features.size()) #([8, 512, 20480])

        # # interpolation of point over node_a  ----------------------------------------------
        # # use attention method to select resnet features for each node_a_feature
        # node_a_attention_score = self.node_a_attention_pn(torch.cat((node_a_features,
        #                                                              img_global_feature_BCMa), dim=1))  # Bx(H*W)xMa
        # node_a_weighted_img_s16_feature_map = torch.mean(
        #     img_s16_feature_map_BCHw.unsqueeze(3) * node_a_attention_score.unsqueeze(1),
        #     dim=2)  # BxC_imgx(H*W)xMa -> BxC_imgxMa
        # # interpolation of node_a over node_b
        # node_a_node_b_diff = torch.norm(node_a.unsqueeze(3) - node_b.unsqueeze(2), p=2, dim=1, keepdim=False)  # BxMaxMb
        # _, interp_nodea_nodeb_topk_idx = torch.topk(node_a_node_b_diff, k=self.opt.k_interp_ab,
        #                                             dim=2, largest=False, sorted=True)
        # interp_ab_weighted_node_b_features = self.upsample_by_interpolation(interp_nodea_nodeb_topk_idx,
        #                                                                     node_a,
        #                                                                     node_b,
        #                                                                     up_node_b_features)

        # # 这个作为融合特征二试试看
        # up_node_a_features = self.node_a_pn(torch.cat((node_a_features,
        #                                                interp_ab_weighted_node_b_features,
        #                                                node_a_weighted_img_s16_feature_map),
        #                                               dim=1))  # BxCxMa  8 * 128 * 128

        # #将融合的特征送到一层pn中进行测试
        # # abstract_features = self.assemble_pn(torch.cat((up_node_b_features,
        # #                                         up_node_a_features), dim=1)) #BXfXMa  8X32X128
        # abstract_features = self.assemble_pn(up_node_b_features) #BXfXMa  8X32X128

        # pc_num_f = abstract_features.permute(0,2,1)   # BXMaXF
        # pooled_features,attention_scores = self.attention(pc_num_f) #BXFX1
        # # print("pooled_features shape is ", pooled_features.shape) #([8, 32, 1])

        # scores = pooled_features.permute(0,2,1)   # BX1XF
        # scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        # # print("scores: ", scores.shape) #([8, 1, 16])
        # score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)

        return sim_mat, sim_paths




