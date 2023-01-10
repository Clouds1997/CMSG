import torch
import torch.nn as nn

from models import networks_img
from models.mmcv.conv_module import ConvModule
from models import resnet
from models import networks_pc
from models import layers_common
from models import layers_pc
from kitti.options import Options
from util import pytorch_helper
from models import dgcnn
from models import layers_batch

class SG(torch.nn.Module):

    def __init__(self, args):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SG, self).__init__()
        self.args = args
        # self.device = torch.device('cuda', self.args.cuda)
        self.number_labels = 12
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.attention = layers_batch.AttentionModule(self.args)
        self.tensor_network = layers_batch.TenorNetworkModule (self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        bias_bool = False # TODO
        self.dgcnn_s_conv1 = nn.Sequential(
            nn.Conv2d(3*2, self.args.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_f_conv1 = nn.Sequential(
            nn.Conv2d(self.number_labels * 2, self.args.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_s_conv2 = nn.Sequential(
            nn.Conv2d(self.args.filters_1*2, self.args.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_f_conv2 = nn.Sequential(
            nn.Conv2d(self.args.filters_1 * 2, self.args.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_s_conv3 = nn.Sequential(
            nn.Conv2d(self.args.filters_2*2, self.args.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_f_conv3 = nn.Sequential(
            nn.Conv2d(self.args.filters_2 * 2, self.args.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv_end = nn.Sequential(nn.Conv1d(self.args.filters_3,
                                                      self.args.filters_3, kernel_size=1, bias=bias_bool),
                                            nn.BatchNorm1d(self.args.filters_3), nn.LeakyReLU(negative_slope=0.2))

    def dgcnn_conv_pass(self, x):

        # 这里进来应该是 b * 30 * 12

        self.k = self.args.K
        # xyz = x[:,:3,:] # Bx3xN
        sem = x.permute(0,2,1) # Bx12x30

        # 这里就是将label取出来然后进行图网络编码的地方
        # print("the sem size is", sem.shape)
        sem = dgcnn.get_graph_feature(sem, k=self.k)  # Bx2fxNxk
        sem = self.dgcnn_f_conv1(sem)
        sem1 = sem.max(dim=-1, keepdim=False)[0]
        sem = dgcnn.get_graph_feature(sem1, k=self.k)
        sem = self.dgcnn_f_conv2(sem)
        sem2 = sem.max(dim=-1, keepdim=False)[0]
        sem = dgcnn.get_graph_feature(sem2, k=self.k)
        sem = self.dgcnn_f_conv3(sem)
        sem3 = sem.max(dim=-1, keepdim=False)[0]
        # print("the sem3 size is: ", sem3.shape)

        # x = self.dgcnn_conv_all(x)
        # x = self.dgcnn_conv_end(sem3)
        # print(x.shape)

        # x = x.permute(0, 2, 1)  # [node_num, 32]
        sem3 = sem3.permute(0,2,1)

        return sem3

    def forward(self, graph1, graph2):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        # print('there is start')
        # features_1 = data["features_1"].cuda('cuda:'+self.args.cuda)
        # features_2 = data["features_2"].cuda('cuda:'+self.args.cuda)

        # print('the features_1 device is :', features_1.device)
        # print('the features_2 device is :', features_2.device)
        # print("comming here success!!!!")
        # print("the graph size is: ", graph.shape)
        # features B x (3+label_num) x node_num
        abstract_features_1 = self.dgcnn_conv_pass(graph1) # node_num x feature_size(filters-3)
        abstract_features_2 = self.dgcnn_conv_pass(graph2)
        # abstract_features_2 = self.dgcnn_conv_pass(features_2)  #BXNXF
        # print("abstract feature: ", abstract_features_1) #([1024, 25, 32])
        # print("abstract feature: ", abstract_features.shape) #([64, 30, 32])
        # print('the abstract_features_1 device is :', abstract_features_1.device)
        # print('the abstract_features_2 device is :', abstract_features_1.device)
        pooled_features_1, attention_scores_1 = self.attention(abstract_features_1) # bxfx1
        pooled_features_2, attention_scores_2 = self.attention(abstract_features_2)
        # print("pooled_features: ", pooled_features.shape)  #([64, 32, 1])
        # print('the pooled_features_1 device is :', pooled_features_1.device)
        # print('the pooled_features_2 device is :', pooled_features_2.device)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        # print("scores: ", scores.shape) #([1024, 16, 1])
        # print('the scores device is :', scores.device)
        # scores = scores.permute(0,2,1) # bx1xf
        # print("scores: ", scores.shape) #([1024, 1, 16])

        # scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        # # print("scores: ", scores.shape) #([1024, 1, 16])
        # score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)
        # print("scores: ", score.shape) #([1024])
        # print('the score device is :', score.device)
        # print('there is end')
        return scores

# 特征点检测器
class KeypointDetectorG(nn.Module):
    def __init__(self, opt: Options):
        super(KeypointDetectorG, self).__init__()
        self.opt = opt

        self.pc_encoder = networks_pc.PCEncoder(opt, Ca=64, Cb=256, Cg=512).to(self.opt.device)
        self.img_encoder = networks_img.ImageEncoder(self.opt).to(self.opt.device)
        self.pc_graph_encoder = SG(self.opt).to(self.opt.device)
        self.img_graph_encoder = SG(self.opt).to(self.opt.device)

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
        assemble_channels = 512 + 16
        # assemble_channels = 512
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
                pc, intensity, sn, label, node_a, node_b, pc_graph, img_graph, 
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
        # 图节点特征编码
        graph_feature = self.pc_graph_encoder(pc_graph, img_graph)  #([64, 16, 1])
        C_graph = graph_feature.size(1)
        graph_feature_BCMb = graph_feature.expand(B, C_graph, Mb) #([64, 16, 128])
        # print("the graph size is: ", graph_feature_BCMb.shape)
        # img_graph_feature = self.img_graph_encoder(img_graph) #([64, 32, 1])


        # 下面是两个attention模块
        # assemble node_a_features, node_b_features, global_feature, img_feature, img_s32_features
        # ----------------------------------------
        # use attention method to select resnet features for each node_b_feature
        node_b_attention_score = self.node_b_attention_pn(torch.cat((node_b_features,
                                                                     img_global_feature_BCMb), dim=1))  # Bx(H*W)xMb  5 * 16
        # print('node_b_attention_score size is ', node_b_attention_score.shape)  #([8, 80, 128])
        node_b_weighted_img_s32_feature_map = torch.mean(img_s32_feature_map_BCHw.unsqueeze(3) * node_b_attention_score.unsqueeze(1),
                                                  dim=2)  # BxC_imgx(H*W)xMb -> BxC_imgxMb
        # print('node_b_weighted_img_s32_feature_map size is ', node_b_weighted_img_s32_feature_map.shape) #([8, 512, 128])
        # 这里就是attetion权重加权后的结果

        #尝试将这里作为合并特征一
        up_node_b_features = self.node_b_pn(torch.cat((node_b_features,
                                                       global_feature.expand(B, C_global, Mb),
                                                       node_b_weighted_img_s32_feature_map,
                                                       img_global_feature_BCMb), dim=1))  # BxCxMb
        # print('up_node_b_features size is ', up_node_b_features.shape) #([8, 512, 128])  通过pn再提取一遍特征  (这里组合了很多的特征)

        # interpolation of pc over node_b
        # print('pc shape is ', pc.shape) #([8, 3, 20480])
        # print('node_b shale is ', node_b.shape) #([8, 3, 128])
        # pc_node_b_diff = torch.norm(pc.unsqueeze(3) - node_b.unsqueeze(2), p=2, dim=1, keepdim=False)  # BxNxMb
        # print('pc_node_diff size is ', pc_node_b_diff.shape ) #([8, 20480, 128])
        # BxNxk
        # _, interp_pc_node_b_topk_idx = torch.topk(pc_node_b_diff, k=self.opt.k_interp_point_b,
        #                                           dim=2, largest=False, sorted=True)
        # print('the size of interp_pc_node_b_topk_idx is', interp_pc_node_b_topk_idx.shape)   #([8, 20480, 3])
        # 感觉像找到关联性最大的三个特征
        #然后使用了一个上采样操作
        # interp_pb_weighted_node_b_features = self.upsample_by_interpolation(interp_pc_node_b_topk_idx,
        #                                                                     pc,
        #                                                                     node_b,
        #                                                                     up_node_b_features)
        # print('the size of interp_pb_weighted_node_b_features is', interp_pb_weighted_node_b_features.size()) #([8, 512, 20480])

        # interpolation of point over node_a  ----------------------------------------------
        # use attention method to select resnet features for each node_a_feature
        node_a_attention_score = self.node_a_attention_pn(torch.cat((node_a_features,
                                                                     img_global_feature_BCMa), dim=1))  # Bx(H*W)xMa
        node_a_weighted_img_s16_feature_map = torch.mean(
            img_s16_feature_map_BCHw.unsqueeze(3) * node_a_attention_score.unsqueeze(1),
            dim=2)  # BxC_imgx(H*W)xMa -> BxC_imgxMa
        # interpolation of node_a over node_b
        node_a_node_b_diff = torch.norm(node_a.unsqueeze(3) - node_b.unsqueeze(2), p=2, dim=1, keepdim=False)  # BxMaxMb
        _, interp_nodea_nodeb_topk_idx = torch.topk(node_a_node_b_diff, k=self.opt.k_interp_ab,
                                                    dim=2, largest=False, sorted=True)
        interp_ab_weighted_node_b_features = self.upsample_by_interpolation(interp_nodea_nodeb_topk_idx,
                                                                            node_a,
                                                                            node_b,
                                                                            up_node_b_features)

        # 这个作为融合特征二试试看
        up_node_a_features = self.node_a_pn(torch.cat((node_a_features,
                                                       interp_ab_weighted_node_b_features,
                                                       node_a_weighted_img_s16_feature_map),
                                                      dim=1))  # BxCxMa  8 * 128 * 128

        # 将融合的特征送到一层pn中进行测试
        # abstract_features = self.assemble_pn(torch.cat((up_node_b_features,
        #                                         up_node_a_features,
        #                                         graph_feature_BCMb), dim=1)) #BXfXMa  8X32X128
        abstract_features = self.assemble_pn(torch.cat((up_node_b_features,
                                                                                graph_feature_BCMb), dim=1)) #BXfXMa  8X32X128

        pc_num_f = abstract_features.permute(0,2,1)   # BXMaXF
        pooled_features,attention_scores = self.attention(pc_num_f) #BXFX1
        # print("pooled_features shape is ", pooled_features.shape) #([8, 32, 1])

        scores = pooled_features.permute(0,2,1)   # BX1XF
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        # print("scores: ", scores.shape) #([8, 1, 16])
        score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)

        return score  # Bx2xN

class KeypointDetector_Base(nn.Module):
    def __init__(self, opt: Options):
        super(KeypointDetector_Base, self).__init__()
        self.opt = opt

        self.pc_encoder = networks_pc.PCEncoder_Base(opt, Ca=64, Cb=256, Cg=512).to(self.opt.device)
        self.img_encoder = networks_img.ImageEncoder_Base(self.opt).to(self.opt.device)

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
        assemble_channels = 128 + 512
        # assemble_channels = 512
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
        # print("the pc shape is: ",pc.shape)
        # print("the instesity shape is: ",intensity.shape)
        # print("the sn shape is: ",sn.shape)
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
        # print("the img shape is",img[:,:3,:,:].shape)
        img_s16_feature_map, img_s32_feature_map, img_global_feature = self.img_encoder(img[:,:3,:,:])
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

        # 下面是两个attention模块
        # assemble node_a_features, node_b_features, global_feature, img_feature, img_s32_features
        # ----------------------------------------
        # use attention method to select resnet features for each node_b_feature
        node_b_attention_score = self.node_b_attention_pn(torch.cat((node_b_features,
                                                                     img_global_feature_BCMb), dim=1))  # Bx(H*W)xMb  5 * 16
        # print('node_b_attention_score size is ', node_b_attention_score.shape)  #([8, 80, 128])
        node_b_weighted_img_s32_feature_map = torch.mean(img_s32_feature_map_BCHw.unsqueeze(3) * node_b_attention_score.unsqueeze(1),
                                                  dim=2)  # BxC_imgx(H*W)xMb -> BxC_imgxMb
        # print('node_b_weighted_img_s32_feature_map size is ', node_b_weighted_img_s32_feature_map.shape) #([8, 512, 128])
        # 这里就是attetion权重加权后的结果

        #尝试将这里作为合并特征一
        up_node_b_features = self.node_b_pn(torch.cat((node_b_features,
                                                       global_feature.expand(B, C_global, Mb),
                                                       node_b_weighted_img_s32_feature_map,
                                                       img_global_feature_BCMb), dim=1))  # BxCxMb
        # print('up_node_b_features size is ', up_node_b_features.shape) #([8, 512, 128])  通过pn再提取一遍特征  (这里组合了很多的特征)

        # interpolation of pc over node_b
        # print('pc shape is ', pc.shape) #([8, 3, 20480])
        # print('node_b shale is ', node_b.shape) #([8, 3, 128])
        # pc_node_b_diff = torch.norm(pc.unsqueeze(3) - node_b.unsqueeze(2), p=2, dim=1, keepdim=False)  # BxNxMb
        # print('pc_node_diff size is ', pc_node_b_diff.shape ) #([8, 20480, 128])
        # BxNxk
        # _, interp_pc_node_b_topk_idx = torch.topk(pc_node_b_diff, k=self.opt.k_interp_point_b,
        #                                           dim=2, largest=False, sorted=True)
        # print('the size of interp_pc_node_b_topk_idx is', interp_pc_node_b_topk_idx.shape)   #([8, 20480, 3])
        # 感觉像找到关联性最大的三个特征
        #然后使用了一个上采样操作
        # interp_pb_weighted_node_b_features = self.upsample_by_interpolation(interp_pc_node_b_topk_idx,
        #                                                                     pc,
        #                                                                     node_b,
        #                                                                     up_node_b_features)
        # print('the size of interp_pb_weighted_node_b_features is', interp_pb_weighted_node_b_features.size()) #([8, 512, 20480])

        # interpolation of point over node_a  ----------------------------------------------
        # use attention method to select resnet features for each node_a_feature
        node_a_attention_score = self.node_a_attention_pn(torch.cat((node_a_features,
                                                                     img_global_feature_BCMa), dim=1))  # Bx(H*W)xMa
        node_a_weighted_img_s16_feature_map = torch.mean(
            img_s16_feature_map_BCHw.unsqueeze(3) * node_a_attention_score.unsqueeze(1),
            dim=2)  # BxC_imgx(H*W)xMa -> BxC_imgxMa
        # interpolation of node_a over node_b
        node_a_node_b_diff = torch.norm(node_a.unsqueeze(3) - node_b.unsqueeze(2), p=2, dim=1, keepdim=False)  # BxMaxMb
        _, interp_nodea_nodeb_topk_idx = torch.topk(node_a_node_b_diff, k=self.opt.k_interp_ab,
                                                    dim=2, largest=False, sorted=True)
        interp_ab_weighted_node_b_features = self.upsample_by_interpolation(interp_nodea_nodeb_topk_idx,
                                                                            node_a,
                                                                            node_b,
                                                                            up_node_b_features)

        # 这个作为融合特征二试试看
        up_node_a_features = self.node_a_pn(torch.cat((node_a_features,
                                                       interp_ab_weighted_node_b_features,
                                                       node_a_weighted_img_s16_feature_map),
                                                      dim=1))  # BxCxMa  8 * 128 * 128

        # 将融合的特征送到一层pn中进行测试
        abstract_features = self.assemble_pn(torch.cat((up_node_b_features,
                                                up_node_a_features), dim=1)) #BXfXMa  8X32X128
        # abstract_features = self.assemble_pn(up_node_b_features) #BXfXMa  8X32X128

        pc_num_f = abstract_features.permute(0,2,1)   # BXMaXF
        pooled_features,attention_scores = self.attention(pc_num_f) #BXFX1
        # print("pooled_features shape is ", pooled_features.shape) #([8, 32, 1])

        scores = pooled_features.permute(0,2,1)   # BX1XF
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        # print("scores: ", scores.shape) #([8, 1, 16])
        score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)

        return score  # Bx2xN

class KeypointDetector(nn.Module):
    def __init__(self, opt: Options):
        super(KeypointDetector, self).__init__()
        self.opt = opt

        self.pc_encoder = networks_pc.PCEncoder(opt, Ca=64, Cb=256, Cg=512).to(self.opt.device)
        self.img_encoder = networks_img.ImageEncoder(self.opt).to(self.opt.device)

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
        assemble_channels = 128 + 512
        # assemble_channels = 512
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

        # 下面是两个attention模块
        # assemble node_a_features, node_b_features, global_feature, img_feature, img_s32_features
        # ----------------------------------------
        # use attention method to select resnet features for each node_b_feature
        node_b_attention_score = self.node_b_attention_pn(torch.cat((node_b_features,
                                                                     img_global_feature_BCMb), dim=1))  # Bx(H*W)xMb  5 * 16
        # print('node_b_attention_score size is ', node_b_attention_score.shape)  #([8, 80, 128])
        node_b_weighted_img_s32_feature_map = torch.mean(img_s32_feature_map_BCHw.unsqueeze(3) * node_b_attention_score.unsqueeze(1),
                                                  dim=2)  # BxC_imgx(H*W)xMb -> BxC_imgxMb
        # print('node_b_weighted_img_s32_feature_map size is ', node_b_weighted_img_s32_feature_map.shape) #([8, 512, 128])
        # 这里就是attetion权重加权后的结果

        #尝试将这里作为合并特征一
        up_node_b_features = self.node_b_pn(torch.cat((node_b_features,
                                                       global_feature.expand(B, C_global, Mb),
                                                       node_b_weighted_img_s32_feature_map,
                                                       img_global_feature_BCMb), dim=1))  # BxCxMb
        # print('up_node_b_features size is ', up_node_b_features.shape) #([8, 512, 128])  通过pn再提取一遍特征  (这里组合了很多的特征)

        # interpolation of pc over node_b
        # print('pc shape is ', pc.shape) #([8, 3, 20480])
        # print('node_b shale is ', node_b.shape) #([8, 3, 128])
        # pc_node_b_diff = torch.norm(pc.unsqueeze(3) - node_b.unsqueeze(2), p=2, dim=1, keepdim=False)  # BxNxMb
        # print('pc_node_diff size is ', pc_node_b_diff.shape ) #([8, 20480, 128])
        # BxNxk
        # _, interp_pc_node_b_topk_idx = torch.topk(pc_node_b_diff, k=self.opt.k_interp_point_b,
        #                                           dim=2, largest=False, sorted=True)
        # print('the size of interp_pc_node_b_topk_idx is', interp_pc_node_b_topk_idx.shape)   #([8, 20480, 3])
        # 感觉像找到关联性最大的三个特征
        #然后使用了一个上采样操作
        # interp_pb_weighted_node_b_features = self.upsample_by_interpolation(interp_pc_node_b_topk_idx,
        #                                                                     pc,
        #                                                                     node_b,
        #                                                                     up_node_b_features)
        # print('the size of interp_pb_weighted_node_b_features is', interp_pb_weighted_node_b_features.size()) #([8, 512, 20480])

        # interpolation of point over node_a  ----------------------------------------------
        # use attention method to select resnet features for each node_a_feature
        node_a_attention_score = self.node_a_attention_pn(torch.cat((node_a_features,
                                                                     img_global_feature_BCMa), dim=1))  # Bx(H*W)xMa
        node_a_weighted_img_s16_feature_map = torch.mean(
            img_s16_feature_map_BCHw.unsqueeze(3) * node_a_attention_score.unsqueeze(1),
            dim=2)  # BxC_imgx(H*W)xMa -> BxC_imgxMa
        # interpolation of node_a over node_b
        node_a_node_b_diff = torch.norm(node_a.unsqueeze(3) - node_b.unsqueeze(2), p=2, dim=1, keepdim=False)  # BxMaxMb
        _, interp_nodea_nodeb_topk_idx = torch.topk(node_a_node_b_diff, k=self.opt.k_interp_ab,
                                                    dim=2, largest=False, sorted=True)
        interp_ab_weighted_node_b_features = self.upsample_by_interpolation(interp_nodea_nodeb_topk_idx,
                                                                            node_a,
                                                                            node_b,
                                                                            up_node_b_features)

        # 这个作为融合特征二试试看
        up_node_a_features = self.node_a_pn(torch.cat((node_a_features,
                                                       interp_ab_weighted_node_b_features,
                                                       node_a_weighted_img_s16_feature_map),
                                                      dim=1))  # BxCxMa  8 * 128 * 128

        # 将融合的特征送到一层pn中进行测试
        abstract_features = self.assemble_pn(torch.cat((up_node_b_features,
                                                up_node_a_features), dim=1)) #BXfXMa  8X32X128
        # abstract_features = self.assemble_pn(up_node_b_features) #BXfXMa  8X32X128

        pc_num_f = abstract_features.permute(0,2,1)   # BXMaXF
        pooled_features,attention_scores = self.attention(pc_num_f) #BXFX1
        # print("pooled_features shape is ", pooled_features.shape) #([8, 32, 1])

        scores = pooled_features.permute(0,2,1)   # BX1XF
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        # print("scores: ", scores.shape) #([8, 1, 16])
        score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)

        return score  # Bx2xN



