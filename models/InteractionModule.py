import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init
import sys
import os
sys.path.append(os.getcwd())
from kitti import options_clip
from models.DynamicInteraction import DynamicInteraction_Layer0, DynamicInteraction_Layer

class InteractionModule(nn.Module):
    def __init__(self, opt, num_layer_routing=3, num_cells=4, path_hid=128):
        super(InteractionModule, self).__init__()
        self.opt = opt
        self.num_cells = num_cells = 4
        self.dynamic_itr_l0 = DynamicInteraction_Layer0(opt, num_cells, num_cells)
        self.dynamic_itr_l1 = DynamicInteraction_Layer(opt, num_cells, num_cells)
        self.dynamic_itr_l2 = DynamicInteraction_Layer(opt, num_cells, 1)
        total_paths = num_cells ** 2 * (num_layer_routing - 1) + num_cells
        self.path_mapping = nn.Linear(total_paths, path_hid)
        self.bn = nn.BatchNorm1d(opt.embed_size)

    # 计算两者之间的相似度
    def calSim_i2t(self, ref_imgs, img_f, lidar_f):
        ''' ref_imgs--(n_img,80, 512),
            img_f --(8*80*512)
            lidar_f -- (8, 512)
        '''
        n_img, n_rgn, d = ref_imgs.size()
        ref_imgs = (self.bn(ref_imgs.contiguous().view(n_img*n_rgn, d))).view(n_img,n_rgn, d)
        ref_imgs = ref_imgs.mean(1) + ref_imgs.max(1)[0]
        ref_imgs = F.normalize(ref_imgs, dim=-1)    # (8 * 512)
        # print("ref_imgs size is: ", ref_imgs.shape) #(8 * 512)
        lidar_f = lidar_f
        sim = (lidar_f * ref_imgs).sum(-1)
        return sim

    def calScores(self, aggr_res, img_f, lidar_f):
        assert len(aggr_res) == 1
        aggr_res = aggr_res[0]
        scores = self.calSim_i2t(aggr_res, img_f, lidar_f)
        return scores

    def forward(self, img_f, lidar_f, node_b):
        img_f = img_f.permute(0,2,1)
        lidar_f = lidar_f.squeeze(2)
        node_b = node_b.permute(0,2,1)
        # print("the img_f shape is ",img_f.shape) #([64, 80, 512])
        # print("the lidar_f shape is ",lidar_f.shape) #([64, 512])
        # print("the node_b shape is ",node_b.shape) #([64, 30, 256])
        # self.dynamic_itr_l0(img_f, lidar_f,node_b)
        pairs_emb_lst, paths_l0 = self.dynamic_itr_l0(img_f, lidar_f,node_b)
        # print("paths_l0 shape is ",len(paths_l0), paths_l0[0])
        # print("pairs_emb_lst shape is ",len(pairs_emb_lst), pairs_emb_lst[0].shape)
        pairs_emb_lst, paths_l1 = self.dynamic_itr_l1(pairs_emb_lst, img_f, lidar_f,node_b)
        # print("paths_l1 shape is ",len(paths_l1), paths_l1[0])
        # print("pairs_emb_lst shape is ",len(pairs_emb_lst), pairs_emb_lst[0].shape)
        # print("paths_l2 shape is ",len(paths_l2), paths_l2[0].shape)
        pairs_emb_lst, paths_l2 = self.dynamic_itr_l2(pairs_emb_lst, img_f, lidar_f,node_b)
        # print("pairs_emb_lst shape is ",len(pairs_emb_lst), pairs_emb_lst[0].shape)
        # print("paths_l2 shape is ",len(paths_l2), paths_l2[0])
        score = self.calScores(pairs_emb_lst, img_f, lidar_f)

        # print(score)

        n_img= paths_l2.size(0)
        # print("path_l2 size is： ", paths_l2.size())
        paths_l0 = paths_l0.view(n_img, -1)
        paths_l1 = paths_l1.view(n_img, -1)
        paths_l2 = paths_l2.view(n_img, -1)
        # print(paths_l0.shape)
        # print(paths_l1.shape)
        # print(paths_l2.shape)
        paths = torch.cat([paths_l0, paths_l1, paths_l2], dim=-1) # (n_img, n_stc, total_paths)
        # print(paths.shape)
        # paths = paths.mean(dim=1) # (n_img, total_paths)


        paths = self.path_mapping(paths)
        paths = F.normalize(paths, dim=-1)
        sim_paths = paths.matmul(paths.t())

        # print(sim_paths)

        if self.training:
            return score, sim_paths
        else:
            return score, sim_paths

if __name__ == '__main__':
    opt = options_clip.Options()
    se = InteractionModule(opt)
    img_f=torch.Tensor(8,512,80).permute(0,2,1) # BxC_imgx(H*W) ([8, 80, 512])
    node_b = torch.Tensor(8,512,30).permute(0,2,1) #([8, 128, 512]
    # print(img_f.shape)
    lidar_f = torch.Tensor(8,512,1).squeeze(2)
    b = se(img_f,lidar_f,node_b)
    # print(b)  #torch.Size([4])