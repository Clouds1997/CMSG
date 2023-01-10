import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .SelfAttention import SelfAttention
from .Router import Router
from .Refinement import Refinement

# 对于简单的语句，直接一个relu完事
class RectifiedIdentityCell(nn.Module):
    def __init__(self, opt, num_out_path):
        super(RectifiedIdentityCell, self).__init__()
        self.keep_mapping = nn.ReLU()
        self.router = Router(num_out_path, opt.embed_size, opt.hid_router)

    def forward(self, x):
        path_prob = self.router(x)
        # print("router is ok")
        emb = self.keep_mapping(x)

        return emb, path_prob

# 这个是加上了模态内自注意力机制
class IntraModelReasoningCell(nn.Module):
    def __init__(self, opt, num_out_path):
        super(IntraModelReasoningCell, self).__init__()
        self.opt = opt
        self.router = Router(num_out_path, opt.embed_size, opt.hid_router)
        self.sa = SelfAttention(opt.embed_size, opt.hid_IMRC, opt.num_head_IMRC)

    def forward(self, inp, stc_lens=None):
        path_prob = self.router(inp)
        if inp.dim() == 4:
            n_img, n_stc, n_local, dim = inp.size()
            x = inp.view(-1, n_local, dim)
        else:
            x = inp

        sa_emb = self.sa(x)
        if inp.dim() == 4:
            sa_emb = sa_emb.view(n_img, n_stc, n_local, -1)
        return sa_emb, path_prob

# 这个是用另外一个模态的局部信息信息来影响这个模态局部信息的提取
class CrossModalRefinementCell(nn.Module):
    def __init__(self, opt, num_out_path):
        super(CrossModalRefinementCell, self).__init__()
        self.direction = opt.direction
        self.refine = Refinement(opt.embed_size, opt.raw_feature_norm_CMRC, opt.lambda_softmax_CMRC, opt.direction) 
        self.router = Router(num_out_path, opt.embed_size, opt.hid_router)

    def forward(self, img_f, lidar_f, node_b):
        l_emb = img_f

        path_prob = self.router(l_emb)
        rf_pairs_emb = self.refine(img_f, lidar_f,node_b)
        return rf_pairs_emb, path_prob

# 用另外一个模态的全局信息来影响这个模态的局部信息的提取  这里就相当于是做了一个attention
class GlobalLocalGuidanceCell(nn.Module):
    def __init__(self, opt, num_out_path):
        super(GlobalLocalGuidanceCell, self).__init__()
        self.opt = opt
        self.direction = self.opt.direction
        self.router = Router(num_out_path, opt.embed_size, opt.hid_router)
        self.fc_1 = nn.Linear(opt.embed_size, opt.embed_size)
        self.fc_2 = nn.Linear(opt.embed_size, opt.embed_size)

    def regulate(self, l_emb, g_emb_expand):
        l_emb_mid = self.fc_1(l_emb)
        x = l_emb_mid * g_emb_expand
        x = F.normalize(x, dim=-2)
        ref_l_emb = (1 + x) * l_emb
        return ref_l_emb

    def forward_i2t(self, img_f, lidar_f):
        #img_f [batch,80 * 512] lidar_f [batch,512]
        n_img = img_f.size(0)
        n_rgn = img_f.size(-2)
        ref_rgns = []
        query = img_f  #8 * 80 * 512
            
        # stc_i = stc[i].unsqueeze(0).unsqueeze(1).contiguous()
        lidar_f = lidar_f.unsqueeze(1)
        lidar_expand = lidar_f.expand(n_img, n_rgn, -1)
        # print("there is ", lidar_expand.shape)
        ref_rgn = self.regulate(query, lidar_expand)

        return ref_rgn

    def forward(self, img_f, lidar_f):
        path_prob = self.router(img_f)

        ref_emb = self.forward_i2t(img_f, lidar_f)

        return ref_emb, path_prob



