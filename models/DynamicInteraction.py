import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pickle

from models.Cells import RectifiedIdentityCell, IntraModelReasoningCell, GlobalLocalGuidanceCell, CrossModalRefinementCell

def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)

def unsqueeze3d(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DynamicInteraction_Layer0(nn.Module):
    def __init__(self, opt, num_cell, num_out_path):
        super(DynamicInteraction_Layer0, self).__init__()
        self.opt = opt
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path
        self.ric = RectifiedIdentityCell(opt, num_out_path)
        self.imrc = IntraModelReasoningCell(opt, num_out_path)
        self.glgc = GlobalLocalGuidanceCell(opt, num_out_path)
        self.cmrc = CrossModalRefinementCell(opt, num_out_path)

    def forward(self, imgf, lidar_f, node_b):
        aggr_res_lst = self.forward_i2t(imgf, lidar_f,node_b)
        return aggr_res_lst

    def forward_i2t(self, imgf, lidar_f, node_b):
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(imgf)  #torch.Size([8, 80, 512])   torch.Size([8, 4])
        # print(emb_lst[0].shape, " ", path_prob[0].shape)
        emb_lst[1], path_prob[1] = self.glgc(imgf, lidar_f) #torch.Size([8, 80, 512])   torch.Size([8, 4])
        # print(emb_lst[1].shape, " ", path_prob[1].shape)
        emb_lst[2], path_prob[2] = self.imrc(imgf) #8*8*512
        # print(emb_lst[2].shape, " ", path_prob[2].shape)
        emb_lst[3], path_prob[3] = self.cmrc(imgf, lidar_f,node_b) #node b size is ([8, 256, 128])
        # path_prob表示通往下面一层单元的概率
        # print(emb_lst[3].shape, " ", path_prob[3].shape)

        gate_mask = (sum(path_prob) < self.threshold).float()
        # print("gate_mask is ", gate_mask)
        # 这一步就不太看得懂
        all_path_prob = torch.stack(path_prob, dim=2)
        # print("all_path_prob is", all_path_prob.shape) #([8, 4, 4])  表示8个数据，从这4个单元输出往4个单元的概率
        #第一个4表示当前神经元，第二个4表示后向神经元
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]
        # print("path_prob size is ", path_prob) #这是4个8*4的元素
        #这里得出的是当前的四个神经元输出到第i个的概率，其实就是还原了一下最开始的path_prob

        aggr_res_lst = []
        for i in range(self.num_out_path):
            skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]
            res = 0
            for j in range(self.num_cell):
                # print("path_prob is ", path_prob[j].shape)
                cur_path = unsqueeze2d(path_prob[j][:, i])
                # 这里是第j个单元对于所有第i个输出的概率
                # print("cur_path is ", path_prob[j].shape)
                # if emb_lst[j].dim() == 3:
                #     cur_emb = emb_lst[j].unsqueeze(1)
                # else:   # 4
                cur_emb = emb_lst[j]
                # print("cur_emb shape is: ", cur_emb.shape)
                # print("cur_path shape is: ", cur_path.shape)
                res = res + cur_path * cur_emb
                # print("res shape is; ",res.shape)
            res = res + skip_emb
            # print("res shape is; ",res.shape)
            aggr_res_lst.append(res)  #4个元素，每个元素中存的是要送到对应神经元中的8*80*512的特征

        return aggr_res_lst, all_path_prob

class DynamicInteraction_Layer(nn.Module):
    def __init__(self, opt, num_cell, num_out_path):
        super(DynamicInteraction_Layer, self).__init__()
        self.opt = opt
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path

        self.ric = RectifiedIdentityCell(opt, num_out_path)
        self.glgc = GlobalLocalGuidanceCell(opt, num_out_path)
        self.imrc = IntraModelReasoningCell(opt, num_out_path)
        self.cmrc = CrossModalRefinementCell(opt, num_out_path)


    def forward(self, ref_rgn, imgf, lidar_f, node_b):
        aggr_res_lst = self.forward_i2t(ref_rgn, imgf, lidar_f, node_b)

        return aggr_res_lst

    def forward_i2t(self, ref_rgn, imgf, lidar_f, node_b):
        # assert len(ref_rgn) == self.num_cell and ref_rgn[0].dim() == 4
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        # print("ref_rgn[0] shape", ref_rgn[0].shape)
        emb_lst[0], path_prob[0] = self.ric(ref_rgn[0])
        # print(emb_lst[0].shape, " ", path_prob[0].shape)
        emb_lst[1], path_prob[1] = self.glgc(ref_rgn[1], lidar_f)
        # print(emb_lst[0].shape, " ", path_prob[0].shape)
        emb_lst[2], path_prob[2] = self.imrc(ref_rgn[2])
        # print(emb_lst[0].shape, " ", path_prob[0].shape)
        emb_lst[3], path_prob[3] = self.cmrc(ref_rgn[3], lidar_f, node_b)
        # print(emb_lst[3].shape, " ", path_prob[3].shape)

        if self.num_out_path == 1:
            aggr_res_lst = []
            gate_mask_lst = []
            res = 0
            for j in range(self.num_cell):
                gate_mask = (path_prob[j] < self.threshold / self.num_cell).float()
                # print("the gate mask size is: ", gate_mask.shape)
                # print("the ref_rgn size is: ", ref_rgn[j].shape)
                gate_mask_lst.append(gate_mask)
                skip_emb = gate_mask.unsqueeze(-1) * ref_rgn[j]
                # print("the skip_emb size is: ", skip_emb.shape)
                # print("the path_prob[j] size is: ", path_prob[j].shape)
                res += path_prob[j].unsqueeze(-1) * emb_lst[j]
                res += skip_emb

            res = res / (sum(gate_mask_lst) + sum(path_prob)).unsqueeze(-1)
            # print("the res size is: ", res.shape)
            all_path_prob = torch.stack(path_prob, dim=2)
            # print(all_path_prob.shape) #([8, 1, 4]) 表示从4个单元输出往一个单元的概率
            aggr_res_lst.append(res)
        else:
            gate_mask = (sum(path_prob) < self.threshold).float()
            all_path_prob = torch.stack(path_prob, dim=2)
            # print(all_path_prob.shape)

            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]
            aggr_res_lst = []
            for i in range(self.num_out_path):
                skip_emb = unsqueeze2d(gate_mask[:,i]) * emb_lst[0]
                res = 0
                for j in range(self.num_cell):
                    cur_path = unsqueeze2d(path_prob[j][:,i])
                    res = res + cur_path * emb_lst[j]
                res = res + skip_emb
                aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob




