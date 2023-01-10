import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def func_attention(query, context, raw_feature_norm, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d) 8 80 512
    context: (n_context, sourceL, d) 8 30 512
    """
    # print(query.shape)
    # print(context.shape)
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)   #(n, d, qL) 8 512 80

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    # print(queryT.shape)
    # print(context.shape)
    attn = torch.bmm(context, queryT)   #(n, cL, qL)  8 30 80
    # print(attn.shape)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=-1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        # attn = l2norm(attn, 2)
        attn = F.normalize(attn, dim=2)
    elif raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous() #(n, qL, cL)
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)    #(n*qL, cL)
    attn = F.softmax(attn*smooth, dim=-1)                #(n*qL, cL)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)   #(n, qL, cL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()    #(n, cL, qL)

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)   #(n, d, cL)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)    #(n, d, qL)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)    #(n, qL, d)

    return weightedContext, attnT


class Refinement(nn.Module):
    def __init__(self, embed_size, raw_feature_norm, lambda_softmax, direction):
        super(Refinement, self).__init__()
        self.raw_feature_norm = raw_feature_norm
        self.lambda_softmax = lambda_softmax
        self.direction = direction

        self.fc_scale = nn.Linear(embed_size, embed_size)
        self.fc_shift = nn.Linear(embed_size, embed_size)
        self.fc_1 = nn.Linear(embed_size, embed_size)
        self.fc_2 = nn.Linear(embed_size, embed_size)

    def refine(self, query, weiContext):
        scaling = torch.tanh(self.fc_scale(weiContext))
        shifting = self.fc_shift(weiContext)
        modu_res = self.fc_2(F.relu(self.fc_1(query * scaling + shifting)))
        ref_q = modu_res + query

        return ref_q

    def forward_i2t(self, img_f, lidar_f,node_b):
        if(img_f.dim() == 4):
            query = img_f.squeeze(1)
        else:
            query = img_f
        # Get the i-th text description
        # print("query shape is:", query.shape)
        # print("node_b shape is:", node_b.shape)
        weiContext, attn = func_attention(query, node_b, self.raw_feature_norm, smooth=self.lambda_softmax)
        ref_img = self.refine(query, weiContext)

        return ref_img


    def forward(self, img_f, lidar_f, node_b):
        ref_emb = self.forward_i2t(img_f, lidar_f, node_b)


        return ref_emb



