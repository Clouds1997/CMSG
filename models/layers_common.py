import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.batchnorm as _BatchNorm

from functools import partial
import math
from typing import Tuple, List

class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, filters : int=82):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.filters_3 = filters
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.filters_3, self.filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        batch_size = embedding.shape[0]
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=1) # 0 # nxf -> f  bxnxf->bxf
        transformed_global = torch.tanh(global_context) # f  bxf
        sigmoid_scores = torch.sigmoid(torch.matmul(embedding,transformed_global.view(batch_size,-1, 1)))   #weights      nxf fx1  bxnxf bxfx1 bxnx1
        representation = torch.matmul(embedding.permute(0,2,1),sigmoid_scores)    # bxnxf bxfxn bxnx1 bxfx1
        return representation, sigmoid_scores

class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k, stage_num=1, is_normalized=False):
        super(EMAU, self).__init__()
        self.stage_num = stage_num
        self.is_normalized = is_normalized

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv1d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv1d(c, c, 1, bias=False),
            nn.BatchNorm1d(c))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """

        :param x: BxCxN
        :return: BxCxN
        """
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, n = x.size()
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        # x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        if self.is_normalized:
            return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))
        else:
            return inp
