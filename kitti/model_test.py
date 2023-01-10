import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math

from models.VisNet import EncoderImage

class ACE(object):
    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size, opt.direction, 
                                    opt.finetune, use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm, drop=opt.drop)
        if torch.cuda.is_available(): 
            self.img_enc.cuda()
            cudnn.benchmark = True
        
        # params = list(self.txt_enc.parameters())
        params = list(self.img_enc.parameters())
        # params += list(self.itr_module.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        # self.txt_enc.train()
        # self.itr_module.train()

    def forward_emb(self, batch_data, volatile=False):
        """Compute the image and caption embeddings
        """
        # images, input_ids, lengths, ids, attention_mask, token_type_ids = batch_data
        pc, intensity, sn, node_a, node_b, \
            P, img, K, t_ij, target = batch_data
        if torch.cuda.is_available():
            img = img.cuda()
            # input_ids = input_ids.cuda()
            # attention_mask = attention_mask.cuda()
            # token_type_ids = token_type_ids.cuda()

        # Forward
        # stc_emb, wrd_emb = self.txt_enc(input_ids, attention_mask, token_type_ids, lengths)
        print("comming here .....")
        f = self.img_enc(img)
        print('img_emb shape is :', f.shape)
        # print('rgn_emb shape is :', rgn_emb.shape)
        return f

    def train_emb(self, epoch, batch_data, *args):
        """One training step given images and captions.
        """
        # self.Eiters += 1
        # self.logger.update('Eit', self.Eiters)
        # self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # images, input_ids, lengths, ids, attention_mask, token_type_ids = batch_data
        img_emb = self.forward_emb(batch_data)

