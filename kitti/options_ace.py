import numpy as np
import math
import torch


class Options:
    def __init__(self):
        self.dataroot = 'dataset/kitti'
        # self.dataroot = '/data/personal/jiaxin/datasets/kitti'
        self.checkpoints_dir = 'checkpoints'
        self.version = '1.27'
        self.is_debug = False #True
        self.is_fine_resolution = True
        self.is_remove_ground = False
        # self.accumulation_frame_num = 3
        self.accumulation_frame_num = 0
        self.accumulation_frame_skip = 6

        self.delta_ij_max = 40
        self.translation_max = 10.0

        self.crop_original_top_rows = 50
        self.img_scale = 0.5
        self.img_H = 160  # 320 * 0.5
        self.img_W = 512  # 1224 * 0.5
        # the fine resolution is img_H/scale x img_W/scale
        self.img_fine_resolution_scale = 32

        self.input_pt_num = 20480
        self.pc_min_range = -1.0
        self.pc_max_range = 80.0
        # 这里的node_a 和 node_b表示的是什么意思
        self.node_a_num = 128
        self.node_b_num = 128
        self.k_ab = 16
        self.k_interp_ab = 3
        self.k_interp_point_a = 3
        self.k_interp_point_b = 3

        # CAM coordinate
        self.P_tx_amplitude = 0
        self.P_ty_amplitude = 0
        self.P_tz_amplitude = 0
        self.P_Rx_amplitude = 0.0 * math.pi / 12.0
        self.P_Ry_amplitude = 2.0 * math.pi
        self.P_Rz_amplitude = 0.0 * math.pi / 12.0
        self.dataloader_threads = 10

        self.epochs = 100
        self.batch_size = 32
        self.gpu_ids = [1,2,3]
        self.device = torch.device('cuda', self.gpu_ids[0])
        # self.device = torch.device('cuda', 0)
        self.normalization = 'batch'
        self.norm_momentum = 0.1
        self.activation = 'relu'
        self.lr = 0.001
        self.lr_decay_step = 20
        self.lr_decay_scale = 0.5
        self.vis_max_batch = 4
        if self.is_fine_resolution:
            self.coarse_loss_alpha = 50
        else:
            self.coarse_loss_alpha = 1
        
        #下面是DIME的参数
        self.logger_name = 'runs/runX'
        self.data_path = '/data'

    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--logger_name', default='runs/runX', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--data_path', default='/data', help='path to datasets')
    parser.add_argument('--data_name', default='precomp', help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument("--bert_path", default='./', type=str, help="The BERT model path.")
    parser.add_argument('--max_words', default=32, type=int, help='maximum number of words in a sentence.')
    parser.add_argument('--extra_stc', type=int, default=0, help='Sample (extra_stc * bs) extra sentences.')
    parser.add_argument('--extra_img', type=int, default=0, help='Sample (extra_stc * bs) extra images.')
    # Optimization
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--learning_rate', default=.0002, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int, help='Number of epochs to update the learning rate.')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_true', default=True, help='Use max instead of sum in the rank loss.')
    parser.add_argument('--trade_off', default=0.5, type=float, help='Trade-off parameter for path regularization.')
    # Base
    parser.add_argument('--img_dim', default=2048, type=int, help='Dimensionality of the image embedding.')
    # parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input.')
    parser.add_argument('--embed_size', default=256, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--direction', type=str, default='i2t',help='Version of model, i2t | t2i')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=2000, type=int, help='Number of steps to run validation.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--finetune', action='store_true', help='Fine-tune the image encoder.')
    # parser.add_argument('--cnn_type', default='vgg19', help="""The CNN used for image encoder(e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true', help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine', help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true', help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true', help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true', help='Ensure the training is always done in train mode (Not recommended).')
    # DIME
    parser.add_argument('--num_head_IMRC', type=int, default=16, help='Number of heads in Intra-Modal Reasoning Cell')
    parser.add_argument('--hid_IMRC', type=int, default=512, help='Hidden size of FeedForward in Intra-Modal Reasoning Cell')
    parser.add_argument('--raw_feature_norm_CMRC', default="clipped_l2norm", help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--lambda_softmax_CMRC', default=4., type=float, help='Attention softmax temperature.')
    parser.add_argument('--hid_router', type=int, default=512, help='Hidden size of MLP in routers')
    opt = parser.parse_args()


