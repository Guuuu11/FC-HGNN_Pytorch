import datetime
import argparse
import random
import numpy as np
import torch

class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of FC-HGNN')
        parser.add_argument('--train', default=1, type=int, help='train(default) or evaluate')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')
        parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
        parser.add_argument('--wd', default=5e-5, type=float, help='weight decay')
        parser.add_argument('--num_iter', default= 400, type=int, help='number of epochs for training')
        parser.add_argument('--dropout', default=0.3, type=float, help='ratio of dropout')
        parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
        parser.add_argument('--n_folds', type=int, default=10, help='number of folds')
        parser.add_argument('--ckpt_path', type=str, default='./', help='checkpoint path to save trained models')

        parser.add_argument('--log_path', type=str, default=r'./inffus_log.txt', help='the path of the log')
        parser.add_argument('--subject_IDs_path', type=str, default='----------------\subjects.txt', help='the path of the subject_IDs')
        parser.add_argument('--phenotype_path', type=str, default=r"---------------------\phenotypic_information.csv", help='the path of the phenotype data')
        parser.add_argument('--data_path', type=str, default=r'-----------------\data',help='the path of the data')

        parser.add_argument('--alpha', default=0.65, type=float, help='adjacency threshold set when building  Brain_connectomic_graph')
        parser.add_argument('--beta', default=1.5, type=float, help='adjacency threshold set when building HPG')
        parser.add_argument('--k1', default=0.9, type=float, help='the pooling ratio of the channel 1 of the LGP')
        parser.add_argument('--k2', default=0.5, type=float,help='the pooling ratio of the channel 2 of the LGP')

        args = parser.parse_args()

        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(" Using GPU in torch")

        self.args = args

    def print_args(self):
        # self.args.printer args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = 'train' if self.args.train==1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        self.set_seed(666)
        self.print_args()
        return self.args

    def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


