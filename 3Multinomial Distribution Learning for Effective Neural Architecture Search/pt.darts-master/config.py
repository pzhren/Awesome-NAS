""" Config class for search/augment """
import argparse
import os
import genotypes as gt
from functools import partial
import torch
import time
import random
import string


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--layers', type=int, default=8, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                            help='weight decay for alpha')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('searchs', self.name)
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)


class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=False, default='CIFAR10',
                            help='CIFAR10 / MNIST / FashionMNIST / cifar10_24 / cifar10_16')
        parser.add_argument('--data_path', help='data_path', default='/userhome/temp_data/cifar10')
        parser.add_argument('--data_loader_type', required=False, help='Torch / DALI', default='Torch')
        parser.add_argument('--batch_size', type=int, default=96, help='batch size')
        parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=600, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int, default=20, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')

        parser.add_argument('--genotype', required=False, help='Cell genotype')
        parser.add_argument('--save_path', default='', help='save_path')
        parser.add_argument('--save_dir', default='/userhome/project/pt.darts/experiment/', help='save_dir')
        parser.add_argument('--file', default='', help='file_save_')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
        time_str = time.asctime(time.localtime()).replace(' ', '_')
        random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        if self.save_path:
            self.path = os.path.join(self.save_dir, self.name, self.save_path)
        else:
            self.path = os.path.join(self.save_dir, self.name, time_str + random_str)
        if os.path.exists(self.path):
            exit('The path is exists!!')
        if self.file:
            file_ = open(self.file)
            lines = file_.readlines()
            for i, line in enumerate(lines):
                self.path = os.path.join('/userhome/project/pt.darts/experiment/', self.name, str(i))
                if os.path.isdir(self.path):
                    continue
                else:
                    self.genotype = line
                    print(line)
                    break
        self.genotype = gt.from_str(self.genotype)
        self.gpus = parse_gpus(self.gpus)


class RandomSearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST / cifar10_24 / cifar10_16')
        parser.add_argument('--data_path', help='data_path', default='/userhome/temp_data/cifar10')
        parser.add_argument('--data_loader_type', required=False, help='Torch / DALI', default='Torch')
        parser.add_argument('--batch_size', type=int, default=96, help='batch size')
        parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=600, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int, default=20, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')

        parser.add_argument('--genotype', required=False, help='Cell genotype')
        parser.add_argument('--save_path', default='', help='save_path')
        parser.add_argument('--save_dir', default='/userhome/project/pt.darts/experiment/random_search', help='save_dir')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        # if self.save_path:
        #     self.path = os.path.join(self.save_dir, self.name, self.save_path)
        # else:
        #     self.path = os.path.join(self.save_dir, self.name, time_str + random_str)
        self.file = False
        for i in range(100):
            self.path = os.path.join(self.save_dir, self.name, str(i))
            if os.path.exists(self.path):
                continue
            else:
                self.file = '/userhome/project/Auto_NAS/experiment/random_architecture/random_10/{0}.txt'\
                    .format(str(i))
                break
        if not self.file:
            exit()
        self.gpus = parse_gpus(self.gpus)
