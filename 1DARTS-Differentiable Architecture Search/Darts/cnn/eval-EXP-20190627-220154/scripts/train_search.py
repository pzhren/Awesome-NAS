# coding=utf-8
import os
import sys
import time
import glob
import numpy as np
import torch
import utils #文件模块
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network #文件模块
from architect import Architect #文件模块

#参数
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping') #梯度裁剪
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1) #如果cuda不可用则抛出异常

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu) #gpu device = 0
  logging.info("args = %s", args)
  #args = Namespace(arch_learning_rate=0.0003, arch_weight_decay=0.001, batch_size=64, cutout=False, cutout_length=16, data='../data', drop_path_prob=0.3, epochs=50, gpu=0, grad_clip=5, init_channels=16, layers=8, learning_rate=0.025, learning_rate_min=0.001, model_path='saved_models', momentum=0.9, report_freq=50, save='search-EXP-20190624-154343', seed=2, train_portion=0.5, unrolled=False, weight_decay=0.0003)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  #定义一个8层的模型
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  #06/25 03:00:02 PM param size = 1.930618MB

  #定义优化器
  optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                              momentum=args.momentum, weight_decay=args.weight_decay)
  #model.parameters()：这里说明模型优化的包括cell里面的参数alpha

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  #导入数据，将训练集train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)的一半做训练队列，另一半做验证队列
  #训练队列
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=1)
  #torch.utils.data.sampler.SubsetRandomSampler：采用无放回随机采样

  #验证队列
  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=1)

  # 优化器的学习率调整策略：采用CosineAnnealingLR，余弦退火调整学习率
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  #执行训练过程
  for epoch in range(args.epochs):
    scheduler.step() #这里应该是得到一个参数簇与学习率组成的词典：param_group['lr'] = lr

    # 得到学习率
    lr = scheduler.get_lr()[0]
    logging.info('epoch = %d, lr = %e', epoch, lr)

    #得到当前cell的结构
    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    #打印权重
    # print(F.softmax(model.alphas_normal, dim=-1))
    # print(F.softmax(model.alphas_reduce, dim=-1))

    # training，训练集训练阶段
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc) #当前epoch的平均acc

    # validation，验证集训练阶段
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc) #当前epoch的平均acc

    utils.save(model, os.path.join(args.save, 'weights.pt')) #保存模型参数


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  #train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    #对训练集按照小的batch_size进行训练
    model.train() #让模型变成训练状态，DropOut和BN部分可进行调整
    n = input.size(0) #样本个数,其实等于batch_size

    #封装改变数据类型
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad() #梯度清零
    logits = model(input) #预测
    loss = criterion(logits, target) #计算损失

    loss.backward() #反向传播
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip) #梯度裁剪
    optimizer.step() #参数更新，更新参数W和alpha二者同时更新

    #计算预测准确率
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data[0], n) #包含当前batch在内的所有训练样本的平均损失
    top1.update(prec1.data[0], n) #包含当前batch在内的所有训练样本的平均准确率
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  #最终返回每个epoch的平均准确率与平均损失
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval() #让模型变成测试状态，固定DropOut和BN部分

  for step, (input, target) in enumerate(valid_queue):

    #封装数据集
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits = model(input) #预测
    loss = criterion(logits, target) #计算损失

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

