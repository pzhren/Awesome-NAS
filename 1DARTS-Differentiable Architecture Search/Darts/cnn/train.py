# -*- coding:utf-8 -*-
import os
import sys
import time
import glob #文件查找操作的一个模块
import numpy as np
import torch
import utils #文件模块
import logging
import argparse
import torch.nn as nn
import genotypes #文件模块
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network #文件模块

#实例化解释器
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping') 
#引入梯度裁剪，减缓梯度爆炸与消失，梯度会被约束到一定的范围内

args = parser.parse_args()

#创建实验目录，保存脚本文件
args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
#Experiment dir : eval-EXP-20190618-170816
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

'''
logging.basicConfig(**kwargs):
函数用于指定“要记录的日志级别”、“日志格式”、“日志输出位置”、
“日志文件的打开模式”等信息，其他几个都是用于记录各个级别日志的函数。
'''
log_format = '%(asctime)s %(message)s'
#打印日期时间
logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p')
#06/18 05:08:17

fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def train(train_queue, model, criterion, optimizer):
  #train_acc, train_obj = train(train_queue, model, criterion, optimizer)
  objs = utils.AvgrageMeter() #用来存储当前损失
  top1 = utils.AvgrageMeter() #用于存储最好的准确率
  top5 = utils.AvgrageMeter()
  model.train() #打开训练模式

  for step, (input, target) in enumerate(train_queue):
    #train_queue：训练集有batch_size=96，每次计算一个batch_size,即以下操作是在一个batch_size下进行的
    print()
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad() #清零梯度
    logits, logits_aux = model(input)
    loss = criterion(logits, target) #计算损失
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux #args.auxiliary_weight=0.4：weight for auxiliary loss，辅助损失权重
    loss.backward() #进行反向传播
    
    #进行梯度裁剪，设置阈值，当梯度大于或小于阈值时，更新梯度为阈值
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip) #梯度裁剪的阈值被设置为5
    optimizer.step() #更新参数

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5)) #返回第1和第5的准确率
    n = input.size(0) #n=batch_size=96
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    #每report_freq=50汇报一次
    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    #打印当前batch之前loss及所有变量的均值
  #最后返回当前epoch的准确率和loss的均值
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval() #打开评价模式

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def main():
  #判断是否有GPU可用
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu) #设置当前设备
  cudnn.benchmark = True #加速计算
  torch.manual_seed(args.seed) #为cpu设置随机数种子
  cudnn.enabled=True #cuDNN是一个GPU加速深层神经网络原语库，开启cudnn
  torch.cuda.manual_seed(args.seed)#为当前GPU设置随机种子
  #打印日志信息
  logging.info('gpu device = %d' % args.gpu)
  #06/27 06:48:36 PM gpu device = 0
  logging.info("args = %s", args)
  '''
  06/27 06:55:45 PM args = Namespace(arch='DARTS', auxiliary=False, auxiliary_weight=0.4, batch_size=96, cutout=False, cutout_length=16, data='../data', drop_path_prob=0.2, epochs=600, gpu=0, grad_clip=5, init_channels=36, layers=20, learning_rate=0.025, model_path='saved_models', momentum=0.9, report_freq=50, save='eval-EXP-20190618-170816', seed=0, weight_decay=0.0003)
  '''

  genotype = eval("genotypes.%s" % args.arch) #应该是输出一个框架类型。eval() 函数用来执行一个字符串表达式，并返回表达式的值
  
  #from model import NetworkCIFAR as Network #文件模块
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  #model = Network(通道个数=36, CIFAR_CLASSES=10, 总体layers=20, args.auxiliary使用辅助塔, genotype=框架类型)
  model = model.cuda()

  #打印模型参数的大小，即所占空间
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  #param size = 3.349342MB
    
  criterion = nn.CrossEntropyLoss() #定义损失函数
  criterion = criterion.cuda()
  
  #定义优化器
  optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                              momentum=args.momentum, weight_decay=args.weight_decay)

  #获得预处理之后的训练集和验证集
  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  #获取数据集
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  '''
  Files already downloaded and verified
  Files already downloaded and verified
  '''  
  
  
  #对数据进行封装为Tensor，主要用来读取数据集
  '''
  pin_memory:If True, the data loader will copy tensors into CUDA pinned memory before returning them,在数据返回前，是否将数据复制到CUDA内存中
  num_workers:加快数据导入速度,工作者数量，默认是0。使用多少个子进程来导入数据。设置为0，就是使用主进程来导入数据。注意：这个数字必须是大于等于0的，不能太大，2的时候报错
  '''
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)

  #优化器的学习率调整策略：采用CosineAnnealingLR，余弦退火调整学习率
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  
  #默认epochs=600
  for epoch in range(args.epochs):
        
    scheduler.step() #这里应该是得到一个参数簇与学习率组成的词典：param_group['lr'] = lr
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    #epoch 0 lr 2.500000e-02
    
    #进行dropout：大小与模型的深度相关，模型深度越深，dropout的概率越大，最大0.2
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    #调用下面定义的函数train()
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    '''
    train_queue:要训练的队列
    model：采用的model；
    criterion：定义的损失函数
    optimizer：所采用的优化器
    '''
    logging.info('train_acc %f', train_acc) #打印当前epoch在训练集上的精度
    
    #计算在验证集上的精度
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    #保存模型参数，这里主要是在cell里面的参数alpha已经固定的情况下去优化model里面的参数w
    utils.save(model, os.path.join(args.save, 'weights.pt'))
    
    

if __name__ == '__main__':
  main() 

