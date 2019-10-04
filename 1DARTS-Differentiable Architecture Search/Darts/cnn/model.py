# coding=utf-8
import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path

"""Pytorch中神经网络模块化接口nn的了解"""
"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法。
定义自已的网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中，
    不具有可学习参数的层(如ReLU)可放在构造函数中，也可不放在构造函数中(而在forward中使用nn.functional来代替)
    
    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
    在forward函数中可以使用任何Variable支持的函数，毕竟在整个pytorch构建的图中，是Variable在流动。还可以使用
    if,for,print,log等python语法.
    
    注：Pytorch基于nn.Module构建的模型中，只支持mini-batch的Variable输入方式，
    比如，只有一张输入图片，也需要变成 N x C x H x W 的形式：
    
    input_image = torch.FloatTensor(1, 28, 28)
    input_image = Variable(input_image)
    input_image = input_image.unsqueeze(0)   # 1 x 1 x 28 x 28
    
"""
#定义Cell
class Cell(nn.Module):
  #cell = Cell(genotype='DARTS', C_prev_prev=108, C_prev=108, C_curr=36, reduction:由所在层数决定, reduction_prev=false）
  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    
    #打印两个输入层的通道个数和当前层的通道个数
    print(C_prev_prev,C_prev,C)
    
    #对第一个输入执行的操作preprocess0
    if reduction_prev:
        self.preprocess0 = FactorizedReduce(C_prev_prev, C) 
        #FactorizedReduce：operations中的一种操作{nn.ReLU-->nn.Conv_1,nn.ReLU-->nn.Conv_2,torch.cat()-->BN}，卷积核的步长=2，输出的通道数减半
    else:
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0) #ReLUConvBN：operations中的一种操作
        #ReLUConvBN(C_prev_prev, C, 1, 1, 0){nn.ReLU-->nn.Conv2d-->nn.BN}正常操作
    
    #对第2个输入执行的操作preprocess1
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce) #将操作的名字和具体的参数分开
      '''
      DARTS_V2,reduce:
      op_names=('max_pool_3x3', 'max_pool_3x3', 'skip_connect', 'max_pool_3x3', 'max_pool_3x3', 'skip_connect', 'skip_connect', 'max_pool_3x3')
      indices=(0, 1, 2, 1, 0, 2, 2, 1)
      '''
      concat = genotype.reduce_concat
      #reduce_concat=[2, 3, 4, 5]
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2 #等于4
    self._concat = concat
    self.multiplier = len(concat) #等于4，reduce_concat=[2, 3, 4, 5]

    self._ops = nn.ModuleList() #存放操作集
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True) #OPS操作的定义放在from operations import *
      self._ops += [op]
    self._indices = indices #DARTS_V2_normal情况下indices=[0,1,0,1,1,0,0,2]

  def forward(self, s0, s1, drop_prob):
    #s0, s1 = s1, cell(s0, s1, self.drop_path_prob=0.2)
    #self.preprocess首先经过一个简单的卷积网络操作，得到s0，s1
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1] #存储之前的状态
    #self._steps=4
    for i in range(self._steps):
      #初始状态,i=0时：h1=s0，h2=s1
      h1 = states[self._indices[2*i]] #[0,0,1,0]
      h2 = states[self._indices[2*i+1]] #[1,1,0,2]
      # 最后一个0，2，0状态的那个来构成跳跃连接
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      #进行dropout操作
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2 #每个中间状态由之前的两个操作加和而来
      states += [s]
    #将最后4个状态cat起来
    return torch.cat([states[i] for i in self._concat], dim=1)

#相当于是一个2层卷积1层全连接的一个完整的分类器
class AuxiliaryHeadCIFAR(nn.Module):
  # self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary=第二个输入的通道数, num_classes=10)
  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    '''
    model = Network(args.init_channels = C = 36, CIFAR_CLASSES = num_classes = 10,
    args.layers = 20, args.auxiliary = 'action=store_true', default=False', genotype = 'DARTS')
    C:通道的个数
    '''
    super(NetworkCIFAR, self).__init__()
    self._layers = layers #模型的层数
    self._auxiliary = auxiliary

    stem_multiplier = 3
    C_curr = stem_multiplier * C
    
    #串联网络操作
    '''
     nn.Conv2d:3-输入通道，C_curr-输出通道，3-卷积核大小
    '''
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C #通道数：上上个层=3*36，上层=3×36，当前层=36
    
    #nn.ModuleList()仅仅类似于python中的list类型，只是将一系列层装入列表，并没有实现forward()方法
    self.cells = nn.ModuleList() #用来存储层的空列表
    reduction_prev = False
    
    
    for i in range(layers):
      #在第6层和第13层采用reduction结构，在其他层采用正常结构
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
       
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell] #将每一层构建的cell进行串联
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr #cell.multiplier=4
      if i == 2*layers//3: #i==12
        C_to_auxiliary = C_prev
    
    #auxiliary:action=store_true
    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1) #二维的自适应平均池化，class torch.nn.AdaptiveAvgPool2d(output_size)，output_size为目标输出的大小
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input) #stem：是卷积+BN操作，构造最初的两个输入s0,s1
    for i, cell in enumerate(self.cells):
	  #得到当前的层数i，结构cell
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        #如果i==12
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux

