# coding=utf-8

#定义cell
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

'''进行所谓的对操作集中的所有操作进行松弛化，即对操作集中的操作进行加权求和'''
class MixedOp(nn.Module):
  def __init__(self, C, stride):
      #op = MixedOp(C=16(初始的通道个数), stride=2 or 1)
    super(MixedOp, self).__init__()

    #这一部分的作用：对在PRIMITIVES中包含pool操作的操作添加BN操作
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False) #在OPS中查找对应的操作
      if 'pool' in primitive:
          #对于有pool操作的执行
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    #cell = Cell(steps=4, multiplier=4, C_prev_prev=48, C_prev=48, C_curr=16, reduction=False, reduction_prev=False)
  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
        #判断前一层是否使用reduction预处理结构
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps #=4
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    # self._bns = nn.ModuleList()
    for i in range(self._steps):
        #i in range(4)
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride) #松弛化离散的操作集
        self._ops.append(op) #_ops总共有14个操作

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0 #偏移量

    #构建Figure1(b):每个中间节点与之前的节点之间都有连接
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      '''offset=0,生成状态s2时，与状态s0和s1都有关系
      #j=0,h=s0:s = sum(self._ops[0](s0, weights[0])
      #j=1,h=s1:s = sum(self._ops[1](s1, weights[1])
      '''
      '''offset=0+2，生成状态s3时，与状态s0，s1和s2都有关系
      #j=0,h=s0:s = sum(self._ops[2](s0, weights[2])
      #j=1,h=s1:s = sum(self._ops[3](s1, weights[3])
      #j=2,h=s2:s = sum(self._ops[4](s2, weights[4])
      '''
      '''offset=2+3
      ......
      '''
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)
    #返回将之前的所有状态进行拼接


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    #model = Network(args.init_channels=16, CIFAR_CLASSES=10, args.layers=8, criterion=nn.CrossEntropyLoss())
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C #=48
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C #（48,16,16）
    self.cells = nn.ModuleList() #存储操作列表
    reduction_prev = False #是否采用reduction结构
    for i in range(layers):
        #总共8层
      if i in [layers//3, 2*layers//3]:
          #在i=2,4的情况下，即第3层和第5层的时候使用reduction结构，一共8层
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      print('第{}层:{},{},{}'.format(i,C_prev_prev, C_prev, C_curr))
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      #cell = Cell(steps=4, multiplier=4, C_prev_prev=48, C_prev=16, C_curr=16, reduction=True/False, reduction_prev=True/False)
      reduction_prev = reduction
      self.cells += [cell] #对cell进行组合
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input) #经过stem进行预处理
    for i, cell in enumerate(self.cells):
      #总共有8层，即总共有8个cell
      #对权重进行处理，已使其对细微变化更加敏感
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights) #执行将cell进行串接
    out = self.global_pooling(s1) #执行全局平均池化
    logits = self.classifier(out.view(out.size(0),-1)) #执行分类
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  #初始化操作的权重
  def _initialize_alphas(self):
    '''在一个小的cell中权重的个数，k=14,如下：
    [n for i in range(4) for n in range(2+i)]
    [0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]
    '''
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES) #操作数=8

    #权重的数量：14*8，初始化两组权重
    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [self.alphas_normal, self.alphas_reduce,]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
    def _parse(weights):
      gene = []
      n = 2
      start = 0
      #self._steps:我认为是用来控制cell中的节点个数，此处=4
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
          #PRIMITIVES[k_best]：挑选出具有最大权重的操作
          #j in [0,1,2,3,4],代表的是当前节点与几号节点相连，中间节点只取权重最大的两个操作
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

