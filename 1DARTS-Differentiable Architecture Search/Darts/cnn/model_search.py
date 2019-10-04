# coding=utf-8

#定义cell
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

use_cuda = torch.cuda.is_available()

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
        #i in range(4),这里的i代表是第i个节点
      for j in range(2+i):
        # 当reduction=True且j=0,1时，stride=2
        # reduction在模型的1/3,2/3层处等于True
        # 限定j=0,1是对cell的两个输入的feature_map的大小进行变更，以便对cell内的其他节点的大小进行相应的变化
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride) #松弛化离散的操作集
        self._ops.append(op) #_ops总共有14组操作

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0 #偏移量，为了让相应的state乘以对应的weights

    #构建Figure1(b):每个中间节点与之前的节点之间都有连接
    for i in range(self._steps):
      #i是cell中间生成节点的标号
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      '''
      offset=0:生成状态s2时，与状态s0和s1都有关系:
        #j=0,h=s0:s0 = self._ops[0](s0, weights[0])
        #j=1,h=s1:s1 = self._ops[1](s1, weights[1])
        sum(s0,s1)
      offset=0+2:生成状态s3时，与状态s0，s1和s2都有关系:
        #j=0,h=s0:s0 = self._ops[2](s0, weights[2])
        #j=1,h=s1:s1 = self._ops[3](s1, weights[3])
        #j=2,h=s2:s2 = self._ops[4](s2, weights[4])
        sum(s0,s1,s2)
      offset=2+3:
      ......
      '''
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)
    #返回将之前的所有状态进行拼接


class Network(nn.Module):
  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    '''

    :param C: 初始的通道个数
    :param num_classes: 类别个数
    :param layers: 层数
    :param criterion: 使用的损失函数
    :param steps: 中间节点的个数
    :param multiplier: 乘数因子
    :param stem_multiplier:
    '''
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
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C #（48,48,16）
    self.cells = nn.ModuleList() #存储操作列表
    reduction_prev = False #是否采用reduction结构
    for i in range(layers):
        #总共8层
      if i in [layers//3, 2*layers//3]:
          #在i=2,4的情况下，即第2层和第5层的时候使用reduction结构，一共8层，这个是可以自己设定的
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      if i == 0:
        print('层数=8,s0(上上层/s1-->s0),s1(上层/cell-->s1:torch_cat(4*C_curr)))，C_curr')
      print('第{}层:{},{},{}'.format(i,C_prev_prev, C_prev, C_curr))
      """C_prev_prev, C_prev, C_curr
      第0层:48,48,16
      第1层:48,64,16
      第2层:64,64,32
      第3层:64,128,32
      第4层:128,128,32
      第5层:128,128,64
      第6层:128,256,64
      第7层:256,256,64
      """
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      #cell = Cell(steps=4, multiplier=4, C_prev_prev=48, C_prev=48, C_curr=16, reduction=True/False, reduction_prev=True/False)
      reduction_prev = reduction
      self.cells += [cell] #对cell进行堆叠
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1) #二维自适应平均池化，返回的大小是1*1的
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    if use_cuda:
      model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    else:
      model_new = Network(self._C, self._num_classes, self._layers, self._criterion)

    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input) #对初始的输入图像仅有3个通道经过stem进行预处理生成48个通道
    for i, cell in enumerate(self.cells):
      #总共有8层，即总共有8个cell
      #对权重进行处理，已使其对细微变化更加敏感
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights) #执行将cell进行串接
    out = self.global_pooling(s1) #执行二维自适应平均池化,batch_size*256*1*1*1
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
    if use_cuda:
      self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True) #requires_grad=True要求梯度
      self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True) #requires_grad=True要求梯度
    else:
      self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops), requires_grad=True)  # requires_grad=True要求梯度
      self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops), requires_grad=True)
    self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
    #用来生成cell的节点之间的连接结构
    def _parse(weights):
      gene = []
      n = 2
      start = 0
      #self._steps:我认为是用来控制cell中的节点个数，此处=4
      #i代表的是生成中间节点的第几个，i=0，表示第2个节点，...,i=3,表示第5个节点，
      # 因此最后将[2,3,4,5],即range(2,6)个节点进行torch.cat()操作
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy() #在这一部分就已经相应位置的alpha取了出来
        '''weights.size()=[14,8],14个连线，每个线有8个操作
        i=0,[start:end]=[0:2],n=3
        i=1,[start:end]=[2:5],n=4
        i=2,[start:end]=[5:9],n=5
        i=3,[start:end]=[9:14],n=6
        '''

        #选出权重最大的两条边,即：第i个节点与哪两条边相连
        '''range(i+2)
        i=0,range(2)=[0,1]
        i=1,range(3)=[0,1,2]
        i=2,range(4)=[0,1,2,3]
        i=3,range(5)=[0,1,2,3,4]
        '''
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]

        #为上面选出的两条边分别确定一个权重最大的操作：j代表边的位置：即起始点的连接节点，相应的i代表对应的终点连接的节点。
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
          #PRIMITIVES[k_best]：挑选出具有最大权重的操作
          #j in [0,1,2,3,4],代表的是当前节点与几号节点相连，生成的中间节点[2,3,4,5]只保留权重最大的两个操作
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
  '''genotype的一个示例：
  genotype = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_3x3', 4), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('skip_connect', 3), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))
  '''

