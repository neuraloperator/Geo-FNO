import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class Module(torch.nn.Module):
    '''Standard module format. 
    '''
    def __init__(self):
        super(Module, self).__init__()
        self.activation = None
        self.initializer = None
        
        self.__device = None
        self.__dtype = None
        
    @property
    def device(self):
        return self.__device
        
    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
        elif d == 'gpu':
            self.cuda()
        else:
            raise ValueError
        self.__device = d
    
    @dtype.setter    
    def dtype(self, d):
        if d == 'float':
            self.to(torch.float)
        elif d == 'double':
            self.to(torch.double)
        else:
            raise ValueError
        self.__dtype = d

    @property
    def Device(self):
        if self.__device == 'cpu':
            return torch.device('cpu')
        elif self.__device == 'gpu':
            return torch.device('cuda')
        
    @property
    def Dtype(self):
        if self.__dtype == 'float':
            return torch.float32
        elif self.__dtype == 'double':
            return torch.float64

    @property
    def act(self):
        if self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'relu':
            return torch.relu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'elu':
            return torch.elu
        else:
            raise NotImplementedError
    
    @property        
    def Act(self):
        if self.activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif self.activation == 'relu':
            return torch.nn.ReLU()
        elif self.activation == 'tanh':
            return torch.nn.Tanh()
        elif self.activation == 'elu':
            return torch.nn.ELU()
        else:
            raise NotImplementedError

    @property
    def weight_init_(self):
        if self.initializer == 'He normal':
            return torch.nn.init.kaiming_normal_
        elif self.initializer == 'He uniform':
            return torch.nn.init.kaiming_uniform_
        elif self.initializer == 'Glorot normal':
            return torch.nn.init.xavier_normal_
        elif self.initializer == 'Glorot uniform':
            return torch.nn.init.xavier_uniform_
        elif self.initializer == 'orthogonal':
            return torch.nn.init.orthogonal_
        elif self.initializer == 'default':
            if self.activation == 'relu':
                return torch.nn.init.kaiming_normal_
            elif self.activation == 'tanh':
                return torch.nn.init.orthogonal_
            else:
                return lambda x: None
        else:
            raise NotImplementedError
            
class StructureNN(Module):
    '''Structure-oriented neural network used as a general map based on designing architecture.
    '''
    def __init__(self):
        super(StructureNN, self).__init__()
        
    def predict(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.Dtype, device=self.Device)
        return self(x).cpu().detach().numpy() if returnnp else self(x)


class FNN(StructureNN):
    '''Fully connected neural networks.
    '''
    def __init__(self, ind, outd, layers=2, width=50, activation='relu', initializer='default', softmax=False):
        super(FNN, self).__init__()
        self.ind = ind
        self.outd = outd
        self.layers = layers
        self.width = width
        self.activation = activation
        self.softmax = softmax
        
        self.modus = self.__init_modules()
        
    def forward(self, x):
        for i in range(1, self.layers):
            LinM = self.modus['LinM{}'.format(i)]
            NonM = self.modus['NonM{}'.format(i)]
            x = NonM(LinM(x))
        x = self.modus['LinMout'](x)
        if self.softmax:
            x = nn.functional.softmax(x, dim=-1)
        return x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.layers > 1:
            modules['LinM1'] = nn.Linear(self.ind, self.width)
            modules['NonM1'] = self.Act
            for i in range(2, self.layers):
                modules['LinM{}'.format(i)] = nn.Linear(self.width, self.width)
                modules['NonM{}'.format(i)] = self.Act
            modules['LinMout'] = nn.Linear(self.width, self.outd)
        else:
            modules['LinMout'] = nn.Linear(self.ind, self.outd)
            
        return modules

class DeepONet(StructureNN):
    '''Deep operator network.
    Input: [batch size, branch_dim + trunk_dim]
    Output: [batch size, 1]
    '''

    def __init__(self, branch_dim, trunk_dim, branch_depth=2, trunk_depth=3, width=50,
                 activation='relu'):
        super(DeepONet, self).__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.activation = activation
        
        self.modus = self.__init_modules()
        self.params = self.__init_params()

        
    def forward(self, x):
        x_branch, x_trunk = x[..., :self.branch_dim], x[..., -self.trunk_dim:]
        # x_branch = self.modus['Branch'](x_branch)
        for i in range(1, self.branch_depth):
            x_branch = self.modus['BrActM{}'.format(i)](self.modus['BrLinM{}'.format(i)](x_branch))
        x_branch = self.modus['BrLinM{}'.format(self.branch_depth)](x_branch)

        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        return torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias']

    def trunk_forward(self, x):
        x_trunk = x[..., -self.trunk_dim:]
        
        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        return x_trunk
    
    def branch_forward(self, x):
        x_branch = x[..., :self.branch_dim]
        
        for i in range(1, self.branch_depth):
            x_branch = self.modus['BrActM{}'.format(i)](self.modus['BrLinM{}'.format(i)](x_branch))
        x_branch = self.modus['BrLinM{}'.format(self.branch_depth)](x_branch)
        
        return x_branch
        
    def __init_modules(self):
        modules = nn.ModuleDict()
        # modules['Branch'] = FNN(self.branch_dim, self.width, self.branch_depth, self.width,
        #                         self.activation, self.initializer)
        if self.branch_depth > 1:
            modules['BrLinM1'] = nn.Linear(self.branch_dim, self.width)
            modules['BrActM1'] = self.Act
            for i in range(2, self.branch_depth):
                modules['BrLinM{}'.format(i)] = nn.Linear(self.width, self.width)
                modules['BrActM{}'.format(i)] = self.Act
            modules['BrLinM{}'.format(self.branch_depth)] = nn.Linear(self.width, self.width)
        else:
            modules['BrLinM{}'.format(self.branch_depth)] = nn.Linear(self.branch_dim, self.width)
            

        modules['TrLinM1'] = nn.Linear(self.trunk_dim, self.width)
        modules['TrActM1'] = self.Act
        for i in range(2, self.trunk_depth):
            modules['TrLinM{}'.format(i)] = nn.Linear(self.width, self.width)
            modules['TrActM{}'.format(i)] = self.Act
        return modules

            
    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        return params


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        # x = (x - self.mean) / (self.std + self.eps)
        # return x

        # x -= self.mean
        # x /= (self.std + self.eps)
        return (x - self.mean) / (self.std + self.eps)
    
    def encode_(self, x):
        # x = (x - self.mean) / (self.std + self.eps)
        # return x

        x -= self.mean
        x /= (self.std + self.eps)
        

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x * std) + mean
        # return x

        # x *= std 
        # x += mean
        return (x * std) + mean

    def decode_(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x * std) + mean
        # return x

        x *= std 
        x += mean

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
