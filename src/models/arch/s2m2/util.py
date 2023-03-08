import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.weight_norm import WeightNorm

act = torch.nn.ReLU()


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)

        return scores


def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    if torch.cuda.is_available():
        y_onehot = y_onehot.cuda()

    y_onehot.zero_()
    x = inp.type(torch.LongTensor)
    if torch.cuda.is_available():
        x = x.cuda()

    x = torch.unsqueeze(x, 1)
    y_onehot.scatter_(1, x, 1)

    return Variable(y_onehot, requires_grad=False)
    # return y_onehot


def mixup_data(x, y, lam):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if torch.cuda.is_available():
        index = index.cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam