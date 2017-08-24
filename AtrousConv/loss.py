import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1
    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    index = index.type(torch.LongTensor)
    return mask.scatter_(1, index, ones)

def one_hot_dev(index, classes):
    one_hot = np.zeros((index.size(1), classes))
    index_np = index.data.cpu().numpy() - 1
    one_hot[np.arange(index.size(1)), index_np] = 1
    return torch.from_numpy(one_hot).type(torch.IntTensor).cuda()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, input, target):
        y = one_hot_dev(target, input.size(1))
        print(y)

        logit = F.softmax(input)
        logit = logit.clamp(self.eps, 1. - self.eps)

        # logit = logit.permute(0,2,3,1)#.view(logit.size(0), -1, input.size(1))
        # print(logit)
        # logit = logit.view(1,-1,4)
        # print(logit)
        loss = -1 * y * torch.log(logit)
        loss = loss * (1 - logit) ** self.gamma

        return loss.sum()


if __name__ == '__main__':
    from torch.autograd import Variable
    criterion = FocalLoss(2)
    print(criterion)