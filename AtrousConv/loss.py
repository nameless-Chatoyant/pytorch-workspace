import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        # Cross Entropy Loss
        target = target.type(torch.LongTensor).cuda()
        loss = self.ce_loss(input, target)
        # Focal Loss
        input = F.softmax(input)
        logit = input.gather(1, target.unsqueeze(1))
        loss = loss * (1 - logit) ** self.gamma

        return loss.sum()


if __name__ == '__main__':
    from torch.autograd import Variable
    criterion = FocalLoss(2)
    input = Variable(torch.randn(3, 5), requires_grad=True)
    target = Variable(torch.LongTensor(3).random_(5))
    res = criterion(input, target)
    print(res)
