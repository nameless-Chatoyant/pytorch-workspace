import cv2
import torch
from dataset import Dataset
#import torch
import os
from torch import optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from model import Net

from loss import FocalLoss
from cfgs.config import cfg

torch.manual_seed(cfg.seed)
if cfg.cuda:
    torch.cuda.manual_seed(cfg.seed)

# Dataset
train_dataset = Dataset('train')
# test_dataset = Dataset('test')

# Input Pipeline
kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.cuda else {}
train_loader = data_utils.DataLoader(dataset=train_dataset, 
                                        batch_size=cfg.batch_size, 
                                        shuffle=True, **kwargs)
'''
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=cfg.batch_size, 
                                        shuffle=False, **kwargs)
'''

net = Net()
if cfg.cuda:
    net.cuda()
criterion = FocalLoss(cfg.gamma)
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum=0.9)

def train(epoch):
    net.train()
    for batch_idx, (images, labels) in enumerate(train_loader):  

        images, labels = Variable(images), Variable(labels)
        images, labels = images.type(torch.FloatTensor), labels.type(torch.IntTensor)
        if cfg.cuda:
            images, labels = images.cuda(), labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                %(epoch+1, cfg.num_epochs, batch_idx+1, len(train_dataset)//cfg.batch_size, loss.data[0]))

# if __name__ == '__main__':
for epoch in range(cfg.num_epochs):
    train(epoch)
