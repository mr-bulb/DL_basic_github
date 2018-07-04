#!/usr/bin/env Python
# coding=utf-8

##************************************************************************** 
##-------------------------------------------------------------------------- 
## The Implement of LeNet-5 model by Pytorch
## 
## Author         Hao Zeng 
## version        dev 0.1.0 
## Date           2018.07.04 
##
## Description    使用PyTorch实现LeNet-5模型
##
##-------------------------------------------------------------------------- 
##************************************************************************** 

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# Print Training settings' Information
print(args)

# Write Training settings to file
file = r'./LeNet-5_loss.txt'
with open(file, 'a+') as f:
    f.write(str(args) + "\n")

file1 = r'./LeNet-5_accuracy.txt'
with open(file1, 'a+') as f:
    f.write(str(args) + "\n")


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# data loader in train step
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# data loader in evaluate step, evaluate the losss and accuracy in test set
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# data loader in evaluate step, evaluate loss and accuracy in train set
train_eval_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


# Define Network Structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84 , 10)

    def forward(self, x):
        x = F.sigmoid(F.avg_pool2d(self.conv1(x), 2)) #conv->avg_pool->sigmoid   28x28x1->24x24x6->12x12x6
        x = F.sigmoid(F.avg_pool2d(self.conv2(x), 2)) #conv->avg_pool->sigmoid   12x12x6->8x8x16->4x4x16
        x = x.view(-1, 256) # 16*4*4 = 256
        x = F.sigmoid(self.fc1(x))#fc->relu
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

model = Net()


# Test whether use cuda or not
if args.cuda:
    print("\n\n********cuda********\n\n")
    model.cuda()
else:
    print("\n\n********no cuda********\n\n")

# Define Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
         
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    print("\n\n************Eval********\n\n")
    print("Eval train set\n");

    model.eval()
    train_loss = 0
    correct = 0
    data_num = 0
    max_data_num = 10000
    for batch_idx, (data, target) in enumerate(train_loader):
        if data_num >= max_data_num:
            print("break num:{0}".format(data_num))
            break

        data_num = data_num+len(data)

        if data_num%1000 == 0:
            print("test process:{:.2f}".format(1.0*data_num/max_data_num))

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        output = model(data)

        train_loss += F.nll_loss(output, target, size_average=False).data[0]#Variable.data
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    train_loss /= data_num # loss function already averages over batch size
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        train_loss, correct, data_num,
        100. * correct / data_num))
    train_accuracy = 1.0 * correct / data_num

    print("data num: {0}".format(data_num))
    print("train batch size: {0} train loader len : {1} dataset len : {2}".format(len(data), len(train_loader), len(train_loader.dataset)))

#----------------------------------------------------
    print("Eval test set\n");    
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        output = model(data)

        test_loss += F.nll_loss(output, target, size_average=False).data[0]#Variable.data
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader.dataset) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = 1.0 * correct / len(test_loader.dataset)

    print("test batch size: {0} test loader len : {1} dataset len : {2}".format(len(data), len(test_loader), len(test_loader.dataset)))


    file = r'./LeNet-5_loss.txt'
    with open(file, 'a+') as f:
        f.write(str(train_loss) + " " + str(test_loss) + "\n")

    file1 = r'./LeNet-5_accuracy.txt'
    with open(file1, 'a+') as f:
        f.write(str(train_accuracy/100.0) + " " + str(test_accuracy/100.0) + "\n")


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)


  
# Write Enter as a split to different hyperparameter      
file = r'./LeNet-5_loss.txt'
with open(file, 'a+') as f:
    f.write("\n")

file1 = r'./LeNet-5_accuracy.txt'
with open(file1, 'a+') as f:
    f.write("\n")


