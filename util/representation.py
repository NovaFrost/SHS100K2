import torch
import torch.utils
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import sys, os
import numpy as np
import argparse

class Net(nn.Module):
    def __init__(self, weight):
        super(Net, self).__init__()
        num_classes, middle = 10000, weight
        self.fc1 = nn.Linear(900, middle)
        self.fc2 = nn.Linear(middle, num_classes)

    def forward(self, x):
        x = self.fc1(x.view(x.size(0), -1))
        output = self.fc2(x)
        return output, x

def load_2dfm(file_list, in_dir='youtube_2dfm_npy/'):
    inputs, labels = [], []
    for filename in file_list:
        in_path = in_dir + filename + '.npy'
        input = np.load(in_path) 
        inputs.append(input)
    return np.array(inputs)

def norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def generate_rep(net, two_dfm, batch_size=100):
    new_feas = None
    for i in range(0, len(two_dfm), batch_size):
        inputs = two_dfm[i: (i + batch_size if i + batch_size < len(two_dfm) else len(two_dfm)), :]
        inputs = Variable(torch.from_numpy(inputs).cuda())
        outputs, new_fea = net(inputs)
        outputs, new_fea = outputs.data.cpu().numpy(), new_fea.data.cpu().numpy()
        if new_feas is not None:
            new_feas = np.concatenate((new_feas, new_fea), axis=0)
        else:
            new_feas = new_fea
    new_feas = norm(new_feas)
    return new_feas

net = torch.load('100')
flist, in_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
with open(flist, 'r') as fp:
    l = [row.rstrip() for row in fp]
two_dfm = load_2dfm(l, in_dir)
new_feas = generate_rep(net, two_dfm)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
for i, new_fea in enumerate(new_feas):
    np.save(os.path.join(out_dir, l[i]), new_fea)
