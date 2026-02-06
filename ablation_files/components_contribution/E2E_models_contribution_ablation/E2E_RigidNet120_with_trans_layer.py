from layers.rigidtransform import RigidTransform
from layers.rigidtransforminit import RigidTransformInit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry.core as tgmc

depth_1 = 128
kernel_size_1 = 3
stride_size = 2
depth_2 = 64
kernel_size_2 = 1
num_hidden = 512
num_labels = 120
dims = 3


def inv_exp_batch(X, Y):
    skeleton = torch.einsum('bnik,btnk->btni', X, Y)
    tr = torch.einsum('btii->bt', skeleton).abs()
    tr = torch.clamp(tr, -1.0, 1.0) 
    theta = torch.acos(tr)  
    theta = torch.where(torch.sin(theta) < 1e-4, theta.new_tensor(0.1), theta)
    sin_theta = torch.sin(theta) + 1e-8
    scale = (theta / sin_theta).unsqueeze(-1).unsqueeze(-1) 
    invExp = scale * (Y - torch.cos(theta).unsqueeze(-1).unsqueeze(-1) * X)
    invExp = torch.nan_to_num(invExp, nan=0.0, posinf=1e4, neginf=-1e4)

    return invExp

class RigidNet120(nn.Module):
    def __init__(self,  mod = 'RigidTransform', num_frames = 100, num_joints = 25, r=0):
        super(RigidNet120, self).__init__()
        self.num_channels = num_joints * dims
        self.mod = mod
        self.num_frames = num_frames
        self.num_joints = num_joints
        if mod == 'RigidTransform':
            self.rot = RigidTransform(num_frames,num_joints,r)
        elif mod == 'RigidTransformInit':
            self.rot = RigidTransformInit(num_frames,num_joints,r)
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.num_channels, depth_1,kernel_size=kernel_size_1, stride=stride_size),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(depth_1, depth_2, kernel_size=kernel_size_2, stride=stride_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.LSTM = nn.LSTM(12, hidden_size=12, bidirectional =True)  
        self.pool=nn.MaxPool1d(kernel_size=2, stride=stride_size)
        self.fc1 = nn.Sequential(
            nn.Linear(depth_2*24, num_hidden),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(num_hidden, num_labels),
        )
        
    def forward(self, x):
        ref = x[0:1, 0:1]  
        x_t = inv_exp_batch(ref, x) 
        x = x_t
        x = x.reshape(x.size(0),self.num_joints*dims,self.num_frames)
        x = self.conv1(x)
        x = self.pool(self.conv2(x))
        x, _ = self.LSTM(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
