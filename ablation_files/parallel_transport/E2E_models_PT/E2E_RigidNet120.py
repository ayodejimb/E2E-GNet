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

def log_map_sphere(x, y):
    inner = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(inner)
    sin_theta = torch.sin(theta) + 1e-8
    return ((theta / sin_theta) * (y - inner * x))

def exp_map_sphere(x, u):
    norm_u = torch.norm(u, dim=-1, keepdim=True) + 1e-8
    cos_term = torch.cos(norm_u)
    sin_term = torch.sin(norm_u)
    return cos_term * x + sin_term * (u / norm_u)

def pole_ladder_transport(a, b, u, n_rungs=1):
    dt = 1.0 / n_rungs
    transported = u.clone()
    for _ in range(n_rungs):
        mid1 = exp_map_sphere(a, dt*transported)

        log_ab = log_map_sphere(a, b)
        mid2 = exp_map_sphere(a, 0.5*log_ab)

        log_mid2_mid1 = log_map_sphere(mid2, mid1)
        sym = exp_map_sphere(mid2, -log_mid2_mid1)

        transported = log_map_sphere(b, sym)
        a = b.clone()

    return transported

def compute_parallel_transport(x):
    B, T, N, D = x.shape
    ref = x[:, 0:1] 
    B_shapes = x[:, :-1]
    C_shapes = x[:, 1:]

    U = log_map_sphere(B_shapes, C_shapes) 
    ref_expand = ref.expand(B, T-1, -1, -1)
    U_A = pole_ladder_transport(B_shapes, ref_expand, U, n_rungs=1)

    first_vec = log_map_sphere(ref, x[:, 0:1])
    transported_vecs = torch.cat([first_vec, U_A], dim=1)
    return transported_vecs

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
        x = self.rot(x)
        x = compute_parallel_transport(x)
        x = x.view(x.size(0),self.num_joints*dims,self.num_frames)
        x = self.conv1(x)
        x = self.pool(self.conv2(x))
        x, _ = self.LSTM(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
