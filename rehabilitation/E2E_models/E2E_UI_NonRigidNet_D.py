from layers.nonrigidtransform import NonRigidTransform
from layers.nonrigidtransforminit import NonRigidTransformInit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry.core as tgmc

depth_1 = 32
kernel_size_1 = 3
stride_size = 2
depth_2 = 128
kernel_size_2 = 1
num_hidden = 32
dims = 3
num_labels = 1

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

class NonRigidNet(nn.Module):
    def __init__(self, mod = 'NonRigidTransform', num_frames = 150, num_joints = 22, r=0, dml_run=0,
                 glob_in=False, glob_h=False, loc_in=False, loc_h=False):
        super(NonRigidNet, self).__init__()
        self.num_channels = num_joints * dims
        self.mod = mod
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.glob_in = glob_in
        self.glob_h = glob_h
        self.loc_in = loc_in
        self.loc_h = loc_h
        if mod == 'NonRigidTransform':
            self.rot = NonRigidTransform(num_frames,num_joints, r)
        elif mod == 'NonRigidTransformInit':
            self.rot = NonRigidTransformInit(num_frames,num_joints, r)
        self.interm_conv = nn.Sequential(
            nn.Conv1d(self.num_channels, depth_2, kernel_size=kernel_size_1, stride=stride_size),
            nn.ReLU(),
            nn.Dropout(0.50)
        )
        self.LSTM = nn.LSTM(depth_2, hidden_size=12, bidirectional =True)  
        self.pool=nn.MaxPool1d(kernel_size=2, stride=stride_size)
        self.fc1 = nn.Sequential(
            nn.Linear(37*24, num_hidden),
            nn.Dropout(0.50)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(num_hidden, num_labels),
        )
        torch.manual_seed(dml_run)
        self.mat_adap_glob_h = nn.Parameter(2 * torch.rand(1) -1)
        self.mat_adap_glob_in = nn.Parameter(2 * torch.rand(self.num_joints,1) -1)
        self.mat_adap_loc_in = nn.Parameter(2 * torch.rand(self.num_frames, self.num_joints,1) -1)
        self.mat_adap_loc_h = nn.Parameter(2 * torch.rand(self.num_frames,1) -1)
        
    def forward(self, x):
        x = x.view(x.size(0),self.num_frames*self.num_joints,1,dims)
        x = self.rot(x)
        x = x.view(x.size(0), self.num_frames, self.num_joints, dims)
        ref = x[0:1, 0:1]  
        x_t = inv_exp_batch(ref, x)
        x = x_t
        if self.glob_h:
            diag_mat_adap = torch.diag(torch.exp(self.mat_adap_glob_h.repeat(3)))
            diag_mat_adap = diag_mat_adap.unsqueeze(0).repeat(self.num_joints, 1, 1)
            x = torch.einsum('ijkl,kno->ijkl', x, diag_mat_adap)
        if self.glob_in:
            diag_mat_adap = torch.exp(self.mat_adap_glob_in.expand(-1, 3))
            diag_mat_adap = torch.diag_embed(diag_mat_adap)
            x = torch.einsum('ijkl,kno->ijkl', x, diag_mat_adap)
        if self.loc_in: 
            diag_mat_adap = torch.exp(self.mat_adap_loc_in.expand(-1,-1,3))
            diag_mat_adap = torch.diag_embed(diag_mat_adap)
            x = torch.einsum('ijkl,jkno->ijkl', x, diag_mat_adap)
        if self.loc_h: 
            diag_mat_adap = torch.exp(self.mat_adap_loc_h.repeat(1, 3))
            diag_mat_adap = torch.diag_embed(diag_mat_adap).unsqueeze(1)
            diag_mat_adap = diag_mat_adap.repeat(1, self.num_joints, 1, 1)
            x = torch.einsum('ijkl,jkno->ijkl', x, diag_mat_adap)
        x = x.view(x.size(0),self.num_joints*dims,self.num_frames)
        x = self.pool(self.interm_conv(x))   
        x = x.permute(0,2,1)
        x,  _= self.LSTM(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
