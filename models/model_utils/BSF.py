import torch
from torch import nn
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self, in_dim=24):
        super(Fusion, self).__init__()
        self.dim = in_dim
        self.device='cuda'

        # self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.fc1x = nn.Linear(in_dim, in_dim).to(self.device)
        # self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.fc1y = nn.Linear(in_dim//2, in_dim).to(self.device)



    def forward(self, x, y):
        x = self.fc1x(x)    #初始特征变形后   [4,24]
        y = self.fc1y(y)      #初始特征变形后 [4,24]
        energy = x * y      #图中乘的部分   [4,24]
        attention = F.softmax(energy, dim=-1)  #[4,24]    #图中sigmoid
        attention_x = x * attention    #原输入特征与注意力特征相乘
        attention_y = y * attention    ##原输入特征与注意力特征相乘

        out_x=x+attention_x
        out_y=y+attention_y


        out=out_x+out_y

        return out
