import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class TabEncoder(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.l1 = nn.Linear(7, 24)
        self.l2 = nn.Linear(24, 24)
        self.a = nn.ReLU()
        self.d = nn.Dropout(0.2)

    def forward(self,t):
        t=self.l1(t)
        t=self.a(t)
        t=self.d(t)
        t=self.l2(t)
        return t

    def args_dict(self):
        model_args={}
        return model_args
