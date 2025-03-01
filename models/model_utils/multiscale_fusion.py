import torch
import math
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from .BSF import Fusion


class Cross_Attention(nn.Module):
    def __init__(self, query_dim: int, keyval_dim: int, num_heads: int):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(keyval_dim, query_dim)
        self.v_proj = nn.Linear(keyval_dim, query_dim)
        self.num_heads = num_heads
        self.query_dim=query_dim
        self.scale = (query_dim // num_heads) ** -0.5

    def forward(self, query: Tensor, keyval: Tensor) -> Tensor:
        # query: [B, N, D_q] (如N=1)
        # keyval: [B, M, D_kv] (如M=1)
        # query=query.unsqueeze(1)  # [4, 24] -> [4, 1, 24]
        # keyval=keyval.unsqueeze(1)  # [4, 24] -> [4, 1, 24]
        q = self.q_proj(query)  # [B, N, D_q]
        k = self.k_proj(keyval)  # [B, M, D_q]
        v = self.v_proj(keyval)  # [B, M, D_q]

        # 分头
        q = q.view(q.size(0), q.size(1), self.num_heads, -1).permute(0, 2, 1, 3)  # [B, H, N, C]
        k = k.view(k.size(0), k.size(1), self.num_heads, -1).permute(0, 2, 1, 3)  # [B, H, M, C]
        v = v.view(v.size(0), v.size(1), self.num_heads, -1).permute(0, 2, 1, 3)  # [B, H, M, C]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, M]
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v  # [B, H, N, C]
        out = out.permute(0, 2, 1, 3).contiguous().view(out.size(0), -1, self.num_heads * (self.query_dim // self.num_heads))

        return out+query   # 残差连接

class MultiScale_Fusion(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.norm_img=nn.BatchNorm1d(embedding_dim)
        self.norm_tab=nn.BatchNorm1d(embedding_dim)
        self.down0 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(0.5)
        )
        self.down1 = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim//2),  #[bs,2,12]
                nn.ReLU(),
                nn.Dropout(0.5)
        )
        #self.down1 = nn.Linear(embedding_dim, embedding_dim//2)  #[bs,2,12]
        self.down2 = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 4),   #[bs,2,6]
                nn.ReLU(),
                nn.Dropout(0.5)
        )


        self.cross_attention1=Cross_Attention(query_dim=embedding_dim,keyval_dim=embedding_dim,num_heads=4)
        self.cross_attention2 = Cross_Attention(query_dim=embedding_dim//2,keyval_dim=embedding_dim//2, num_heads=2)
        self.cross_attention3 = Cross_Attention(query_dim=embedding_dim//4,keyval_dim=embedding_dim//4, num_heads=2)
        self.sample_img1=nn.Linear(embedding_dim, embedding_dim//2)  #[bs,2,12]
        self.sample_tab1 = nn.Linear(embedding_dim, embedding_dim // 2)  # [bs,2,12]
        self.sample_img2=nn.Linear(embedding_dim//2, embedding_dim//4) #[bs,2,6]
        self.sample_tab2 = nn.Linear(embedding_dim // 2, embedding_dim // 4)  # [bs,2,6]

        self.SF1=Fusion(embedding_dim//2)
        self.SF2=Fusion(embedding_dim)


    def forward(self,img,tab):
        img=self.norm_img(img)
        tab=self.norm_tab(tab)
        img = img.unsqueeze(1)  # [4, 24] -> [4, 1, 24]
        tab = tab.unsqueeze(1)  # [4, 24] -> [4, 1, 24]
        '''
        x = torch.cat((img, tab), dim=1)  # [4, 2, 24]
        x=self.down0(x)
        x1=self.down1(x)  #scale2
        x2=self.down2(x)  #scale3
        out1=self.cross_attention1(x) #[bs,2,24]

        x1=x1+self.sample1(out1)
        out2=self.cross_attention2(x1) #[bs,2,12]

        x2=x2+self.sample2(out2)
        out3=self.cross_attention3(x2)  #[bs,2,6]
        '''
        img0=img
        tab0=tab
        img1=self.down1(img)
        tab1=self.down1(tab)
        img2=self.down2(img)
        tab2=self.down2(tab)

        out_img1=self.cross_attention1(img0,tab0)
        # out_img1=self.drop1(out_img1)
        out_tab1=self.cross_attention1(tab0,img0)
        # out_tab1 = self.drop1(out_tab1)
        out1 = torch.cat([out_img1, out_tab1], dim=1)  # [4,2,24]
        out1 = out1.mean(dim=1)  # 全局平均 [4,24]
        img1=img1+self.sample_img1(out_img1)
        tab1=tab1+self.sample_tab1(out_tab1)
        out_img2 = self.cross_attention2(img1, tab1)
        # out_img2 = self.drop2(out_img2)
        out_tab2 = self.cross_attention2(tab1, img1)
        # out_tab2 = self.drop2(out_tab2)
        out2 = torch.cat([out_img2, out_tab2], dim=1)  # [4,2,12]
        out2 = out2.mean(dim=1)  # 全局平均 [4,12]

        img2 = img2 + self.sample_img2(out_img2)
        tab2 = tab2 + self.sample_tab2(out_tab2)
        out_img3 = self.cross_attention3(img2, tab2)
        # out_img3 = self.drop3(out_img3)
        out_tab3 = self.cross_attention3(tab2, img2)
        # out_tab3 = self.drop3(out_tab3)
        out3 = torch.cat([out_img3, out_tab3], dim=1)  # [4,2,6]
        out3 = out3.mean(dim=1)  # 全局平均 [4,6]


        f1=self.SF1(out2,out3)
        f2=self.SF2(out1,f1)

        # 分解成两个形状为 [4, 1, 24] 的特征
        #img_fea = f2[:, 0:1, :]  # 提取第一个特征
        #tab_fea = f2[:, 1:2, :]  # 提取第二个特征
        #img_fea = img_fea.squeeze(1)  # [4,1,24] -> [4,24]
        #tab_fea = tab_fea.squeeze(1)  # [4,1,24] -> [4,24]

        return f2

class Simple_Fuse(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
    ) -> None:
        super().__init__()
        self.norm_img = nn.BatchNorm1d(embedding_dim)
        self.norm_tab = nn.BatchNorm1d(embedding_dim)
        self.cross_attention1 = Cross_Attention(query_dim=embedding_dim, keyval_dim=embedding_dim, num_heads=4)

    def forward(self,img,tab):
        img = self.norm_img(img)
        tab = self.norm_tab(tab)
        img = img.unsqueeze(1)  # [4, 24] -> [4, 1, 24]
        tab = tab.unsqueeze(1)  # [4, 24] -> [4, 1, 24]
        out_img = self.cross_attention1(img, tab)
        out_tab = self.cross_attention1(tab, img)
        out1 = torch.cat([out_img, out_tab], dim=1)  # [4,2,24]
        out1 = out1.mean(dim=1)  # 全局平均 [4,24]
        return out1

'''
attention=Cross_Prompt_Attention(embedding_dim=24,num_heads=4)
img = torch.randn(4,24)
tab = torch.randn(4,24)
prompt_embedding=torch.randn(4,2,24)
img_fea,tab_fea=attention(img,tab,prompt_embedding)
print(img_fea.shape,tab_fea.shape)
'''
