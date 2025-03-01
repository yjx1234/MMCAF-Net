import torch
from torch import nn
import torch.nn.functional as F

class BFPUBlock(nn.Module):
    def __init__(self, in_dim=32):
        super(BFPUBlock, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        x_q = self.query_conv(x)    #初始特征变形后
        y_k = self.key_conv(y)      #初始特征变形后
        energy = x_q * y_k      #图中乘的部分
        attention = self.sig(energy)    #图中sigmoid
        attention_x = x * attention    #原输入特征与注意力特征相乘
        attention_y = y * attention    ##原输入特征与注意力特征相乘

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))      #原输入特征与attention特征相加，然后进行最终特征变化
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]   #x_gamma的通道维度是x的两倍，每个通道维度分别与

        y_gamma = self.gamma2(torch.cat((y, attention_y), dim=1))    #原输入特征与attention特征相加，然后进行最终特征变化
        y_out = y * y_gamma[:, [0], :, :] + attention_y * y_gamma[:, [1], :, :]

        x_s = x_out + y_out

        return x_s

class BFPUBlock_3D(nn.Module):
    def __init__(self, in_dim1=1024):
        super(BFPUBlock_3D, self).__init__()

        # 针对两个输入特征分别定义卷积
        self.query_conv = nn.Conv3d(in_dim1, in_dim1, kernel_size=3, stride=1, padding=1, bias=True)
        self.key_conv = nn.Conv3d(in_dim1, in_dim1, kernel_size=3, stride=1, padding=1, bias=True)
        
        #调整通道
        self.adjust_c=nn.Conv3d(in_channels=in_dim1//2, out_channels=in_dim1, kernel_size=1, stride=1, padding=0)

        # 融合两个特征的注意力权重
        self.gamma1 = nn.Conv3d(in_dim1 + in_dim1, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.gamma2 = nn.Conv3d(in_dim1 + in_dim1, 2, kernel_size=3, stride=1, padding=1, bias=True)

        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        # 调整输入特征 y 的深度维度以匹配 x 的深度维度
        if x.shape[2:] != y.shape[2:]:  # 检查 (depth, height, width) 是否一致
            y = F.interpolate(y, size=x.shape[2:], mode='trilinear', align_corners=True)
            y=self.adjust_c(y)

        print(y.shape)
        # 生成 Query 和 Key 特征
        x_q = self.query_conv(x)  # [bs, c1, d, w, h]
        y_k = self.key_conv(y)    # [bs, c2, d, w, h]
        #print(y_k.shape)

        # 计算能量（逐元素相乘）
        energy = x_q * y_k  # [bs, c, d, w, h]
        attention = self.sig(energy)  # 应用 Sigmoid 得到注意力权重

        # 计算注意力加权的特征
        attention_x = x * attention  # [bs, c1, d, w, h]
        attention_y = y * attention  # [bs, c2, d, w, h]

        # 拼接原始特征与注意力特征，然后通过 Gamma 进一步调整
        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))  # [bs, 2, d, w, h]
        x_out = x * x_gamma[:, [0], :, :, :] + attention_x * x_gamma[:, [1], :, :, :]  # [bs, c1, d, w, h]

        y_gamma = self.gamma2(torch.cat((y, attention_y), dim=1))  # [bs, 2, d, w, h]
        y_out = y * y_gamma[:, [0], :, :, :] + attention_y * y_gamma[:, [1], :, :, :]  # [bs, c2, d, w, h]

        # 最终结果是两个输出特征的加和
        x_s = x_out + y_out

        return x_s

if __name__ == '__main__':
    model = BFPUBlock(in_dim=768).cuda()
    bfpu_3d=BFPUBlock_3D(in_dim1=1024).cuda()
    input_tensor1 = torch.randn(4, 1024, 2,12, 12).cuda()
    input_tensor2 = torch.randn(4, 512, 3,24, 24).cuda()

    P = bfpu_3d(input_tensor1,input_tensor2)
    raise ValueError("P的形状是{}".format(P.shape))
