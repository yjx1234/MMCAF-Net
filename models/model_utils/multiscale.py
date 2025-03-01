import torch
import torch.nn as nn
from timm.models.helpers import named_apply
from functools import partial

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle_3D(x, groups):
    batchsize, num_channels, depth, height, width = x.size()
    channels_per_group = num_channels // groups

    # 重塑张量以进行分组
    x = x.view(batchsize, groups, channels_per_group, depth, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, depth, height, width)
    return x

class CAB_3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB_3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        # 替换为 3D pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # 替换为 3D 卷积
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv3d(self.in_channels, self.reduced_channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv3d(self.reduced_channels, self.out_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)  # [bs, c, 1, 1, 1]
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)  # [bs, c, 1, 1, 1]
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)

class SAB_3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB_3D, self).__init__()
        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        # 使用 3D 卷积
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入张量形状: [bs, c, d, w, h]

        # 对通道维度取平均值 (spatial mean)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [bs, 1, d, w, h]

        # 对通道维度取最大值 (spatial max)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [bs, 1, d, w, h]

        # 合并平均池化和最大池化的结果
        x = torch.cat([avg_out, max_out], dim=1)  # [bs, 2, d, w, h]

        # 通过 3D 卷积
        x = self.conv(x)  # [bs, 1, d, w, h]
        return self.sigmoid(x)

class MSDC_3D(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC_3D, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes  # List of kernel sizes
        self.activation = activation
        self.dw_parallel = dw_parallel  # Whether to process scales in parallel

        # Create depth-wise convolution branches for each kernel size
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                # 3D depth-wise convolution
                nn.Conv3d(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=(kernel_size, kernel_size, kernel_size),  # 3D kernel
                    stride=stride,
                    padding=(kernel_size // 2, kernel_size // 2, kernel_size // 2),  # Same padding
                    groups=self.in_channels,  # Depth-wise convolution
                    bias=False
                ),
                nn.BatchNorm3d(self.in_channels),  # 3D Batch Norm
                act_layer(self.activation, inplace=True)  # Activation
            )
            for kernel_size in self.kernel_sizes
        ])

        # Initialize weights
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:  # Sequential processing (non-parallel)
                x = x + dw_out  # Cumulative sum with input
        return outputs  # Return results from all depth-wise branches

class DCFB(nn.Module):
    '''
    3D Depth-Wise Convolution Fusion Block(DCFB)
    '''

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
            add=True, activation='relu6'):
        super(DCFB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # 检查stride值
        assert self.stride in [1, 2], "Stride must be either 1 or 2."
        # 当stride为1时使用跳跃连接
        self.use_skip_connection = True if self.stride == 1 else False

        # 扩展因子
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
                # 点卷积
                nn.Conv3d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
                nn.BatchNorm3d(self.ex_channels),
                act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC_3D(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                dw_parallel=self.dw_parallel)
        if self.add:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
                # 点卷积
                nn.Conv3d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm3d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv3d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle_3D(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out

class UCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(UCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels // 2
        self.up_dwc = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                    padding=kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm3d(self.in_channels),
                act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
                nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle_3D(x, self.in_channels)
        x = self.pwc(x)
        return x

input_tensor = torch.randn(4, 256, 6, 48, 48)
cab_3d = CAB_3D(in_channels=256)
out1 = cab_3d(input_tensor) * input_tensor
sab_3d = SAB_3D()
out2=sab_3d(out1) * out1
mscb3 = MSCB_3D(in_channels=256, out_channels=256, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2,
                          dw_parallel=True, add=True, activation='relu6')
out3=mscb3(out2)
eucb_ks = 3  # kernel size for eucb
eucb2 = EUCB_3D(in_channels=256, out_channels=256, kernel_size=eucb_ks, stride=eucb_ks // 2)
out4=eucb2(out3)

# 打印输出张量的形状
print("输出张量形状: ", out4.shape)
