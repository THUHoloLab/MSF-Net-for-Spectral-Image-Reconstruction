import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2,dilation=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3,dilation=3)
        # self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.last_conv = nn.Conv2d(out_channels*3, out_channels, kernel_size=3, padding=1)
        # 判断是否需要用1x1卷积匹配通道数
        self.use_res_conv = (in_channels != out_channels)
        if self.use_res_conv:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        out = torch.cat([x3, x5, x7], dim=1)
        out = self.last_conv(self.relu(out))
        out = out + self.res_conv(x)  # 残差连接
        return self.relu(out)

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super(MultiScaleConvBlock, self).__init__()
        layers = []
        for i in range(num_blocks):
            input_c = in_channels if i == 0 else out_channels
            layers.append(MultiScaleConv(input_c, out_channels))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class MSF_net_fusion(nn.Module):
    def __init__(self, innput_channels=2, spectral_channels=29):
        super(MSF_net_fusion, self).__init__()

        base_channels = 32
        self.fusion1=MultiScaleConvBlock(innput_channels+spectral_channels, base_channels*3,num_blocks=2)
        self.fusion2=MultiScaleConvBlock(base_channels*3, base_channels*3,num_blocks=6)
        self.fusion_last= MultiScaleConvBlock(base_channels*3, spectral_channels,num_blocks=2)

    def forward(self, x):
        x=self.fusion1(x)
        x=self.fusion2(x)
        x=self.fusion_last(x)
        return x

def measure_avg_inference_time(model, input_tensor, device='cuda', repeat=100):
    model.eval()
    input_tensor = input_tensor.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    starter.record()

    with torch.no_grad():
        for _ in range(repeat):
            _ = model(input_tensor)

    ender.record()
    torch.cuda.synchronize()
    total_time = starter.elapsed_time(ender)  # 总时间（ms）
    avg_time = total_time / repeat
    return avg_time

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    CUDA_DEVICE=0
    model = MSF_net_fusion(innput_channels=2, spectral_channels=29).cuda(CUDA_DEVICE)
    x = torch.randn(2, 31, 256, 256).cuda(CUDA_DEVICE)
    out = model(x)
    print(out.shape)
    a=MSF_net_fusion(innput_channels=2, spectral_channels=29)
    summary(a,(31, 256, 256),device='cpu')
    avg_time_ms = measure_avg_inference_time(model, x, device=CUDA_DEVICE, repeat=100)
    print(f"Average inference time: {avg_time_ms:.3f} ms")