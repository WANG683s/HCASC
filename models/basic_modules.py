import torch
from torch import nn
import torch.nn.functional as F


def make_window(x, window_size, stride=1, padding=0):
    """
    Args:
        x: (B, W, H, C)
        window_size (int): window size

    Returns:
        windows: (B*N,  ws**2, C)
    """
    x = x.permute(0, 3, 1, 2).contiguous()
    B, C, W, H = x.shape
    windows = F.unfold(x, window_size, padding=padding, stride=stride) # B, C*N, #of windows
    windows = windows.view(B, C, window_size**2, -1) #   B, C, ws**2, N
    windows = windows.permute(0, 3, 2, 1).contiguous().view(-1, window_size, window_size, C) # B*N, ws**2, C
    
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
    
def binarize(integer, num_bits=8):   
    """Turn integer tensor to binary representation.        
    Args:           
    integer : torch.Tensor, tensor with integers           
    num_bits : Number of bits to specify the precision. Default: 8.       
    Returns:           
    Tensor: Binary tensor. Adds last dimension to original tensor for           
    bits.    
    """   
    dtype = integer.type()   
    exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)   
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))   
    out = integer.unsqueeze(-1) / 2 ** exponent_bits   
    return (out - (out % 1)) % 2


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, stride=1, use_cwa=True, bottleneck=False, use_se=False):     
        super(double_conv, self).__init__()
        self.use_cwa = use_cwa
        self.bottleneck = bottleneck


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch)
        )

        if self.use_cwa:
            self.cwa = CWA(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        sc = x
        x = self.conv2(x)
        if self.use_cwa:
            x = self.cwa(x)
        x += sc
        x = F.relu(x, inplace=True)
        return x


class CWA(nn.Module):
    def __init__(self, ch_in, reduction=24):
        super(CWA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  
        y = self.fc(y).view(b, c, 1, 1)  
        return x * y.expand_as(x)  



class inconv(nn.Module):
    '''
    inconv only changes the number of channels
    '''

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super(down, self).__init__()
        self.mpconv = double_conv(in_ch, out_ch, stride=2, use_se=use_se)

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, op="none", use_se=False):
        super(up, self).__init__()
        self.bilinear = bilinear
        self.op = op
        self.mixup_ratio = 0.95
        assert op in ["concat", "none", "add", 'mix']

        self.conv = double_conv(in_ch, out_ch, use_se=use_se)

    def forward(self, x1, x2=None):
        if x2 is not None:
            if torch.is_tensor(x2):
                x1 = F.interpolate(x1, x2.size()[-2:], mode='bilinear', align_corners=False)
            else:
                x1 = F.interpolate(x1, x2, mode='bilinear', align_corners=False)
        else:
            x1 = F.interpolate(x1, scale_factor=2,  mode='bilinear', align_corners=False)

        if self.op == "concat":
            x = torch.cat([x2, x1], dim=1)
        elif self.op == 'add':
            x = x1 + x2
        else:
            x = x1
            
        x = self.conv(x)

        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

