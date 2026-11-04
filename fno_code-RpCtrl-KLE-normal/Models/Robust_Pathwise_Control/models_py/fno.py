"""
Fourier Neural Operator (FNO2d) implementations in native PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    '''2D spectral convolution layer.'''
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    '''Fourier Neural Operator for 2D image-to-image mapping.'''
    def __init__(self, modes1=10, modes2=10, width=64, in_channels=1, out_channels=1, dropout=0.1, img_size=(33, 33), padding=3):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding

        # Input channels + 2 (for x and y coordinate grids)
        self.p = nn.Conv2d(in_channels + 2, self.width, 1)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.q1 = nn.Conv2d(self.width, 128, 1)
        self.q2 = nn.Conv2d(128, out_channels, 1)

        self.dropout = nn.Dropout(p=dropout)

        # Register static coordinate grid buffer
        self.register_buffer('grid', self.get_grid(img_size))

    def get_grid(self, shape):
        batchsize = 1
        size_x, size_y = shape[0], shape[1]
        gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(torch.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).permute(0, 3, 1, 2)  # [1, 2, size_x, size_y]

    def forward(self, x):
        # 1. Inject coordinate grids -> [B, 1, 33, 33] to [B, 3, 33, 33]
        grid = self.grid.repeat(x.shape[0], 1, 1, 1)
        x = torch.cat((x, grid), dim=1)

        # 2. Apply zero-padding to mitigate non-periodic boundary effects
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x = self.p(x)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        x = self.dropout(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        x = self.dropout(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)
        x = self.dropout(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)
        x = self.dropout(x)

        x = self.q1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.q2(x)

        # 3. Crop back to original spatial size
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
            
        return x