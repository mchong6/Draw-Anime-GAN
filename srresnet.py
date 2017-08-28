
import torch
import torch.nn as nn
import math
import torch.nn.init as init
from torch.autograd import Variable

class LayerNormalization(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormalization, self).__init__()
        
        self.eps = eps
        self.hidden_size = hidden_size
        self.a2 = nn.Parameter(torch.ones(1, hidden_size, 1, 1), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size, 1, 1), requires_grad=True)
        
    def forward(self, z):
        mu = torch.mean(z, dim=1, keepdim=True)
        sigma = torch.std(z, dim=1, keepdim=True)

        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a2.expand_as(z) + self.b2.expand_as(z)
        return ln_out

class _Residual_Block(nn.Module): 
    def __init__(self, norm):
        super(_Residual_Block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = norm
        if self.norm == 'BatchNorm':
            self.norm1 = nn.BatchNorm2d(64)
            self.norm2 = nn.BatchNorm2d(64) 
        else:
            self.norm1 = LayerNormalization(64) 
            self.norm2 = LayerNormalization(64) 
    def forward(self, x): 
        identity_data = x
        output = self.conv1(x)
        output = self.norm1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = self.relu(output)
        output = torch.add(output,identity_data)
        return output 

class NetG(nn.Module):
    def __init__(self, imageSize):
        super(NetG, self).__init__()
        
        self.imageSize = imageSize
        self.conv_input = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.linear = nn.Linear(128, 512)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.residual = self.make_layer(_Residual_Block, 6)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.BatchNorm2d(64)

        if imageSize == 128:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)
        self.tanh = nn.Tanh()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(norm="BatchNorm"))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear(x.view(-1, 128))
        out = out.view(-1, 2, 16, 16)
        out = self.relu(self.conv_input(out))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        out = self.upscale(out)
        out = self.tanh(self.conv_output(out))
        return out, 0


class NetD(nn.Module):
    def __init__(self, norm, imageSize):
        super(NetD, self).__init__()
        
        self.imageSize = imageSize
        self.norm = norm
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        if imageSize == 128:
            self.conv_input = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
                        self.relu,
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
                        self.relu,
                        LayerNormalization(64)
                    )
        else:
            self.conv_input = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
                        self.relu
                    )
        self.ln1 = LayerNormalization(64)
        self.ln2 = LayerNormalization(64)
        self.residual = self.make_layer(_Residual_Block, 5)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=16, stride=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(self.norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_input(x)
        residual = out
        out = self.residual(out)
        out = self.ln1(self.conv_mid(out))
        out = torch.add(out,residual)
        out = self.relu(self.conv_output(out))
        out = self.ln2(out)
        out_d = self.conv_fc(out)
        out_d = self.sigmoid(out_d)
        return out_d
        out_d = out_d.mean(0).view(1)
        return out_d
