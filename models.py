import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# DCGAN model, fully convolutional architecture
class _netG_1(nn.Module):
    def __init__(self, ngpu, nz, nc , ngf, n_extra_layers_g, pix_shuf, imageSize):
        super(_netG_1, self).__init__()
        self.ngpu = ngpu
        #self.nz = nz
        #self.nc = nc
        #self.ngf = ngf
        if pix_shuf:
            self.conv1 = self.upsample_4x(nz, ngf*8*16, 1, 1, 0) #kernel of size 1 because its 1D latent
            self.conv2 = self.upsample_2x(ngf*8, ngf*4*4, 3, 1, 1)
            self.conv3 = self.upsample_2x(ngf*4, ngf*2*4, 3, 1, 1)
            self.conv4 = self.upsample_2x(ngf*2, ngf*4, 3, 1, 1)
            if imageSize == 128:
                self.conv_out = nn.Sequential(
                                self.upsample_2x(ngf, ngf*4, 3, 1, 1),
                                nn.BatchNorm2d(ngf),
                                nn.LeakyReLU(0.2, inplace=True),
                                self.upsample_2x(ngf, nc*4, 3, 1, 1)
                                )
            else:
                self.conv_out = self.upsample_2x(ngf, nc*4, 3, 1, 1)
        else:
            self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
            self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
            self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
            self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
            if imageSize == 128:
                self.conv_out = nn.Sequential(
                                nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
                                nn.BatchNorm2d(ngf),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
                                )
            else:
                self.conv_out = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

        self.main = nn.Sequential(
                # state size. nz x 1 x 1
                self.conv1,
                nn.BatchNorm2d(ngf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*8) x 4 x 4
                self.conv2,
                nn.BatchNorm2d(ngf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*4) x 8 x 8
                self.conv3,
                nn.BatchNorm2d(ngf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*2) x 16 x 16
                self.conv4,
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True)
                # state size. (ngf) x 32 x 32
                )
                
        # Extra layers
        for t in range(n_extra_layers_g):
            self.main.add_module('extra-layers-{0}.{1}.conv'.format(t, ngf),
                            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False))
            self.main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, ngf),
                            nn.BatchNorm2d(ngf))
            self.main.add_module('extra-layers-{0}.{1}.relu'.format(t, ngf),
                            nn.LeakyReLU(0.2, inplace=True))

        self.main.add_module('final_layer.deconv', 
                         self.conv_out) # 5,3,1 for 96x96
        self.main.add_module('final_layer.tanh', 
                         nn.Tanh())

    def upsample_2x(self, inch, outch, ksize, stride, pad):
        layers = [nn.Conv2d(inch, outch, ksize, stride, pad, bias=False), 
                nn.PixelShuffle(2)]
        return nn.Sequential(*layers)
    def upsample_4x(self, inch, outch, ksize, stride, pad):
        layers = [nn.Conv2d(inch, outch, ksize, stride, pad, bias=False), 
                nn.PixelShuffle(4)]
        return nn.Sequential(*layers)



    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids), 0

class _netD_1(nn.Module):
    def __init__(self, ngpu, nz, nc, ndf,  n_extra_layers_d, norm, imageSize):
        super(_netD_1, self).__init__()
        self.ngpu = ngpu
        if norm == 'BatchNorm':
            self.norm1 = nn.BatchNorm2d(ndf * 2)
            self.norm2 = nn.BatchNorm2d(ndf * 4)
            self.norm3 = nn.BatchNorm2d(ndf * 8)
            self.norm_128 = nn.BatchNorm2d(ndf)
        else:
            self.norm1 = LayerNormalization(ndf * 2)
            self.norm2 = LayerNormalization(ndf * 4)
            self.norm3 = LayerNormalization(ndf * 8)
            self.norm_128 = LayerNormalization(ndf)
        if imageSize == 128:
            self.conv1 = nn.Sequential(
                            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 5,3,1 for 96x96
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False), # 5,3,1 for 96x96
                            self.norm_128
                            )
        else:
            self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 5,3,1 for 96x96

        main = nn.Sequential(
            self.conv1,
            # input is (nc) x 96 x 96
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            self.norm1,
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            self.norm2,
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            self.norm3,
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

        # Extra layers
        for t in range(n_extra_layers_d):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, ndf * 8),
                            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False))
            if norm == 'BatchNorm':
                main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, ndf * 8),
                                nn.BatchNorm2d(ndf * 8))
            else:
                main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, ndf * 8),
                                LayerNormalization(ndf * 8))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, ndf * 8),
                            nn.LeakyReLU(0.2, inplace=True))


        main_2 = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
                )
        #main.add_module('final_layers.conv', nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
        #main.add_module('final_layers.sigmoid', nn.Sigmoid())
        # state size. 1 x 1 x 1
        self.main = main
        self.main_2 = main_2

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        # Avoid multi-gpu if only using one gpu
        emb = self.main(input)
        output = self.main_2(emb)
        return output.mean(0).view(1), emb
        #return output.view(-1, 1)
    




class _netD_2(nn.Module):
    def __init__(self, ngpu, nz, nc , ndf):
        super(_netD_2, self).__init__()
        self.ngpu = ngpu
        self.convs = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1024, 4, 1, 0, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            # state size. 1024 x 1 x 1
        )
        self.fcs = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),            
            nn.Sigmoid()
            )
    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.convs, input, gpu_ids)
        output = self.convs(input)
        output = self.fcs(output.view(-1,1024))
        return output.view(-1, 1)

# with z decoder and fc layers
class _netG_2(nn.Module):
    def __init__(self, ngpu, nz, nc , ngf):
        super(_netG_2, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.fcs = nn.Sequential(
            # input is Z, going into a convolution
            # state size. nz x 1 x 1
            nn.Linear(nz, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            )
        
        self.decode_fcs = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, nz),
            )

        self.convs = nn.Sequential(
            # 1024x1x1
            nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 96 x 96
        )
    def forward(self, input):
        input = self.fcs(input.view(-1,self.nz))
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        z_prediction = self.decode_fcs(input)
        input = input.view(-1,1024,1,1)
        output = nn.parallel.data_parallel(self.convs, input, gpu_ids)
        return output, z_prediction


# DCGAN model with fc layers
class _netG_3(nn.Module):
    def __init__(self, ngpu, nz, nc , ngf):
        super(_netG_3, self).__init__()
        self.ngpu = ngpu
        self.fcs = nn.Sequential(
            # input is Z, going into a convolution
            # state size. nz x 1 x 1
            nn.Linear(nz, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            )
        self.convs = nn.Sequential(
            # 1024x1x1
            nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 96 x 96
        )
    def forward(self, input):
        input = self.fcs(input.view(-1,nz))
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        input = input.view(-1,1024,1,1)
        return nn.parallel.data_parallel(self.convs, input, gpu_ids)

class BaseModel(nn.Module):
    def forward(self, x):
        gpu_ids = None
        if isinstance(x.data, torch.cuda.FloatTensor) and self.num_gpu > 1:
            gpu_ids = range(self.num_gpu)
        if gpu_ids:
            return nn.parallel.data_parallel(self.main, x, gpu_ids)
        else:
            return self.main(x)

class DiscriminatorCNN(BaseModel):
    def __init__(self, input_channel=3, z_num=128, repeat_num=5, hidden_num=128, num_gpu=1):
        super(DiscriminatorCNN, self).__init__()
        self.num_gpu = num_gpu

        # Encoder
        layers = []
        layers.append(nn.Conv2d(input_channel, hidden_num, 3, 1, 1))
        layers.append(nn.ELU(True))
        layers.append(nn.BatchNorm2d(hidden_num))

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            layers.append(nn.Conv2d(prev_channel_num, channel_num, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.BatchNorm2d(channel_num))

            if idx < repeat_num - 1:
                layers.append(nn.Conv2d(channel_num, channel_num, 3, 2, 1))
                #layers.append(nn.MaxPool2d(2))
                #layers.append(nn.MaxPool2d(1, 2))
            else:
                layers.append(nn.Conv2d(channel_num, channel_num, 3, 1, 1))

            layers.append(nn.ELU(True))
            layers.append(nn.BatchNorm2d(channel_num))
            prev_channel_num = channel_num

        self.conv1_output_dim = [channel_num, 8, 8]

        self.conv1 = torch.nn.Sequential(*layers)
        self.fc1 = nn.Linear(8*8*channel_num, z_num)

        # Decoder
        self.conv2_input_dim = [hidden_num, 8, 8]
        self.fc2 = nn.Linear(z_num, np.prod(self.conv2_input_dim))
        
        layers = []
        for idx in range(repeat_num):
            layers.append(nn.Conv2d(hidden_num, hidden_num, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.BatchNorm2d(hidden_num))
            layers.append(nn.Conv2d(hidden_num, hidden_num, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.BatchNorm2d(hidden_num))

            if idx < repeat_num - 1:
                layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        layers.append(nn.Conv2d(hidden_num, input_channel, 3, 1, 1))
        layers.append(nn.Tanh())
        #layers.append(nn.ELU(True))

        self.conv2 = torch.nn.Sequential(*layers)

    def main(self, x):
        conv1_out = self.conv1(x).view(-1, np.prod(self.conv1_output_dim))
        fc1_out = self.fc1(conv1_out)

        fc2_out = self.fc2(fc1_out).view([-1] + self.conv2_input_dim)
        conv2_out = self.conv2(fc2_out)
        return conv2_out

