from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable, grad
import numpy as np

### load project files
import models
import srresnet
from models import weights_init

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', default='./faces', help='path to dataset')
parser.add_argument('--workers', type=int, default=12, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=300000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , type=int, default=1, help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outDir', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--model', required=True, help='DCGAN | RESNET | IGAN | DRAGAN | BEGAN')
parser.add_argument('--d_labelSmooth', type=float, default=0.1, help='for D, use soft label "1-labelSmooth" for real samples')
parser.add_argument('--n_extra_layers_d', type=int, default=0, help='number of extra conv layers in D')
parser.add_argument('--n_extra_layers_g', type=int, default=1, help='number of extra conv layers in G')
parser.add_argument('--pix_shuf'  , type=int, default=1, help='Use pixel shuffle instead of deconvolution')
parser.add_argument('--white_noise'  , type=int, default=0, help='Add white noise to inputs of discriminator to stabilize training')
parser.add_argument('--lambda_'  , type=int, default=10, help='Weight of gradient penalty (DRAGAN)')
parser.add_argument('--binary', action='store_true', help='z from bernoulli distribution, with prob=0.5')
parser.add_argument('--lr_decay_every', type=int, default=3000, help='decay lr this many iterations')
parser.add_argument('--save_step', type=int, default=10000, help='save weights every 50000 iterations ')

# simply prefer this way
# arg_list = [
#     '--dataRoot', '/home/jielei/data/danbooru-faces',
#     '--workers', '12',
#     '--batchSize', '128',
#     '--imageSize', '64',
#     '--nz', '100',
#     '--ngf', '64',
#     '--ndf', '64',
#     '--niter', '80',
#     '--lr', '0.0002',
#     '--beta1', '0.5',
#     '--cuda', 
#     '--ngpu', '1',
#     '--netG', '',
#     '--netD', '',
#     '--outDir', './results',
#     '--model', '1',
#     '--d_labelSmooth', '0.1', # 0.25 from imporved-GAN paper 
#     '--n_extra_layers_d', '0',
#     '--n_extra_layers_g', '1', # in the sense that generator should be more powerful
# ]

opt = parser.parse_args()
# opt = parser.parse_args(arg_list)
print(opt)
if opt.model == 'DRAGAN' or opt.model == 'RESNET':
    #norm = 'LayerNorm'
    norm = 'LayerNorm'
else:
    norm = 'BatchNorm'

# Make directories
opt.outDir = './results/' + opt.outDir
opt.modelsDir = opt.outDir + '/models'
opt.imDir = opt.outDir + '/images'

# Recursively create image and model directory
try:
    os.makedirs(opt.imDir)
except OSError:
    pass
try:
    os.makedirs(opt.modelsDir)
except OSError:
    pass

opt.manualSeed = random.randint(1,10000) # fix seed, a scalar
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

def calc_gradient_penalty_DRAGAN(netD, X):
    alpha = torch.rand(X.size(0), 1, 1, 1)
    alpha = alpha.expand(X.size()).cuda()

    rand = torch.rand(X.size()).cuda()
    x_hat = Variable(alpha * X.data + (1 - alpha) * (X.data + 0.5 * X.data.std() * rand), requires_grad=True).cuda()
    pred_hat = netD(x_hat)
    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = opt.lambda_ * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

def lowerbound(input):
    output = torch.cat([input, epsilon], 0)
    output = torch.max(output)
    return output

    
nc = 3
ngpu = opt.ngpu
nz = opt.nz
ngf = opt.ngf
ndf = opt.ndf
n_extra_d = opt.n_extra_layers_d
n_extra_g = opt.n_extra_layers_g

dataset = dset.ImageFolder(
    root=opt.dataRoot,
    transform=transforms.Compose([
            transforms.Scale(opt.imageSize),
            # transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # bring images to (-1,1)
        ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=opt.workers)
loader = iter(dataloader)

# load models 
if opt.model == 'DCGAN' or opt.model == 'DRAGAN':
    netG = models._netG_1(ngpu, nz, nc, ngf, n_extra_g, opt.pix_shuf, opt.imageSize)
    netD = models._netD_1(ngpu, nz, nc, ndf, n_extra_d, norm, opt.imageSize)
elif opt.model == 'IGAN':
    netG = models._netG_2(ngpu, nz, nc, ngf)
    netD = models._netD_2(ngpu, nz, nc, ndf)
elif opt.model == 'RESNET':
    netG = srresnet.NetG(opt.imageSize)
    netD = srresnet.NetD(norm, opt.imageSize)
elif opt.model == 'BEGAN':
    netG = models._netG_1(ngpu, nz, nc, ngf, n_extra_g, opt.pix_shuf, opt.imageSize)
    netD = models.DiscriminatorCNN()

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()
criterion_MSE = nn.MSELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
additive_noise = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
real_embed = torch.FloatTensor(1, 512, 4, 4).fill_(0.)
fake_embed = torch.FloatTensor(1, 512, 4, 4).fill_(0.)
if opt.binary:
    bernoulli_prob = torch.FloatTensor(opt.batchSize, nz, 1, 1).fill_(0.5)
    fixed_noise = torch.bernoulli(bernoulli_prob)
else:
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor([1])
real_label = 1
fake_label = 0
epsilon = torch.FloatTensor([1e-9])
#BEGAN parameters
gamma = .7
lambda_k = 0.001
k = 0.
lambda_c = 0.03
alpha = 0.6

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterion_MSE.cuda()
    input, label, additive_noise = input.cuda(), label.cuda(), additive_noise.cuda()
    noise, fixed_noise, epsilon = noise.cuda(), fixed_noise.cuda(), epsilon.cuda()
    real_embed, fake_embed = real_embed.cuda(), fake_embed.cuda()
    
input = Variable(input)
label = Variable(label)
additive_noise = Variable(additive_noise)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
epsilon = Variable(epsilon)
real_embed, fake_embed = Variable(real_embed), Variable(fake_embed)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

def adjust_learning_rate(optimizer, niter):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.95 ** (niter // opt.lr_decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

for iteration in range(1, opt.niter+1):
    try: 
        data = loader.next()
    except StopIteration:
        loader = iter(dataloader)
        data = loader.next()

    start_iter = time.time()
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    netD.zero_grad()
    real_cpu, _ = data
    batchSize = real_cpu.size(0)
    input.data.resize_(real_cpu.size()).copy_(real_cpu)

    if opt.model == 'BEGAN':
        noise.data.resize_(batchSize, nz, 1, 1)
        if opt.binary:
            bernoulli_prob.resize_(noise.data.size())
            noise.data.copy_(2*(torch.bernoulli(bernoulli_prob)-0.5))
        else:
            noise.data.normal_(0, 1)

        fake, _ = netG(noise)
        fake_recon = netD(fake.detach())
        real_recon = netD(input)

        err_real = torch.mean(torch.abs(real_recon - input))
        err_fake = torch.mean(torch.abs(fake_recon - fake))

        errD = err_real - k*err_fake
        errD.backward()
        optimizerD.step()

        netG.zero_grad()
        fake, _ = netG(noise)
        fake_recons = netD(fake)
        errG = torch.mean(torch.abs(fake_recons-fake))
        errG.backward()
        optimizerG.step()

        balance = (gamma * err_real - err_fake).data[0]
        k = min(max(k + lambda_k * balance,0),1)
        measure = err_real.data[0] + np.abs(balance)

        end_iter = time.time()
        ########### Logging #########
        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f Measure: %.4f K: %.4f LR: %.8f'
                  % (iteration, opt.niter, 
                     errD.data[0], errG.data[0], measure, k, optimizerD.param_groups[0]['lr']))

    else:
        label.data.fill_(real_label - opt.d_labelSmooth) # use smooth label for discriminator
        real_embed_ = real_embed.expand(input.size(0), 512,4,4).detach()
        fake_embed_ = fake_embed.expand(input.size(0), 512,4,4).detach()
        #fake_embed = fake_embed.detach()

        if opt.white_noise:
            additive_noise.data.resize_(input.size()).normal_(0, 0.005)
            input.data.add_(additive_noise.data)

        output, emb_real = netD(input)
        center_loss_real = torch.sum((real_embed_ - emb_real)**2)

        # Prevent numerical instability
        output = lowerbound(output)
        errD_real = criterion(output, label)
        errD_real += lambda_c / 2 * center_loss_real
        errD_real.backward()
        D_x = output.data.mean()
        # train with fake
        noise.data.resize_(batchSize, nz, 1, 1)
        if opt.binary:
            bernoulli_prob.resize_(noise.data.size())
            noise.data.copy_(2*(torch.bernoulli(bernoulli_prob)-0.5))
        else:
            noise.data.normal_(0, 1)
        fake_o,z_prediction = netG(noise)
        label.data.fill_(fake_label)

        if opt.white_noise:
            additive_noise.data.normal_(0, 0.005)
            fake = fake_o + additive_noise
            #fake.data.add_(additive_noise.data)
        else:
            fake = fake_o

        output, emb_fake = netD(fake.detach()) # add ".detach()" to avoid backprop through G
        center_loss_fake = torch.sum((fake_embed_ - emb_fake)**2)
        output = lowerbound(output)
        errD_fake = criterion(output, label)
        errD_fake += lambda_c / 2 * center_loss_fake
        errD_fake.backward() # gradients for fake/real will be accumulated
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake

        # Gradient penalty for DRAGAN
        if opt.model == 'DRAGAN': #or opt.model == 'RESNET':
            gradient_loss = calc_gradient_penalty_DRAGAN(netD, input)
            gradient_loss.backward()
            errD += gradient_loss

        optimizerD.step() # .step() can be called once the gradients are computed
        real_embed = real_embed - alpha * torch.sum((real_embed_-emb_real)/opt.batchSize, 0)
        fake_embed = fake_embed - alpha * torch.sum((fake_embed_-emb_fake)/opt.batchSize, 0)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        fake_o,z_prediction = netG(noise)
        label.data.fill_(real_label) # fake labels are real for generator cost
        output, emb_real = netD(fake_o)
        center_loss = torch.sum((real_embed_.detach() - emb_real)**2)
        output = lowerbound(output)
        errG = criterion(output, label)
        errG += lambda_c/2 * center_loss
        errG.backward(retain_variables=True) # True if backward through the graph for the second time
        if opt.model == 'IGAN': # with z predictor
            errG_z = criterion_MSE(z_prediction, noise)
            errG_z.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()
    
        end_iter = time.time()

        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s, %.4f real:, %.4f fake'
              % (iteration, opt.niter,
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, end_iter-start_iter, center_loss_real.data[0], center_loss_fake.data[0]))

    ########### Learning Rate Decay #########
    optimizerD = adjust_learning_rate(optimizerD,iteration)
    optimizerG = adjust_learning_rate(optimizerG,iteration)


    if iteration % 500 == 0:
        # the first 64 samples from the mini-batch are saved.
        #vutils.save_image(real_cpu[0:64,:,:,:],
        #        '%s/real_samples_%03d_%04d.png' % (opt.imDir, epoch, i), nrow=8)
        fake,_ = netG(noise)
        vutils.save_image(fake.data[0:64,:,:,:],
                '%s/fake_samples_epoch_%03d.png' % (opt.imDir, iteration), nrow=8)
    if iteration % opt.save_step == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.modelsDir, iteration))
