from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
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
from torch import autograd
from torch.autograd import Variable, grad
import numpy as np

### load project files
import models
import srresnet
from models import weights_init

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', default='./celeb', help='path to dataset')
parser.add_argument('--workers', type=int, default=12, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=300000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , type=int, default=1, help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netENC', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outDir', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--model', required=True, help='DCGAN | RESNET | IGAN | DRAGAN | BEGAN')
parser.add_argument('--d_labelSmooth', type=float, default=0.1, help='for D, use soft label "1-labelSmooth" for real samples')
parser.add_argument('--n_extra_layers_d', type=int, default=0, help='number of extra conv layers in D')
parser.add_argument('--n_extra_layers_g', type=int, default=1, help='number of extra conv layers in G')
parser.add_argument('--pix_shuf'  , type=int, default=1, help='Use pixel shuffle instead of deconvolution')
parser.add_argument('--white_noise'  , type=int, default=1, help='Add white noise to inputs of discriminator to stabilize training')
parser.add_argument('--lambda_'  , type=int, default=10, help='Weight of gradient penalty (DRAGAN)')
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

def adjust_learning_rate(optimizer, niter):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.95 ** (niter // opt.lr_decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def scale_learning_rate(optimizer, niter, target):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if niter > target:
        return adjust_learning_rate(optimizer, niter)
    else:
        lr = niter / target * opt.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, opt.imageSize, opt.imageSize)
    alpha = alpha.cuda()

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)


    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda_
    return gradient_penalty

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
    output = torch.max(input, epsilon)
    return output

def vae_loss(mu, logvar, pred, gt, batch_size): 
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss /= (batch_size * opt.nz)
    #kl_element = torch.add(torch.add(torch.add(mu.pow(2), logvar.exp()), -1), logvar.mul(-1))
    #kl_loss = torch.sum(kl_element).mul(.5)
    #recon_loss = torch.sum(torch.abs(pred - gt))
    recon_loss = torch.sum((pred-gt).pow(2))
    return kl_loss+recon_loss*1e-4

def test():
    z = Variable(torch.randn(64, opt.nz, 1, 1)).cuda()
    output, _ = netG(z)
    vutils.save_image(output.data[0:64,:,:,:],
            '%s/sample.png' % (opt.imDir), nrow=8)

    
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
            transforms.Resize(size=[opt.imageSize, opt.imageSize]),
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
    netENC = models.resVAE(opt.nz)
elif opt.model == 'RESNET':
    netG = srresnet.NetG(opt.imageSize)
    netD = srresnet.NetD(norm, opt.imageSize)

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

netENC.apply(weights_init)
if opt.netENC != '':
    netENC.load_state_dict(torch.load(opt.netENC))
print(netD)

print(netENC)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
additive_noise = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor([1])
real_label = 1
fake_label = 0
epsilon = torch.FloatTensor([1e-8])
#BEGAN parameters
gamma = .7
lambda_k = 0.001
k = 0.
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netENC.cuda()
    criterion.cuda()
    input, label, additive_noise = input.cuda(), label.cuda(), additive_noise.cuda()
    noise, epsilon = noise.cuda(), epsilon.cuda()
    one, mone = one.cuda(), mone.cuda()
    
input = Variable(input)
label = Variable(label)
additive_noise = Variable(additive_noise)
epsilon = Variable(epsilon)
noise = Variable(noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerENC = optim.Adam(netENC.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

def trainVAE(input, batchSize):
    netENC.zero_grad()
    netG.zero_grad()

    mu, logvar = netENC(input)
    # calculate latent variable
    stddev = torch.sqrt(torch.exp(logvar))
    eps = Variable(torch.randn(stddev.size()).normal_()).cuda()
    latent = torch.add(mu, torch.mul(eps, stddev))
    latent = latent.view(-1, nz, 1, 1)
    recon, _ = netG(latent)
    loss = 1e-1 * vae_loss(mu, logvar, input, recon, batchSize)
    loss.backward()
    return loss, recon

for iteration in range(1, opt.niter+1):
    start_iter = time.time()
    ########### Learning Rate Decay #########
    optimizerD = adjust_learning_rate(optimizerD,iteration)
    optimizerG = scale_learning_rate(optimizerG,iteration,target=100)
    #optimizerG = adjust_learning_rate(optimizerG,iteration)
    try: 
        data = loader.next()
    except StopIteration:
        loader = iter(dataloader)
        data = loader.next()

    real_cpu, _ = data
    batchSize = real_cpu.size(0)
    input.data.resize_(real_cpu.size()).copy_(real_cpu)

    loss, recon = trainVAE(input, batchSize)

    # train with real
    netD.zero_grad()
    noise.data.resize_(batchSize, nz, 1, 1)
    noise.data.normal_(0, 1)
    label.data.fill_(real_label)

    if opt.white_noise:
        additive_noise.data.resize_(input.size()).normal_(0, 0.005)
        input.data.add_(additive_noise.data)

    D_real = netD(input)
    D_real_loss = criterion(lowerbound(D_real), label)
    D_real_loss.backward()

    # train with fake
    fake, z_prediction = netG(noise)
    label.data.fill_(fake_label)

    if opt.white_noise:
        additive_noise.data.normal_(0, 0.005)
        fake = fake + additive_noise
        #fake.data.add_(additive_noise.data)

    D_fake = netD(fake.detach()) # add ".detach()" to avoid backprop through G
    D_fake_loss = criterion(lowerbound(D_fake), label)
    D_fake_loss.backward()
    D_loss = D_real_loss + D_fake_loss

    gradient_loss = calc_gradient_penalty_DRAGAN(netD, input)
    gradient_loss.backward()
    D_loss += gradient_loss
    optimizerD.step() # .step() can be called once the gradients are computed

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.data.fill_(real_label)
    noise.data.normal_(0, 1)
    D_fake = netD(fake)
    G_loss = criterion(lowerbound(D_fake), label)
    G_loss.backward()
    optimizerG.step()
    optimizerENC.step()

    end_iter = time.time()

    #print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f VAE: %.4f Elapsed %.2f s'
    #    % (iteration, opt.niter,
    #    errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, loss.data[0], end_iter-start_iter))
    print(iteration, D_loss.data[0], G_loss.data[0], loss.data[0])


    if iteration % 500 == 0:
        # the first 64 samples from the mini-batch are saved.
        vutils.save_image(real_cpu[0:64,:,:,:],
                '%s/real_samples_%03d.png' % (opt.imDir, iteration), nrow=8)
        vutils.save_image(recon.data[0:64,:,:,:],
                '%s/recon_samples_epoch_%03d.png' % (opt.imDir, iteration), nrow=8)
        fake,_ = netG(noise)
        vutils.save_image(fake.data[0:64,:,:,:],
                '%s/fake_samples_epoch_%03d.png' % (opt.imDir, iteration), nrow=8)
    if iteration % opt.save_step == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_2_epoch_%d.pth' % (opt.modelsDir, iteration))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.modelsDir, iteration))
        torch.save(netENC.state_dict(), '%s/netENC_epoch_%d.pth' % (opt.modelsDir, iteration))
