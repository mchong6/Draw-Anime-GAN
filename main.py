from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
parser.add_argument('--dataRoot', default='./data', help='path to dataset')
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
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netD_patch', default='', help="path to netD (to continue training)")
parser.add_argument('--outDir', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--model', required=True, help='DCGAN | RESNET | IGAN | DRAGAN | BEGAN')
parser.add_argument('--d_labelSmooth', type=float, default=0.1, help='for D, use soft label "1-labelSmooth" for real samples')
parser.add_argument('--n_extra_layers_d', type=int, default=0, help='number of extra conv layers in D')
parser.add_argument('--n_extra_layers_g', type=int, default=1, help='number of extra conv layers in G')
parser.add_argument('--pix_shuf'  , type=int, default=1, help='Use pixel shuffle instead of deconvolution')
parser.add_argument('--only_patch'  , type=int, default=0, help='Use pixel shuffle instead of deconvolution')
parser.add_argument('--white_noise'  , type=int, default=0, help='Add white noise to inputs of discriminator to stabilize training')
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
    norm = 'BatchNorm'
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
    return torch.max(input, epsilon)

    
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
    netD_patch = models.patchD(ngpu, nz, nc, ndf, n_extra_d, norm, opt.imageSize)
elif opt.model == 'RESNET':
    netG = srresnet.NetG(opt.imageSize)
    netD = srresnet.NetD(norm, opt.imageSize)

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

if not opt.only_patch:
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

netD_patch.apply(weights_init)
if opt.netD_patch != '':
    netD_patch.load_state_dict(torch.load(opt.netD_patch))
print(netD_patch)
criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
additive_noise = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor([1])
real_label = 1
fake_label = 0
epsilon = torch.FloatTensor([1e-9])
#BEGAN parameters
gamma = .7
lambda_k = 0.001
k = 0.

if opt.cuda:
    if not opt.only_patch:
        netD.cuda()
    netD_patch.cuda()
    netG.cuda()
    criterion.cuda()
    input, label, additive_noise = input.cuda(), label.cuda(), additive_noise.cuda()
    noise, epsilon = noise.cuda(), epsilon.cuda()
    
input = Variable(input)
label = Variable(label)
additive_noise = Variable(additive_noise)
epsilon = Variable(epsilon)
noise = Variable(noise)

# setup optimizer
if not opt.only_patch:
    optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerD_patch = optim.Adam(netD_patch.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

for iteration in range(1, opt.niter+1):
    try: 
        data = loader.next()
    except StopIteration:
        loader = iter(dataloader)
        data = loader.next()

    start_iter = time.time()
    real_cpu, _ = data
    batchSize = real_cpu.size(0)
    input.data.resize_(real_cpu.size()).copy_(real_cpu)
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    if not opt.only_patch:
        netD.zero_grad()
    netD_patch.zero_grad()
    noise.data.resize_(batchSize, nz, 1, 1)
    noise.data.normal_(0, 1)

    label.data.fill_(real_label - opt.d_labelSmooth) # use smooth label for discriminator

    if opt.white_noise:
        additive_noise.data.resize_(input.size()).normal_(0, 0.005)
        input.data.add_(additive_noise.data)

    if not opt.only_patch:
        output = netD(input)
        errD_real = criterion(lowerbound(output), label)
        errD_real.backward()

    patch_local, patch_glob = netD_patch(input)
    err_patch_local_r = criterion(lowerbound(patch_local), label)
    err_patch_glob_r = criterion(lowerbound(patch_glob), label)
    err = err_patch_local_r + err_patch_glob_r
    err.backward()
    D_x = err_patch_local_r.data.mean() + err_patch_glob_r.data.mean() if opt.only_patch else output.data.mean()

    # train with fake
    fake, z_prediction = netG(noise)
    label.data.fill_(fake_label)

    if opt.white_noise:
        additive_noise.data.normal_(0, 0.005)
        fake_n = fake + additive_noise
    else:
        fake_n = fake

    if not opt.only_patch:
        output = netD(fake_n.detach()) # add ".detach()" to avoid backprop through G
        errD_fake = criterion(lowerbound(output), label)
        errD_fake.backward() 

    patch_local, patch_glob = netD_patch(fake_n.detach())
    err_patch_local_f = criterion(lowerbound(patch_local), label)
    err_patch_glob_f = criterion(lowerbound(patch_glob), label)
    err = err_patch_local_f + err_patch_glob_f
    err.backward()
    D_G_z1 = err_patch_local_f.data.mean() + err_patch_glob_f.data.mean() if opt.only_patch else output.data.mean()
    errD = err_patch_local_f + err_patch_local_r + err_patch_glob_f + err_patch_local_f

    #grad_penalty =  calc_gradient_penalty_DRAGAN(netD, input)
    #grad_penalty.backward()
    #grad_penalty_patch =  calc_gradient_penalty_DRAGAN(netD_patch, input)
    if not opt.only_patch:
        optimizerD.step()
    optimizerD_patch.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.data.fill_(real_label) # fake labels are real for generator cost
    if not opt.only_patch:
        output = netD(fake)
        errG = criterion(lowerbound(output), label)
        errG.backward(retain_graph=True) 

    patch_local, patch_glob = netD_patch(fake)
    err_patch_local = criterion(lowerbound(patch_local), label)
    err_patch_glob = criterion(lowerbound(patch_glob), label)
    err = err_patch_local + err_patch_glob
    err.backward()
    D_G_z2 = err_patch_local.data.mean() + err_patch_glob.data.mean() if opt.only_patch else output.data.mean()
    optimizerG.step()
    errG = err_patch_local + err_patch_glob

    end_iter = time.time()

    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
        % (iteration, opt.niter,
        errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, end_iter-start_iter))
    #print(iteration, loss.data[0])

    ########### Learning Rate Decay #########
    #optimizerD = adjust_learning_rate(optimizerD,iteration)
    #optimizerG = adjust_learning_rate(optimizerG,iteration)


    if iteration % 500 == 0:
        # the first 64 samples from the mini-batch are saved.
        #vutils.save_image(real_cpu[0:64,:,:,:],
        #        '%s/real_samples_%03d.png' % (opt.imDir, iteration), nrow=8)
        fake,_ = netG(noise)
        vutils.save_image(fake.data[0:64,:,:,:],
                '%s/fake_samples_epoch_%03d.png' % (opt.imDir, iteration), nrow=8, normalize=True)
    if iteration % opt.save_step == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.modelsDir, iteration))
        torch.save(netD_patch.state_dict(), '%s/netD_patch_epoch_%d.pth' % (opt.modelsDir, iteration))
        if not opt.only_patch:
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.modelsDir, iteration))
