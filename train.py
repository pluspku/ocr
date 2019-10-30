from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from networks import define_G, define_D, GANLoss
from data import get_training_set, get_test_set
from shared import Meter, checksum, train_mode
import torch.backends.cudnn as cudnn
import datetime

from logger import Logger

# Training settings
import settings as opt
print('===> Checksum = %s' % opt.date)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set()
test_set = get_test_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

train_logger = Logger(opt.nEpochs, len(training_data_loader), opt.date)
test_logger = Logger(opt.nEpochs, len(testing_data_loader), opt.date)

print('===> Building model')
netG_A2B = define_G(opt.input_nc, opt.output_nc, opt.ngf, norm = opt.batch_mode, netG = opt.netG, use_dropout = False, gpu_ids = [0])
netD_B = define_D(opt.input_nc + opt.output_nc, opt.ndf, norm = opt.batch_mode, netD = opt.netD, gpu_ids =[0])
netG_B2A = define_G(opt.input_nc, opt.output_nc, opt.ngf, norm = opt.batch_mode, netG = opt.netG, use_dropout = True, gpu_ids =[0])
netD_A = define_D(opt.input_nc + opt.output_nc, opt.ndf, norm = opt.batch_mode, netD = opt.netD, gpu_ids =[0])

criterionGAN = GANLoss(opt.GANMode)
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()
criterion_identity = nn.L1Loss()
criterion_cycle = nn.L1Loss()

# setup optimizer
import itertools
optimizer_G = optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.glr, betas=(opt.beta1, 0.999))
optimizerD_B = optim.Adam(netD_B.parameters(), lr=opt.dlr, betas=(opt.beta1, 0.999))
optimizerD_A = optim.Adam(netD_A.parameters(), lr=opt.dlr, betas=(opt.beta1, 0.999))

print('---------- Networks initialized -------------')
#print_network(netG)
#print_network(netD_B)
print('-----------------------------------------------')

real_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 256, 256)
real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)

if opt.cuda:
    netD_B = netD_B.cuda()
    netG_A2B = netG_A2B.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()

real_a = Variable(real_a)
real_b = Variable(real_b)


def train(epoch):
    print("===> Train")
    train_set.reset()
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        fake_b = netG_A2B(real_a)
        fake_a = netG_B2A(real_b)

        fake_ab = torch.cat((real_a, fake_b), 1)
        fake_ba = torch.cat((real_b, fake_a), 1)
        real_ab = torch.cat((real_a, real_b), 1)
        real_ba = torch.cat((real_b, real_a), 1)

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################

        def step_D(netD, optimizerD, fake, real):
            optimizerD.zero_grad()
            # train with fake
            pred_fake = netD.forward(fake.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            # train with real
            pred_real = netD.forward(real.detach())
            loss_d_real = criterionGAN(pred_real, True)

            # Combined loss
            loss_d = loss_d_fake + loss_d_real

            loss_d.backward()
            optimizerD.step()

            return loss_d

        loss_d_b = step_D(netD_B, optimizerD_B, fake_ab, real_ab)
        loss_d_a = step_D(netD_A, optimizerD_A, fake_ba, real_ba)

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizer_G.zero_grad()
        # First, G(A) should fake the discriminator
        pred_fake = netD_B.forward(fake_ab)
        loss_g_gan_ab = criterionGAN(pred_fake, True)
        pred_fake = netD_A.forward(fake_ba)
        loss_g_gan_ba = criterionGAN(pred_fake, True)

        loss_g_gan = loss_g_gan_ab + loss_g_gan_ba

         # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) #+ criterionL1(fake_a, real_a)

        # Third, E(G(A)) = E(A)
        loss_g_density = criterionMSE(fake_b.mean((1,2,3)), real_a.mean((1,2,3)))

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_b = netG_A2B(real_b)
        loss_identity_b = criterion_identity(same_b, real_b)
        # G_B2A(A) should equal A if real A is fed
        same_a = netG_B2A(real_a)
        loss_identity_a = criterion_identity(same_a, real_a)
        loss_identity = loss_identity_a + loss_identity_b

        # Cycle loss
        recovered_a = netG_B2A(fake_b)
        loss_cycle_aba = criterion_cycle(recovered_a, real_a)

        recovered_b = netG_A2B(fake_a)
        loss_cycle_bab = criterion_cycle(recovered_b, real_b)
        
        loss_cycle = loss_cycle_aba + loss_cycle_bab
        
        loss_g = loss_g_gan + loss_g_l1 * opt.lamb + loss_identity * 0.5 + loss_cycle * 0.5

        loss_g.backward()
        optimizer_G.step()

        #print("===> Epoch[{}]({}/{}): Loss_D: {} Loss_G: {}\r".format(
        #    epoch, iteration, len(training_data_loader), meter_D, meter_G), end = '')
        train_logger.log(losses = {
            'D_A': loss_d_a,
            'D_B': loss_d_b,
            'G': loss_g,
            'G_GAN': loss_g_gan,
            'G_L1': loss_g_l1,
            'G_cycle': loss_cycle,
            'G_identity': loss_identity,
            }, images = {'real_a': real_a, 'fake_b': fake_b, 'real_b': real_b, 'fake_a': fake_a})

def test():
    print("\n===> Test")
    test_set.reset()
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = Variable(batch[0]), Variable(batch[1])
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()

            prediction = netG_A2B(input)
            mse = criterionMSE(prediction, target)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr.item()
            test_logger.log(images = {'test_real_a': input, 'test_real_b': target, 'test_fake_b': prediction}, losses = {'psnr': psnr})
    psnr = avg_psnr / len(testing_data_loader)
    print("\n===> Avg. PSNR: {:.4f} dB".format(psnr))


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.date)):
        os.mkdir(os.path.join("checkpoint", opt.date))
    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.date, epoch)
    net_d_model_out_path = "checkpoint/{}/netD_B_model_epoch_{}.pth".format(opt.date, epoch)
    torch.save(netG_A2B, net_g_model_out_path)
    torch.save(netD_B, net_d_model_out_path)
    print("Checkpoint saved to checkpoint/{}".format(opt.date))

def merge_stats(*stats):
    ret = {'losses': {}, 'images': {}}
    for stat in stats:
        for item in ['losses', 'images']:
            if item in stat:
                ret[item].update(stat[item])
    return ret

if __name__ == '__main__':
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        test()
        if epoch % 20 == 0 or (epoch < 30 and epoch % 2 == 0):
            checkpoint(epoch)
