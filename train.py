from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from networks import define_G, define_D, GANLoss, print_network
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
netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.batch_mode, False, [0])
netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.batch_mode, False, [0])

criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.glr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.dlr, betas=(opt.beta1, 0.999))

print('---------- Networks initialized -------------')
#print_network(netG)
#print_network(netD)
print('-----------------------------------------------')

real_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 256, 256)
real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)

if opt.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()

real_a = Variable(real_a)
real_b = Variable(real_b)


def train(epoch):
    print("===> Train")
    meter_D = Meter(1000)
    meter_G = Meter(1000)
    train_set.reset()
    for iteration, batch in enumerate(training_data_loader, 1):
        mode = train_mode()
        # forward
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        fake_b = netG(real_a)

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################

        optimizerD.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = netD.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)

        # train with random wrong word
        from dataset import random_word
        others = random_word(opt.batchSize)
        if opt.cuda:
            others = others.cuda()
        other_ab = torch.cat((real_a, others), 1)
        pred_other = netD.forward(other_ab)
        loss_d_other = criterionGAN(pred_other, False)
        
        # Combined loss
        loss_d = (loss_d_fake + loss_d_real + loss_d_other * opt.other_loss_rate)

        if mode in ("A", "D"):
            loss_d.backward()
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

         # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        
        loss_g = loss_g_gan + loss_g_l1

        if mode in ("A", "G"):
            loss_g.backward()
            optimizerG.step()

        meter_D.update(loss_d.item())
        meter_G.update(loss_g.item())

        #print("===> Epoch[{}]({}/{}): Loss_D: {} Loss_G: {}\r".format(
        #    epoch, iteration, len(training_data_loader), meter_D, meter_G), end = '')
        train_logger.log(losses = {'D': loss_d, 'G': loss_g}, images = {'real_a': real_a, 'fake_b': fake_b, 'real_b': real_b})

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

            prediction = netG(input)
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
    net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.date, epoch)
    torch.save(netG, net_g_model_out_path)
    torch.save(netD, net_d_model_out_path)
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
        if epoch % 20 == 0:
            checkpoint(epoch)
