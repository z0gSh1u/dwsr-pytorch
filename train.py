'''
    Training code for DWSR.
'''

from torch.utils.tensorboard import SummaryWriter
from util import *
from dataset import DWSRDataset
from model import DWSR
from torch.utils.data import *
from dataset import *
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import lpips
from tqdm import tqdm
from option import opt
import os

torch.backends.cudnn.benchmark = True
tb = SummaryWriter('./out/tb')

if __name__ == '__main__':
    # initialize the model
    model = DWSR(opt['n_conv'], opt['residue_weight']).cuda()

    # using MSELoss
    criterion = nn.MSELoss().cuda()
    # using Adam optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=opt['lr'],  weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-08
    )
    # learning rate decay by 0.75x every 20 epochs
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(20, opt['epochs'], 20)), gamma=0.75)

    # load the datasets
    train_set = DWSRDataset(opt['train_dir'], opt['crop_size'], 'train')
    train_loader = DataLoader(
        train_set, batch_size=opt['batch_size'], shuffle=True, num_workers=opt['num_workers'])
    val_set = DWSRDataset(opt['val_dir'], opt['crop_size'], 'val')
    val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, num_workers=opt['num_workers'], pin_memory=True)

    # prepare LPIPS calculation
    lpips_loss = lpips.LPIPS(net='vgg')
    lpips_loss.eval()

    # start training
    for epoch in range(opt['epochs']):
        model.train()
        epoch_loss = AverageMeter()

        with tqdm(total=(len(train_set) - len(train_set) % opt['batch_size']), ncols=80) as t:
            t.set_description('Epoch: {}/{}'.format(epoch + 1, opt['epochs']))

            for data in train_loader:
                lr, hr = data
                lr = lr.cuda()
                hr = hr.cuda()

                # go through the network
                predict = model(lr).cuda()
                loss = criterion(predict, hr)
                # torch.mean flats the loss output to one scalar
                epoch_loss.update(torch.mean(loss), len(lr))

                # optimize it
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_loss.avg))
                t.update(len(lr))

            # decay learning rate
            scheduler.step(epoch)
            print('current lr: {}'.format(
                optimizer.state_dict()['param_groups'][0]['lr']))

            # save ckp
            torch.save(model.state_dict(), os.path.join(
                './out/ckp', 'epoch_{}.pth'.format(epoch)))

            # validation
            print('\nCalculating metrics on validation set...')
            model.eval()

            epoch_psnr = AverageMeter()
            epoch_lpips = AverageMeter()

            for data in tqdm(val_loader, ncols=80, desc='Validation'):
                lr, hr = data
                lr = lr.cuda()
                hr = hr.cuda()

                with torch.no_grad():
                    predict = model(lr).detach().clamp(0.0, 1.0)

                # calculate PSNR
                epoch_psnr.update(calc_psnr(predict.cpu(), hr.cpu()), len(lr))

                # calculate LPIPS
                # this package requires input range to be [-1, 1], see https://github.com/richzhang/PerceptualSimilarity
                predict = (predict - 0.5) * 2
                hr = (hr - 0.5) * 2
                epoch_lpips.update(lpips_loss(
                    predict.cpu(), hr.cpu()).detach().cpu().numpy().flatten()[0])

            # print PSNR and LPIPS to terminal
            print(
                'Val PSNR: {:.2f}  /  Val LPIPS: {:.2f}'.format(epoch_psnr.avg, epoch_lpips.avg))
            # save PSNR and LPIPS to TensorBoard
            tb.add_scalar('Val PSNR', epoch_psnr.avg, global_step=epoch)
            tb.add_scalar('Val LPIPS', epoch_lpips.avg, global_step=epoch)
            # save PSNR and LPIPS to local file
            save_metric('./out/metric.txt', 'Epoch: {}  /  PSNR: {}  /  LPIPS: {}'.format(
                epoch, epoch_psnr.avg, epoch_lpips.avg))
