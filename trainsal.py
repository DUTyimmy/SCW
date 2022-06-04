import os
import time
import torch
import numpy as np
import torch.nn as nn

from datetime import datetime
from utils.imsave import imsave
from utils.pamr import BinaryPamr
from utils.datainit import valdatainit
from utils.evaluateFM import get_FM
from tensorboardX import SummaryWriter
# Enter [tensorboard --logdir=log] in [Terminal in Pycharm IDE] or [terminal in Ubuntu OS] or [Win+r-cmd in Windows OS]
# to use tensorboard to visualize the training process. You can also use [import torch.utils.tensorboard].

writer = SummaryWriter('log/log_sal')
loss_total = 0


class TrainSal(object):

    def __init__(self, model, optimizer_model, train_loader, val_loader, outpath, max_epoch=20, stage=2, maximum=0.6):
        self.model = model
        self.optim_model = optimizer_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epoch = int(max_epoch)
        self.stage = int(stage)
        self.outpath = outpath
        self.m = maximum

        self.BCEloss = nn.BCELoss()
        self.sshow = 20
        self.iteration = 0
        self.epoch = 0

    @staticmethod
    def rgb2grey(images):
        mean = np.array([0.447, 0.407, 0.386])
        std = np.array([0.244, 0.250, 0.253])

        images1, images2, images3 = images[:, 0:1, :, :], images[:, 1:2, :, :], images[:, 2:3, :, :]
        images1 = images1 * std[0] + mean[0]
        images2 = images2 * std[1] + mean[1]
        images3 = images3 * std[2] + mean[2]
        img_grey = images1 * 0.299 + images2 * 0.587 + images3 * 0.114
        return img_grey

    def criterion(self, img, sal, lbl, delta):

        lbl_self = sal.clone().detach()
        lbl_self = BinaryPamr(img, lbl_self, binary=0.4)
        loss1 = self.BCEloss(sal, lbl.unsqueeze(1)) * (1-delta)
        loss2 = self.BCEloss(sal, lbl_self) * delta
        return loss1, loss2

        # bs-loss
        # lbl_self = sal.clone().detach()
        # lbl_self[lbl_self > 0.5] = 1
        # lbl_self[lbl_self <= 0.5] = 0
        # loss1 = self.BCEloss(sal, lbl.unsqueeze(1)) * (1 - delta)
        # loss2 = self.BCEloss(sal, lbl_self) * delta
        # return loss1, loss2

    def train_epoch(self):
        for batch_idx, (img, lbl) in enumerate(self.train_loader):

            delta = (self.epoch / (self.max_epoch + 1e-5))**0.5
            if delta > self.m:
                delta = self.m

            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration
            if self.epoch >= self.max_epoch*len(self.train_loader):
                break
            img, lbl = img.cuda(), lbl.cuda()
            side5_1, side4_1, sal = self.model(img)

            # proposed self-calibrated loss
            loss1_3, loss2_3 = self.criterion(img, sal, lbl, delta)
            loss1_4, loss2_4 = self.criterion(img, side4_1, lbl, delta)
            loss1_5, loss2_5 = self.criterion(img, side5_1, lbl, delta)
            loss1 = (loss1_3 + loss1_4 + loss1_5)/3
            loss2 = (loss2_3 + loss2_4 + loss2_5)/3
            loss = loss1 + loss2

            # # BCE-loss
            # loss_3 = self.BCEloss(sal, lbl)
            # loss_4 = self.BCEloss(side4_1, lbl)
            # loss_5 = self.BCEloss(side5_1, lbl)
            # loss = (loss_3 + loss_4 + loss_5)/3

            self.optim_model.zero_grad()
            loss.backward()
            self.optim_model.step()

            global loss_total
            loss_total += loss.item()

            # ------------------------------ information exhibition and visualization --------------------------- #
            if iteration % self.sshow == (self.sshow - 1):

                print('[ %s,  Stage: %1d,  '
                      'Epoch: %2d / %2d,  '
                      'Iter: %3d / %3d,  '
                      'Loss: %.4f,  '
                      'Rate: %.3f ]' %
                      (datetime.now().replace(microsecond=0),
                       self.stage,
                       self.epoch+1,
                       self.max_epoch,
                       batch_idx+1,
                       len(self.train_loader),
                       loss_total/self.sshow,
                       delta))

                # tensorboard
                writer.add_scalar('loss of 2nd training stage', loss_total / self.sshow, iteration + 1)
                sal_pamr = BinaryPamr(img, sal, binary=0.4)
                img_show = img.clone()
                img_show = self.rgb2grey(img_show[0].unsqueeze(0))
                image = torch.cat((img_show[0], lbl[0].unsqueeze(0), sal[0], sal_pamr[0]), 0)
                image = torch.unsqueeze(image, 0).transpose(0, 1)
                writer.add_images('sal maps', image, iteration+1, dataformats='NCHW')

                loss_total = 0.0

    def train(self):
        best_mae, best_f = 0.0, 0.0

        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.model.train()
            self.train_epoch()
            print('')

            if self.val_loader is not None:
                # ---------------- validation ---------------- #
                print('\nValidating .....   \n[  ', end='')
                valdatainit()
                self.model.eval()
                total_num = len(self.val_loader)
                count_num = int(total_num / 10)

                start_time = time.time()
                for idx, (data, name, size) in enumerate(self.val_loader):
                    _, _, sal = self.model(data.cuda())

                    if self.stage == 1:
                        sal = BinaryPamr(data.cuda(), sal, binary=0.4)

                    sal = sal.squeeze().cpu().detach()
                    imsave(os.path.join('data/val_map', name[0] + '.png'), sal, size)

                    if idx % count_num == count_num - 1:
                        print((str(round(int(idx + 1) / total_num * 100))) + '.0 %  ', end='')

                print('],  finished,  ', end='')
                final_time = time.time()
                print('cost %d seconds. ' % (final_time - start_time))

                # ---------------- evaluation ---------------- #
                print("\nEvaluating .....")
                f, mae = get_FM(salpath='data/val_map/', gtpath='data/ECSSD/mask/')
                if f > best_f:
                    best_mae, best_f = mae, f
                    savename = ('%s/sal_stage_%d.pth' % (self.outpath, self.stage))
                    torch.save(self.model.state_dict(), savename)

                print('this F_measure:% .4f' % f, end='\t\t')
                print('this MAE:% .4f' % mae)
                print('best F_measure:% .4f' % best_f, end='\t\t')
                print('best MAE:% .4f' % best_mae, end='\n\n')

            # savename = ('%s/sal_stage_%d_epoch_%d.pth' % (self.outpath, self.stage, self.epoch+1))
            # torch.save(self.model.state_dict(), savename)
            savename = ('%s/sal_stage_%d.pth' % (self.outpath, self.stage))
            torch.save(self.model.state_dict(), savename)

            if self.epoch >= self.max_epoch*len(self.train_loader):
                break
