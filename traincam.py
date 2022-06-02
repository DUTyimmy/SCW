import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f

from datetime import datetime
from utils.pamr import BinaryPamr
from tensorboardX import SummaryWriter
# Enter [tensorboard --logdir=log] in [Terminal in Pycharm IDE] or [terminal in Ubuntu OS] or [Win+r-cmd in Windows OS]
# to use tensorboard to visualize the training process. You can also use [import torch.utils.tensorboard].

writer = SummaryWriter('log/log_cam')
running_loss = 0


class TrainCam(object):

    def __init__(self, model, optimizer_model, train_loader, max_epoch, outpath):
        self.model = model
        self.optim_model = optimizer_model
        self.train_loader = train_loader
        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.outpath = outpath
        self.sshow = 20
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss()

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

    def train_epoch(self):

        for idx, pack in enumerate(self.train_loader):
            total_iter = idx + self.epoch * len(self.train_loader)

            img = pack[0].cuda()
            label = pack[1].cuda(non_blocking=True)

            cls, cam = self.model(img)
            loss = f.multilabel_soft_margin_loss(cls, label)

            self.optim_model.zero_grad()
            loss.backward()
            self.optim_model.step()

            global running_loss
            running_loss += loss.item()

            # ------------------------------ model's output supervised --------------------------- #

            if idx % self.sshow == (self.sshow - 1):
                print('[%s,  Epoch: %2d / %2d,  Iter: %3d / %3d,  loss: %.4f]' % (
                    datetime.now().replace(microsecond=0), self.epoch + 1, self.max_epoch, idx + 1,
                    len(self.train_loader), running_loss / self.sshow))

                writer.add_scalar('loss of 1st training stage', running_loss / self.sshow, total_iter + 1)

                # visualization of tensorboard
                cam, cls = cam[0], self.sigmoid(cls)[0].clone().view(label.shape[1], 1, 1)
                cam = cam.cpu().detach().numpy() * cls.cpu().detach().numpy()  # 20*256*256
                cam = torch.from_numpy(np.sum(cam, 0)).cuda().unsqueeze(0).unsqueeze(0)
                cam /= f.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
                out_refine = BinaryPamr(img[0].unsqueeze(0), cam[0].unsqueeze(0), binary=None)
                img_rgb = img.clone()
                img_show = self.rgb2grey(img_rgb[0].unsqueeze(0))
                img_show = f.interpolate(img_show, size=256, mode="bilinear", align_corners=False)
                image = torch.cat((img_show, cam[0].unsqueeze(0), out_refine[0].unsqueeze(0)), 0)
                writer.add_images('cam maps', image, total_iter + 1, dataformats='NCHW')

                running_loss = 0.0

            # for ImageNet
            if idx % 1000 == 999:
                savename = ('%s/cam_iter_%d.pth' % (self.outpath, idx + 1))
                torch.save(self.model.state_dict(), savename)

    def train(self):
        max_epoch = self.max_epoch

        for epoch in range(max_epoch):
            self.epoch = epoch
            self.train_epoch()
            print('')

            if self.epoch >= self.max_epoch * len(self.train_loader):
                break
