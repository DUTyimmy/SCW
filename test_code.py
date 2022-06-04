"""
Project: To be Critical: Self-Calibrated Weakly Supervised Learning for Salient Object Detection
Author: Yimmy (Jian Wang)
College: The IIAU-OIP Lab, Dalian University of Technology
"""

import os
import time
import torch
import argparse

from utils.crf import crf
from utils.imsave import imsave
from utils.pamr import BinaryPamr
from utils.datainit import testdatainit
from model.sal_model import WsodDense
from dataset_loader import MySalTestData
from torch.utils.data import DataLoader


def main():
    # -------------------------------------------------- options --------------------------------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_root', type=str, default='/home/oip/2DSOD/ECSSD/image', help='path to test dataset')

    parser.add_argument('--ckpt_root', type=str, default='snapshot/sal_stage_2.pth', help='path to checkpoint')

    parser.add_argument('--pamr', type=bool, default=False, help='performing PAMR or not')
    parser.add_argument('--crf', type=bool, default=False, help='performing CRF or not')
    parser.add_argument('--salmap_root', type=str, default='sal_map', help='path to output saliency map')
    parser.add_argument('--salmap_crf_root', type=str, default='sal_map_crf', help='path to saliency map through crf')
    parser.add_argument('--num_workers', type=int, default=12, help='the CPU workers number')
    parser.add_argument('--batch', type=int, default=1, help='the CPU workers number')
    args = parser.parse_args()
    testdatainit(args.salmap_root, crf=args.crf)

    # ------------------------------------------------- dataloader -------------------------------------------------- #
    test_loader = torch.utils.data.DataLoader(MySalTestData(args.test_root, transform=True), batch_size=args.batch,
                                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # -------------------------------------------------- networks --------------------------------------------------- #
    model = WsodDense()
    # from torchstat import stat
    # stat(model, (3, 256, 256))
    model.load_state_dict(torch.load(args.ckpt_root))
    model = model.cuda()

    # --------------------------------------------------- infer ----------------------------------------------------- #
    with torch.no_grad():

        print('\nTesting ...  [   ', end='')
        model.eval()
        total_num = len(test_loader)
        count_num = int(total_num / 10)
        total_time = 0

        for idx, (data, img_name, img_size) in enumerate(test_loader):

            start_time = time.time()
            side5, side4, output = model(data.cuda())
            final_time = time.time()
            total_time += final_time - start_time

            if args.pamr:
                output = BinaryPamr(data.cuda(), output, binary=None)

            for i in range(output.shape[0]):
                size = [[img_size[0][i].item()], [img_size[1][i].item()]]
                imsave(os.path.join(args.salmap_root, img_name[i] + '.png'), output[i].squeeze().cpu().detach(), size)

            if idx % count_num == count_num - 1:
                print((str(round(int(idx + 1)/total_num*100))) + '.0 %   ', end='')

        print(']\nFinished.   ', end='')
        print('cost %.4f seconds.   ' % total_time, end='')
        print('FPS: %.1f' % (total_num/total_time*args.batch))

        # ------------- CRF ------------- #
        if args.crf:
            print('\nPerforming CRF .....   \n[   ', end='')

            start_time = time.time()
            crf(input_path=args.test_root, sal_path=args.salmap_root, output_path=args.salmap_crf_root, binary=None)
            final_time = time.time()

            print('],   finished,  ', end='')
            print('cost %d seconds.   ' % (final_time - start_time), end='')
            print('FPS: %.1f' % (total_num/(final_time - start_time)))


if __name__ == '__main__':
    main()
