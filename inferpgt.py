
import os
import time
import torch
import torch.nn.functional as f

from utils.crf import crf
from utils.imsave import imsave
from utils.pamr import BinaryPamr


class InferPgt(object):

    def __init__(self, model, infer_loader, image_root, cam_pamr_root, cam_pamr_crf_root):

        self.model = model
        self.infer_loader = infer_loader
        self.image_root = os.getcwd() + '/' + image_root + '/DUTS-train/image'
        self.cam_pamr_root = cam_pamr_root
        self.cam_pamr_crf_root = cam_pamr_crf_root

    def infer(self):

        print(' Inferring .....   \n[  ', end='')

        self.model.eval()
        torch.cuda.empty_cache()
        total_num = len(self.infer_loader)
        count_num = int(total_num / 10)
        total_time = 0

        for idx, (data, img_name, size) in enumerate(self.infer_loader):
            cams = []
            img_size = (int(size[0]), int(size[1]))
            start_time = time.time()
            outputs = [self.model(img[0].cuda(non_blocking=True)) for img in data]

            for o in outputs:
                cams.append(o[1][0] + o[1][1].flip(-1))
            cam = torch.sum(torch.stack(cams), 0)

            cls = (outputs[0][0] + outputs[1][0] + outputs[2][0] + outputs[3][0]) / 4
            cls = (cls[0] + cls[1]) / 2
            cls = torch.sigmoid(cls).view(cls.shape[0], 1, 1)

            cam = torch.sum(cam * cls, 0).unsqueeze(0)
            cam /= f.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

            img = data[0][0][0].unsqueeze(0)
            img = f.interpolate(img, (img_size[1] // 4, img_size[0] // 4), mode='bilinear', align_corners=False)
            cam = f.interpolate(cam.unsqueeze(0), (img_size[1] // 4, img_size[0] // 4), mode='bilinear',
                                align_corners=False)

            out_refine = BinaryPamr(img.cuda(), cam, binary=None)
            out_refine /= f.adaptive_max_pool2d(out_refine, (1, 1)) + 1e-5
            out_refine = f.interpolate(out_refine, (img_size[1], img_size[0]), mode='bilinear', align_corners=False)

            # cam = cam.squeeze().cpu().detach()
            # imsave(os.path.join('data/pseudo_labels/cam', img_name[0] + '.png'), cam, img_size)

            final_time = time.time()
            total_time += final_time - start_time
            out_refine = out_refine.squeeze().cpu().detach()
            imsave(os.path.join(self.cam_pamr_root, img_name[0] + '.png'), out_refine, size)

            if idx % count_num == count_num - 1:
                print((str(round(int(idx + 1) / total_num * 100))) + '.0 %   ', end='')

        torch.cuda.empty_cache()

        print('],   finished,  ', end='')
        print('cost %d seconds.   ' % total_time, end='')
        print('FPS: %.1f' % (total_num / total_time))

        # ------------- CRF ------------- #
        print('\nPerforming CRF .....   \n[  ', end='')

        start_time = time.time()
        crf(input_path=self.image_root, sal_path=self.cam_pamr_root, output_path=self.cam_pamr_crf_root, binary=None)
        final_time = time.time()

        print('],   finished,  ', end='')
        print('cost %d seconds.   ' % (final_time - start_time), end='')
        print('FPS: %.1f' % (total_num / (final_time - start_time)))
