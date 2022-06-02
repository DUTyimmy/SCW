"""
Paper: To be Critical: Self-Calibrated Weakly Supervised Learning for Salient Object Detection
Author: Yimmy (Jian Wang)
College: The IIAU-OIP Lab, Dalian University of Technology
"""


def main():
    import os
    import time
    import torch
    import argparse

    from traincam import TrainCam
    from inferpgt import InferPgt
    from trainsal import TrainSal
    from utils.imsave import imsave
    from utils.pamr import BinaryPamr
    from utils.datainit import traindatainit
    from model.cls_model import CamDense
    from model.sal_model import WsodDense
    from torch.utils.data import DataLoader
    from dataset_loader import MySalTrainData, MySalInferData, MyCamTrainData,\
        MyCamInferData, MySalValData
    # from utils.crf import crf
    # from dataset_loader import ImageNetClsData

    # -------------------------------------------------- options --------------------------------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_stage', type=str, default='1', choices=['1', '2'], help='training stage')
    parser.add_argument('--param', type=str, default='train', choices=['train', 'infer'])
    parser.add_argument('--num_workers', type=int, default=12, help='the CPU workers number')
    parser.add_argument('--resize', type=int, default=256, help='resized size of images')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='wight_decay')
    parser.add_argument('--ckpt_root', type=str, default='snapshot', help='path to save ckpt')

    # settings of stage 1 (cam)
    parser.add_argument('--cam_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--cam_batch', type=int, default=40, help='batch size')
    parser.add_argument('--cam_max_epoch', type=int, default=20, help='the max epoch')
    parser.add_argument("--scales", default=(1.0, 0.5, 1.5, 2.0), help="multi-scale inferences")
    parser.add_argument('--pgt_root', type=str, default='data/pseudo_labels/label0', help='path to pseudo labels')

    # settings of stage 2 (sal)
    parser.add_argument('--sal_stage', type=int, default=2, help='training stage n')
    parser.add_argument('--sal_lr', type=float, default=3e-6, help='learning rate')
    parser.add_argument('--sal_batch', type=int, default=20, help='batch size')
    parser.add_argument('--first_epoch_nums', type=int, default=5, help='the max epoch')
    parser.add_argument('--second_epoch_nums', type=int, default=15, help='the extra epoch of train stage n')
    parser.add_argument('--lam', type=int, default=0.6, help='the lambda of self-calibrated strategy')
    parser.add_argument('--val', type=bool, default=False, help='whether validation or not')
    parser.add_argument('--data_root', type=str, default='data', help='path to infer and train data')
    args = parser.parse_args()
    ori_root = os.getcwd()
    traindatainit(args.ckpt_root, args.data_root, args.sal_stage)

    # ------------------------------------------------ dataloaders ------------------------------------------------- #
    traincam_loader = DataLoader(MyCamTrainData(root=args.data_root, transform=True, resize=args.resize),
                                 batch_size=args.cam_batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # train_loader = DataLoader(ImageNetClsData(r'G:\ILSVRC2014_devkit', transform=True, resize=args.resize),
    #                           batch_size=args.cam_batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    inferpgt_loader = DataLoader(MyCamInferData(args.data_root, transform=False, scales=args.scales),
                                 batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    infersal_loader = DataLoader(MySalInferData(args.data_root, transform=True), batch_size=1, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    valsal_loader = DataLoader(MySalValData(args.data_root, resize=args.resize, transform=True),
                               batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # -------------------------------------------------- networks -------------------------------------------------- #
    model_cam = CamDense(pretrained=True, cls_num=44)
    model_sal = WsodDense()

    # computing the FLOPs and Parm
    # from torchstat import stat
    # stat(model_sal, (3, args.resize, args.resize))

    model_cam = model_cam.cuda()
    model_sal = model_sal.cuda()

    # -------------------------------------------------- training -------------------------------------------------- #
    if args.train_stage == "1":
        # #################### stage 1 #################### #

        # train
        print('\n[ Stage 1 : training a classification network on DUTS-CLS dataset. ]\n')
        model_cam.train()
        optimizer_model = torch.optim.Adam(model_cam.parameters(), lr=args.cam_lr, weight_decay=args.weight_decay)
        training = TrainCam(
            model=model_cam,
            optimizer_model=optimizer_model,
            train_loader=traincam_loader,
            max_epoch=args.cam_max_epoch,
            outpath=args.ckpt_root)

        training.epoch, training.iteration = 0, 0
        training.train()
        torch.save(model_cam.state_dict(), ('%s/cam_epoch_%d.pth' % (args.ckpt_root, args.cam_max_epoch)))
        print('save: (cam_epoch_%d.pth)' % args.cam_max_epoch)

        # infer
        # model_cam.load_state_dict(torch.load('snapshot/cam_epoch_20.pth'))
        model_cam.eval()
        print('\n[ Stage 1 : performing inference on DUTS-TRAIN dataset to generate pseudo labels. ]\n')
        with torch.no_grad():
            pseudo = InferPgt(model_cam, inferpgt_loader, args.data_root, args.pgt_root, args.pgt_root)
            pseudo.infer()

    # #################### stage 2 #################### #
    print('\n[ Stage 2 : training a saliency network using pseudo labels. ]\n')
    for i in range(1, (args.sal_stage*2)):

        os.chdir(ori_root)
        # train
        if args.param == 'train':
            model_sal.train()
            args.param = 'infer'

            trainsal_loader = DataLoader(MySalTrainData(args.data_root, resize=args.resize, transform=True,
                                                        stage=int((i-1)/2)), batch_size=args.sal_batch,
                                         shuffle=True, num_workers=args.num_workers, pin_memory=True)
            optimizer_model = torch.optim.Adam(model_sal.parameters(), lr=args.sal_lr, weight_decay=args.weight_decay)

            if not args.val:
                valsal_loader = None
            training = TrainSal(
                model=model_sal,
                optimizer_model=optimizer_model,
                train_loader=trainsal_loader,
                val_loader=valsal_loader,
                outpath=args.ckpt_root,
                max_epoch=args.first_epoch_nums + args.second_epoch_nums*((i+1)/2-1),
                stage=(i+1)/2,
                maximum=args.lam)

            training.epoch = 0
            training.iteration = 0
            training.train()

    # --------------------------------------------------- infer ---------------------------------------------------- #
        elif args.param == 'infer':
            args.param = 'train'
            torch.cuda.empty_cache()
            ckpt_name = 'sal_stage_' + str(int(i/2)) + '.pth'
            model_sal.load_state_dict(torch.load(os.path.join(args.ckpt_root, ckpt_name)))
            model_sal.eval()

            print('\nInferring .....   \n[  ', end='')

            total_num = len(infersal_loader)
            count_num = int(total_num / 10)
            start_time = time.time()

            for idx, (data, name, size) in enumerate(infersal_loader):
                _, _, sal = model_sal.forward(data.cuda())

                sal = BinaryPamr(data.cuda(), sal, binary=0.4)
                sal = sal.squeeze().cpu().detach()
                imsave(os.path.join('data/pseudo_labels/label' + str(int(i/2)), name[0] + '.png'), sal, size)

                if idx % count_num == count_num - 1:
                    print((str(round(int(idx + 1) / total_num * 100))) + '.0 %   ', end='')

            print(' ],  finished,  ', end='')
            final_time = time.time()
            print('cost %d seconds.  ' % (final_time - start_time), end='\n\n')
            torch.cuda.empty_cache()

            # # ------------- CRF ------------- #
            # print('\nInferring .....   \n[  ', end='')
            # start_time = time.time()
            #
            # crf(input_path='G:/DUTS-TRAIN/DUTS-TR-Image', sal_path='data/pseudo_labels/label' + str(int(i/2)),
            #     output_path='data/pseudo_labels/label' + str(int(i/2)), binary=None)
            # os.chdir(ori_root)
            # print(' ],  finished,  ', end='')
            # final_time = time.time()
            # print('cost %d seconds.  ' % (final_time - start_time), end='\n\n')

            # reload the model for self-training
            del model_sal
            model_sal = WsodDense()
            model_sal = model_sal.cuda()


if __name__ == '__main__':
    main()
