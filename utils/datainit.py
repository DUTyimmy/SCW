import os


def traindatainit(ckpt_path, data_path, sal_stage):

    data_path = os.path.join(data_path, 'pseudo_labels')

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    for i in range(int(sal_stage)):
        label_path = os.path.join(data_path, 'label' + str(i))
        if not os.path.exists(label_path):
            os.makedirs(label_path)


def valdatainit(val_path='data/val_map'):

    if not os.path.exists(val_path):
        os.makedirs(val_path)


def testdatainit(sal_path, crf=False):

    if os.path.exists(sal_path):
        for file in os.listdir(sal_path):
            file_path = sal_path + '/' + file
            if os.path.isfile(file_path):
                os.remove(file_path)
        # shutil.rmtree(sal_path)

    if not os.path.exists(sal_path):
        os.makedirs(sal_path)

    if crf:
        if os.path.exists(sal_path + '_crf'):
            for file in os.listdir(sal_path + '_crf'):
                file_path = sal_path + '_crf' + '/' + file
                if os.path.isfile(file_path):
                    os.remove(file_path)
        if not os.path.exists(sal_path + '_crf'):
            os.makedirs(sal_path + '_crf')
