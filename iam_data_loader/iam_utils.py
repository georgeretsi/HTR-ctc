import numpy as np
import os
from iam_config import *


def gather_iam_info(set='train', level='word'):

    # train/test file
    if set == 'train':
        valid_set = np.loadtxt(trainset_file, dtype=str)
    elif set == 'test':
        valid_set = np.loadtxt(testset_file, dtype=str)
    else:
        print('shitloader')
        return


    if level == 'word':
        gtfile= word_file
        root_path = word_path
    elif level == 'line':
        gtfile = line_file
        root_path = line_path
    else:
        print('shitloader')
        return

    gt = []
    for line in open(gtfile):
        if not line.startswith("#"):
            info = line.strip().split()
            name = info[0]

            name_parts = name.split('-')
            pathlist = [root_path] + ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]
            if level == 'word':
                line_name = pathlist[-2]
                del pathlist[-2]
            elif level == 'line':
                line_name = pathlist[-1]

            if (info[1] != 'ok') or (line_name not in valid_set):
                continue

            img_path = '/'.join(pathlist)

            transcr = ' '.join(info[8:])
            gt.append((img_path, transcr))

    return gt
