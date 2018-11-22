import argparse
import logging

import numpy as np
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import tqdm
import torch.backends.cudnn as cudnn

from iam_data_loader.iam_loader import IAMLoader
from config import *

import warpctc_pytorch as warp_ctc
import ctcdecode

from models.htr_net import HTRNet

from utils.save_load import my_torch_save, my_torch_load

from utils.auxilary_functions import torch_augm

from os.path import isfile

import torch.nn.functional as F

logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('PHOCNet-Experiment::train')
logger.info('--- Running PHOCNet Training ---')
# argument parsing
parser = argparse.ArgumentParser()
# - train arguments
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                    help='lr')
parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                    help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
parser.add_argument('--display', action='store', type=int, default=50,
                    help='The number of iterations after which to display the loss values. Default: 100')
parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
# - experiment arguments
#parser.add_argument('--load_model', '-lm', action='store', default=None,
#                    help='The name of the pretrained model to load. Defalt: None, i.e. random initialization')
#parser.add_argument('--save_model', '-sm', action='store', default='whateva.pt',
#                    help='The name of the file to save the model')


args = parser.parse_args()

# train as:
# -lrs 5000:1e-4,10000:1e-5 -bs 1 -is 10 -fim 36  -gpu 0 --test_interval 1000

gpu_id = args.gpu_id

# print out the used arguments
logger.info('###########################################')
logger.info('Experiment Parameters:')
for key, value in vars(args).iteritems():
    logger.info('%s: %s', str(key), str(value))
logger.info('###########################################')

# prepare datset loader

logger.info('Loading dataset.')

train_set = IAMLoader('train', level='line', fixed_size=(128, None))
test_set = IAMLoader('test', level='line', fixed_size=(128, None))

# augmentation using data sampler
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)


# load CNN
logger.info('Preparing Net...')

net = HTRNet(cnn_cfg, rnn_cfg, len(classes))

if load_model_name is not None:
    my_torch_load(net, load_model_name)
net.cuda(args.gpu_id)


loss = warp_ctc.CTCLoss()
net_parameters = net.parameters()
nlr = args.learning_rate
optimizer = torch.optim.Adam(net_parameters, nlr, weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 80])

decoder = ctcdecode.CTCBeamDecoder([c for c in classes], beam_width=100)
# decoder = ctcdecode.

def train():
    optimizer.zero_grad()

    closs = []
    for iter_idx, (img, transcr) in enumerate(train_loader):

        img = Variable(img.cuda(gpu_id))
        # cuda augm - alternatively for cpu use it on dataloader
        img = torch_augm(img)
        output = net(img)

        act_lens = torch.IntTensor(img.size(0)*[output.size(0)])
        labels = Variable(torch.IntTensor([cdict[c] for c in ''.join(transcr)]))
        label_lens = torch.IntTensor([len(t) for t in transcr])

        loss_val = loss(output.cpu(), labels, act_lens, label_lens)
        closs += [loss_val.data]

        loss_val.backward()

        if iter_idx % iter_size == iter_size - 1:
            optimizer.step()
            optimizer.zero_grad()


        # mean runing errors??
        if iter_idx % (args.display*iter_size) == (args.display*iter_size)-1:
            logger.info('Iteration %d: %f', iter_idx+1, sum(closs)/len(closs))
            closs = []

            tst_img, tst_transcr = test_set.__getitem__(np.random.randint(test_set.__len__()))
            with torch.no_grad():
                tst_o = net(Variable(tst_img.cuda(gpu_id)).unsqueeze(0))
            tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
            #for i, tdec in enumerate(declbls):

            print('orig:: ' + tst_transcr)
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            print('greedy dec:: ' + ''.join([icdict[t] for t in tt]).replace('_', ''))
            tdec, _, _, tdec_len = decoder.decode(tst_o.softmax(2).permute(1, 0, 2))
            print('beam dec:: ' + ''.join([icdict[t.item()] for t in tdec[0, 0][:tdec_len[0, 0].item()]]))

import editdistance
# slow implementation
def test(epoch):
    net.eval()

    logger.info('Testing at epoch %d', epoch)
    cer, wer = [], []
    for (img, transcr) in test_loader:
        transcr = transcr[0]
        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            o = net(img)
        tdec, _, _, tdec_len = decoder.decode(o.softmax(2).permute(1, 0, 2))
        dec_transcr = ''.join([icdict[t.item()] for t in tdec[0, 0][:tdec_len[0, 0].item()]])

        cer += [float(editdistance.eval(dec_transcr, transcr))/ len(transcr)]
        wer += [float(editdistance.eval(dec_transcr.split(' '), transcr.split(' '))) / len(transcr.split(' '))]

    logger.info('CER at epoch %d: %f', epoch, sum(cer) / len(cer))
    logger.info('WER at epoch %d: %f', epoch, sum(wer) / len(wer))


    net.train()


cnt = 0
logger.info('Training:')
for epoch in range(1, max_epochs + 1):

    scheduler.step()
    train()

    if epoch % 5 == 0:
        test(epoch)

    if epoch % 10 == 0:
        logger.info('Saving net after %d epochs', epoch)
        my_torch_save(net, save_model_name)
        net.cuda(gpu_id)


my_torch_save(net, save_model_name)