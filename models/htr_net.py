import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class HTRNet(nn.Module):
    def __init__(self, cnn_cfg, rnn_cfg, nclasses):
        super(HTRNet, self).__init__()

        #cfg = [(2, 16), 'M', (4, 32), 'M', (6, 64), 'M', (2, 128)]

        in_channels = 1
        self.features = nn.ModuleList([])
        cntm = 0
        cnt = 1
        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            else:
                for i in range(m[0]):
                    x = m[1]
                    self.features.add_module('cnv' + str(cnt), nn.Conv2d(in_channels, x, 3, 1, 1, bias=True))
                    #self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x))
                    in_channels = x
                    self.features.add_module('nl' + str(cnt), nn.Sequential(nn.BatchNorm2d(x, momentum=.5), nn.ReLU()))
                    #self.features.add_module('nl' + str(cnt), nn.ReLU())
                    cnt += 1


        rnn_in = cnn_cfg[-1][-1]
        hidden, num_layers = rnn_cfg

        self.rec = nn.LSTM(rnn_in, hidden, num_layers=num_layers, bidirectional=True)

        self.fnl = nn.Sequential(nn.Linear(2*hidden, 512), nn.ReLU(), nn.Dropout(.5), nn.Linear(512, nclasses))

    def forward(self, x):

        y = x
        for nn_module in self.features:
            y = nn_module(y)

        y = F.max_pool2d(y, [y.size(2), 1], padding=[0, 0])
        y = y.permute(2, 3, 0, 1)[0]  # 1 x seq_len x batch_size x feat
        y = self.rec(y)[0] #.view(1, -1)
        y = self.fnl(y)

        return y

