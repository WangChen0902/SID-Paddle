#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
import os, time, scipy.io
import paddle
import paddle.nn as nn
import paddle.distributed as dist
import numpy as np
import rawpy
import glob
from data.sid import SID
from paddle.io import DataLoader

train_parameters = {
    'start_epoch': 0,
    'num_epoches': 4000,
    'patch_size': 512,   # patch size for training
    'save_freq': 200,
    'learning_rate': 1e-4,
    'DEBUG': 1,  # 0: not debug with full datasets; 1: debug with few datasets
    'data_prefix': './data/',  # path to load datasets
    'output_prefix': './result/',  # path to save checkpoints and output images
    'checkpoint_load_dir': './checkpoint/'  # path to load checkpoints
}

input_dir = os.path.join(train_parameters['data_prefix'], 'Sony/short/')
gt_dir = os.path.join(train_parameters['data_prefix'], 'Sony/long/')
checkpoint_dir = os.path.join(train_parameters['output_prefix'], 'result_Sony/')  # path to save ckeckpoints
result_dir = os.path.join(train_parameters['output_prefix'], 'result_Sony/')  # path to save output images

ps = train_parameters['patch_size']
save_freq = train_parameters['save_freq']

if train_parameters['DEBUG'] == 1:
    save_freq = 5

dist.init_parallel_env()


class DoubleConv(nn.Layer):
    def __init__(self, input_channels, output_channels, filter_size):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(in_channels=input_channels, out_channels=output_channels, kernel_size=filter_size,
                      padding='SAME'),
            nn.LeakyReLU(0.2),
            nn.Conv2D(in_channels=output_channels, out_channels=output_channels, kernel_size=filter_size,
                      padding='SAME'),
            nn.LeakyReLU(0.2)
        )

    def forward(self, inputs):
        out = self.conv(inputs)
        return out


class Network(nn.Layer):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = DoubleConv(4, 32, 3)
        self.pool1 = nn.MaxPool2D(kernel_size=2, padding='SAME')
        self.conv2 = DoubleConv(32, 64, 3)
        self.pool2 = nn.MaxPool2D(kernel_size=2, padding='SAME')
        self.conv3 = DoubleConv(64, 128, 3)
        self.pool3 = nn.MaxPool2D(kernel_size=2, padding='SAME')
        self.conv4 = DoubleConv(128, 256, 3)
        self.pool4 = nn.MaxPool2D(kernel_size=2, padding='SAME')
        self.conv5 = DoubleConv(256, 512, 3)
        self.up6 = nn.Conv2DTranspose(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding='VALID')
        self.conv6 = DoubleConv(512, 256, 3)
        self.up7 = nn.Conv2DTranspose(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding='VALID')
        self.conv7 = DoubleConv(256, 128, 3)
        self.up8 = nn.Conv2DTranspose(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding='VALID')
        self.conv8 = DoubleConv(128, 64, 3)
        self.up9 = nn.Conv2DTranspose(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding='VALID')
        self.conv9 = DoubleConv(64, 32, 3)
        self.conv10 = nn.Conv2D(in_channels=32, out_channels=12, kernel_size=1, padding='SAME')
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, inputs):
        x1_0 = self.conv1(inputs)
        x1_1 = self.pool1(x1_0)
        x2_0 = self.conv2(x1_1)
        x2_1 = self.pool2(x2_0)
        x3_0 = self.conv3(x2_1)
        x3_1 = self.pool3(x3_0)
        x4_0 = self.conv4(x3_1)
        x4_1 = self.pool4(x4_0)
        x5 = self.conv5(x4_1)
        x6_0 = self.up6(x5)
        x6_1 = self.conv6(paddle.concat([x6_0, x4_0], 1))
        x7_0 = self.up7(x6_1)
        x7_1 = self.conv7(paddle.concat([x7_0, x3_0], 1))
        x8_0 = self.up8(x7_1)
        x8_1 = self.conv8(paddle.concat([x8_0, x2_0], 1))
        x9_0 = self.up9(x8_1)
        x9_1 = self.conv9(paddle.concat([x9_0, x1_0], 1))
        out = self.conv10(x9_1)
        out = self.pixel_shuffle(out)
        return out


dataset = SID(input_dir, gt_dir, ps, DEBUG=train_parameters['DEBUG'])
loader = DataLoader(dataset, batch_size=1, shuffle=True)

net_model = Network()
net_model = paddle.DataParallel(net_model)
net_model.train()
G_loss = paddle.nn.L1Loss(reduction='mean')
optimizer = paddle.optimizer.Adam(learning_rate=train_parameters['learning_rate'], parameters=net_model.parameters())

g_loss = np.zeros((5000, 1))

if train_parameters['start_epoch'] != 0:
    model_name = os.path.join(train_parameters['checkpoint_load_dir'], 'model_%04d.pdparams' % train_parameters['start_epoches'])
    opt_name = os.path.join(train_parameters['checkpoint_load_dir'], 'optimizer_%04d.pdopt' % train_parameters['start_epoches'])
    model_state_dict = paddle.load(model_name)
    opt_state_dict = paddle.load(opt_name)
    net_model.set_state_dict(model_state_dict)
    optimizer.set_state_dict(opt_state_dict)

print('Network initial finished!!')

for epoch in range(train_parameters['start_epoch'], train_parameters['num_epoches']+1):
    cnt = 0
    # print(cnt)
    if epoch > 2000:
        optimizer.set_lr(1e-5)

    for i, data in enumerate(loader()):
        st = time.time()
        input_patch, gt_patch, train_id, ratio = data

        output = net_model(paddle.transpose(input_patch, [0, 3, 1, 2]))
        output = paddle.transpose(output, [0, 2, 3, 1])
        G_current = G_loss(output, gt_patch)
        G_current.backward()
        optimizer.step()
        optimizer.clear_grad()
        output = np.minimum(np.maximum(output.numpy(), 0), 1)
        g_loss[i] = G_current

        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

        if epoch % save_freq == 0:
            if not os.path.isdir(os.path.join(result_dir, '%04d' % epoch)):
                os.makedirs(os.path.join(result_dir, '%04d' % epoch))

            temp = np.concatenate((gt_patch.numpy()[0, :, :, :], output[0, :, :, :]), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                os.path.join(result_dir, '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio)))

            paddle.save(net_model.state_dict(), os.path.join(checkpoint_dir, 'model_%04d.pdparams' % epoch))
            paddle.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_%04d.pdopt' % epoch))
