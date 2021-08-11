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

train_parameters = {
    'start_epoch': 0,
    'num_epoches': 4001,
    'patch_size': 512,   # patch size for training
    'save_freq': 200,
    'learning_rate': 1e-4,
    'DEBUG': 0,  # 0: not debug with full datasets; 1: debug with few datasets
    'data_prefix': './data/',  # path to load datasets
    'output_prefix': './result/',  # path to save checkpoints and output images
    'checkpoint_load_dir': './checkpoint/'  # path to load checkpoints
}

input_dir = os.path.join(train_parameters['data_prefix'], 'Sony/short/')
gt_dir = os.path.join(train_parameters['data_prefix'], 'Sony/long/')
checkpoint_dir = os.path.join(train_parameters['output_prefix'], 'result_Sony/')  # path to save ckeckpoints
result_dir = os.path.join(train_parameters['output_prefix'], 'result_Sony/')  # path to save output images

# get train IDs
train_fns = glob.glob(os.path.join(gt_dir, '0*.ARW'))
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

ps = train_parameters['patch_size']
save_freq = train_parameters['save_freq']

if train_parameters['DEBUG'] == 1:
    save_freq = 5
    train_ids = train_ids[0:5]

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


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


net_model = Network()
net_model = paddle.DataParallel(net_model)
net_model.train()
G_loss = paddle.nn.L1Loss(reduction='mean')
optimizer = paddle.optimizer.Adam(learning_rate=train_parameters['learning_rate'], parameters=net_model.parameters())

# Raw data takes long time to load. Keep them in memory after loaded.
gt_images = [None] * 6000
input_images = {}
input_images['300'] = [None] * len(train_ids)
input_images['250'] = [None] * len(train_ids)
input_images['100'] = [None] * len(train_ids)

g_loss = np.zeros((5000, 1))

if train_parameters['start_epoch'] != 0:
    model_name = os.path.join(train_parameters['checkpoint_load_dir'], 'model_%04d.pdparams' % train_parameters['start_epoch'])
    opt_name = os.path.join(train_parameters['checkpoint_load_dir'], 'optimizer_%04d.pdopt' % train_parameters['start_epoch'])
    model_state_dict = paddle.load(model_name)
    opt_state_dict = paddle.load(opt_name)
    net_model.set_state_dict(model_state_dict)
    optimizer.set_state_dict(opt_state_dict)

print('Network initial finished!!')

for epoch in range(train_parameters['start_epoch'], train_parameters['num_epoches']):
    cnt = 0
    # print(cnt)
    if epoch > 2000:
        optimizer.set_lr(1e-5)

    for ind in np.random.permutation(len(train_ids)):
        # print(ind)
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(os.path.join(input_dir, '%05d_00*.ARW' % train_id))
        in_path = in_files[np.random.randint(0, len(in_files))]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(os.path.join(gt_dir, '%05d_00*.ARW' % train_id))
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)

        output = net_model(paddle.transpose(paddle.to_tensor(input_patch), [0, 3, 1, 2]))
        output = paddle.transpose(output, [0, 2, 3, 1])
        G_current = G_loss(output, paddle.to_tensor(gt_patch))
        G_current.backward()
        optimizer.step()
        optimizer.clear_grad()
        output = np.minimum(np.maximum(output.numpy(), 0), 1)
        g_loss[ind] = G_current

        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

        if epoch % save_freq == 0:
            if not os.path.isdir(os.path.join(result_dir, '%04d' % epoch)):
                os.makedirs(os.path.join(result_dir, '%04d' % epoch))

            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                os.path.join(result_dir, '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio)))

            paddle.save(net_model.state_dict(), os.path.join(checkpoint_dir, 'model_%04d.pdparams' % epoch))
            paddle.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_%04d.pdopt' % epoch))
