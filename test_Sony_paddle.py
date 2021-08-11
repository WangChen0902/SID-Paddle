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
import os, scipy.io
import paddle
import paddle.nn as nn
import numpy as np
import rawpy
import glob

test_parameters = {
    'last_epoch': 4000,  # last epoch of training
    'DEBUG': 0,  # 0: not debug with full datasets; 1: debug with few datasets
    'data_prefix': './data/',  # path to load datasets
    'output_prefix': './result/',  # path to save checkpoints and output images
    'checkpoint_load_dir': './checkpoint/'  # path to load checkpoints
}


input_dir = os.path.join(test_parameters['data_prefix'], 'Sony/short/')
gt_dir = os.path.join(test_parameters['data_prefix'], 'Sony/long/')
checkpoint_dir = os.path.join(test_parameters['output_prefix'], 'result_Sony/')  # path to save ckeckpoints
result_dir = os.path.join(test_parameters['output_prefix'], 'result_Sony/')  # path to save output images

# get test IDs
test_fns = glob.glob(os.path.join(gt_dir, '1*.ARW'))
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

DEBUG = test_parameters['DEBUG']
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


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
net_model.eval()
model_name = os.path.join(test_parameters['checkpoint_load_dir'], 'model_%04d.pdparams' % test_parameters['last_epoch'])
model_state_dict = paddle.load(model_name)
net_model.set_state_dict(model_state_dict)
print('Network initial finished!!')
for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(os.path.join(input_dir, '%05d_00*.ARW' % test_id))
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_files = glob.glob(os.path.join(gt_dir, '%05d_00*.ARW' % test_id))
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        input_full = np.minimum(input_full, 1.0)

        output = net_model(paddle.transpose(paddle.to_tensor(input_full), [0, 3, 1, 2]))
        output = paddle.transpose(output, [0, 2, 3, 1])
        output = np.minimum(np.maximum(output.numpy(), 0), 1)

        output = output[0, :, :, :]
        gt_full = gt_full[0, :, :, :]
        scale_full = scale_full[0, :, :, :]
        scale_full = scale_full * np.mean(gt_full) / np.mean(
            scale_full)  # scale the low-light image to the same mean of the groundtruth

        if not os.path.isdir(os.path.join(result_dir, 'final')):
            os.mkdir(os.path.join(result_dir, 'final'))
        scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            os.path.join(result_dir, 'final/%5d_00_%d_out.png' % (test_id, ratio)))
        scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            os.path.join(result_dir, 'final/%5d_00_%d_scale.png' % (test_id, ratio)))
        scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            os.path.join(result_dir, 'final/%5d_00_%d_gt.png' % (test_id, ratio)))
