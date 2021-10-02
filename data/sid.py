import paddle
from paddle.io import Dataset
import os
import glob
import rawpy
import numpy as np

class SID(Dataset):
    def __init__(self, input_dir, gt_dir, ps, DEBUG=False):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps
        # get train IDs
        train_fns = glob.glob(os.path.join(gt_dir, '0*.ARW'))
        self.train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * 6000
        self.input_images = {}
        self.input_images['300'] = [None] * len(self.train_ids)
        self.input_images['250'] = [None] * len(self.train_ids)
        self.input_images['100'] = [None] * len(self.train_ids)
        
        if DEBUG==1:
            self.train_ids = self.train_ids[0:5]
        print(self.train_ids)

    def __len__(self):
        return len(self.train_ids)
    
    def __getitem__(self, index):
        ps = self.ps
        train_id = self.train_ids[index]
        in_files = glob.glob(os.path.join(self.input_dir, '%05d_00*.ARW' % train_id))
        in_path = in_files[np.random.randint(0, len(in_files))]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(os.path.join(self.gt_dir, '%05d_00*.ARW' % train_id))
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        if self.input_images[str(ratio)[0:3]][index] is None:
            raw = rawpy.imread(in_path)
            self.input_images[str(ratio)[0:3]][index] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.gt_images[index] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = self.input_images[str(ratio)[0:3]][index].shape[1]
        W = self.input_images[str(ratio)[0:3]][index].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = self.input_images[str(ratio)[0:3]][index][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = self.gt_images[index][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

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
        input_patch = paddle.to_tensor(input_patch).squeeze(0)
        gt_patch = paddle.to_tensor(gt_patch).squeeze(0)
        return input_patch, gt_patch, train_id, ratio


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
    