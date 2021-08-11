import numpy as np
from PIL import Image
import os
import utils.PSNR as PSNR
import utils.SSIM as SSIM


ROOT_PATH = 'E:/fsdownload/result_Sony/final/'

gt_list = []
out_list = []
scale_list = []
psnr_list = []
ssim_list = []

for root, dirs, files in os.walk(ROOT_PATH):
    for f in files:
        fn_list=f.split('.')[0].split('_')
        f_first=fn_list[0]+'_'+fn_list[1]+'_'+fn_list[2]
        gt_name = os.path.join(ROOT_PATH, f_first+'_gt.png')
        out_name = os.path.join(ROOT_PATH, f_first+'_out.png')
        scale_name = os.path.join(ROOT_PATH, f_first + '_scale.png')
        if gt_name not in gt_list:
            gt_list.append(gt_name)
        if out_name not in out_list:
            out_list.append(out_name)
        if scale_name not in scale_list:
            scale_list.append(scale_name)

for i in range(len(gt_list)):
    # gt_array = np.array(cv2.imread(gt_list[i], 0))
    # out_array = np.array(cv2.imread(out_list[i], 0))
    gt_array = np.array(Image.open(gt_list[i]))
    out_array = np.array(Image.open(out_list[i]))
    r12 = PSNR.psnr(gt_array, out_array)
    ss12 = SSIM.calculate_ssim(gt_array, out_array)
    print('r12', r12)
    print('ss12', ss12)
    psnr_list.append(r12)
    ssim_list.append(ss12)

print('mean psnr:', np.mean(psnr_list))
print('mean ssim:', np.mean(ssim_list))
