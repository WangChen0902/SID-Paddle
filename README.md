# SID-Paddle

English | [简体中文](./README_cn.md)

   * [SID-Paddle](#sid-paddle)
      * [1 Introduction](#1-introduction)
      * [2 Reproduction Accuracy](#2-reproduction-accuracy)
      * [3 Dataset](#3-dataset)
      * [4 Environment](#4-environment)
      * [5 Quick Start](#5-quick-start)
         * [step1: Clone](#step1-clone)
         * [step2: Download Data](#step2-download-data)
         * [step3: Train](#step3-train)
         * [step4: Test](#step4-test)
         * [step5: Evaluate](#step5-evaluate)
      * [6 Code Structure and Explanation](#6-code-structure-and-explanation)
         * [6.1 Code Structure](#61-code-structure)
         * [6.2 Parameter Explanation](#62-parameter-explanation)
      * [7 Information](#7-model-information)

## 1 Introduction

This is a Paddlepaddle implementation of：Learning to See in the Dark in CVPR 2018, by [Chen Chen](http://cchen156.github.io/), [Qifeng Chen](http://cqf.io/), [Jia Xu](http://pages.cs.wisc.edu/~jiaxu/), and [Vladlen Koltun](http://vladlen.info/).
  
**Raw code：**[Learning-to-See-in-the-Dark](https://github.com/cchen156/Learning-to-See-in-the-Dark)

**Paper：**[Learning to See in the Dark ](http://cchen156.github.io/paper/18CVPR_SID.pdf)

This code includes the default model for training and testing on the See-in-the-Dark (SID) dataset.

## 2 Reproduction Accuracy

| Index | Raw Paper | Raw Code | Reproduction |
| --- | --- | --- | --- |
| PSNR | 28.88 | 28.96 | 28.82 |
| SSIM | 0.787 | 0.785 | 0.787 |

## 3 Dataset
The dataset is [SID-Sony](https://pan.baidu.com/s/1fk8EibhBe_M1qG0ax9LQZA#list/path=%2F).
If you only use Sony dataset, please download the files started with Sony. After you download all the parts, you can combine them together by running: "cat SonyPart* > Sony.zip" and "cat FujiPart* > Fuji.zip".

- As mentioned in raw README, the authors found some misalignment with the ground-truth for image 10034, 10045, 10172. Please remove those images for quantitative results, but they still can be used for qualitative evaluations.

- The dataset contains 2697 short-exposed images and 231 long-exposed images. Note that multiple short-exposed images may correspond to the same long-exposed image.
  - Train set：1865 short-exposed +161 long-exposed images
  - Validation set：234 short-exposed +20 long-exposed images
  - Test set：598 short-exposed +50 long-exposed images
- Data format: The proposed method is designed for sensor raw data. The pretrained model probably not work for data from another camera sensor. We do not have support for other camera data. It also does not work for images after camera ISP, i.e., the JPG or PNG data.

## 4 Environment
- Hardware: Paddle AI Studio 4 * Tesla V100, 128G. It takes 16 hours for 4000 epochs. The GPU memory should be greater than 64 GB.
- Frameworks: 
  - Paddlepaddle >= 2.0.0
  - rawpy
  - scipy == 1.1.0

## 5 Quick Start

### Step1: Clone

```bash
# clone this repo
git clone git://github.com/WangChen0902/SID-Paddle.git
```
After downloading, please create three folders: "data", "checkpoint" and "result" in the root path.

### Step2: Download Data

Place the datafiles mentioned above in "data" directory as format: "SID-Paddle/data/Sony".

Place the pre-trained models in "checkpoint" directory as format: "SID-Paddle/checkpoint".

### Step3: Train

```bash
python train_Sony_paddle.py  # Single GPU
python -m paddle.distributed.launch train_Sony_paddle.py  # Multi-GPU
```


### Step4: Test

```bash
python test_Sony_paddle.py
```

### Step5: Evaluate

```bash
python eval.py
```

## 6 Code Structure and Explanation

### 6.1 Code Structure

```
├── checkpoint  # path to saved models
├── data  # path to data
├── result  # path to output
├── utils  # utils
│   ├── PSNR.py
│   └── SSIM.py
├── eval.py  # the script of calculating PSNR/SSIM
├── README.md
├── run.sh  # the script AI Studio multi-GPU training
├── run_test.sh  # the script of single GPU testing
├── test_Sony_paddle.py  # Testing
└── train_Sony_paddle.py  # Training
```

### 6.2 Parameter Explanation

|  Parameter Name  | Default Value  | Description |
|  ----  |  ----  |  ----  |
| start_epoch | 0 | start epoch |
| num_epoches | 4001 | number of epochs |
| patch_size | 512 | patch size of training images |
| save_freq | 200 | save frequency of training |
| learning_rate | 1e-4 | learning rate |
| DEBUG | 0 | if debug |
| data_prefix | './data/' | path to data |
| output_prefix | './result/' | path to output |
| checkpoint_load_dir | './checkpoint/' | path to checkpoint |
| last_epoch | 4000 | epoch of testing |

## 7 Model Information

|  Information Name   |  Description |
|  ----  |  ----  |
| Author | Wangchen0902 |
| Time | 2021.08 |
| Framework | Paddle 2.1.2 |
| Application Scenario | Image enhancement |
| Hardware | GPU>=64G |
| Download | [Pretrained model](https://pan.baidu.com/s/1FF1K3lbsTT24tY91qIUZWg) Access code: 6hbx |
| Download | [Training logs](https://pan.baidu.com/s/1q7HvQVRwZxoGQHon_tO2YA) Access code: brfz |
| Online Running | [SID-Paddle](https://aistudio.baidu.com/aistudio/projectdetail/2275443) |
