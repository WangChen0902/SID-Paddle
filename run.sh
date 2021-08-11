#! /bin/bash
unzip -n -d /root/paddlejob/workspace/train_data/datasets/data/ /root/paddlejob/workspace/train_data/datasets/data103409/Sony.zip
pip install rawpy
pip install scipy==1.1.0
python -m paddle.distributed.launch train_Sony_paddle.py
