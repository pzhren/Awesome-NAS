#!/usr/bin/env bash
nvidia-smi
mount -t tmpfs -o size=1G tmpfs /userhome/temp_data
cp -r /userhome/data/cifar10 /userhome/temp_data/
pip3 install -U torch==1.1.0.post2
python /userhome/project/pt.darts/augment.py --name DARTS --dataset cifar10 --genotype "Genotype(normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_3x3', 1), ('skip_connect', 0)], [('skip_connect', 0), ('dil_conv_3x3', 2)]], normal_concat=[2, 3, 4, 5], reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('skip_connect', 2), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('skip_connect', 2), ('max_pool_3x3', 1)]], reduce_concat=[2, 3, 4, 5])"