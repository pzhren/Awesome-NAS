#!/usr/bin/env bash
nvidia-smi
mount -t tmpfs -o size=1G tmpfs /userhome/temp_data
cp -r /userhome/data/cifar10 /userhome/temp_data/
# pip3 install -U torch==1.1.0.post2
python /userhome/project/pt.darts/augment.py --dataset cifar10 --name random_100 \
--file /userhome/project/Auto_NAS/experiment/random_darts_architecture.txt --epochs 100
