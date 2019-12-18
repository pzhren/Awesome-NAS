#!/usr/bin/env bash
echo "System information:"
cat /etc/issue
echo "cuda_information:"
cat /usr/local/cuda/version.txt
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2