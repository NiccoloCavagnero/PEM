#!/bin/bash

python train_net.py \
       --config-file ./configs/ade20k/panoptic-segmentation/pem_R50_bs32_160k.yaml \
       --num-gpus 2 \
       DATALOADER.NUM_WORKERS 8