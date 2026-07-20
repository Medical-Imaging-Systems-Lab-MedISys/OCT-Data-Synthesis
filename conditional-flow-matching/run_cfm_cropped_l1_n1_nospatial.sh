#!/bin/bash
# Submit this script on tanuh n1 node directly via bash

# Run L1 model without spatial weighting on GPU 3
CUDA_VISIBLE_DEVICES=3 nohup /data/vds/env_pt/bin/python -u conditional-flow-matching/train_cropped_multidataset.py \
    --loss_type l1 \
    --pseudo_weight 0.5 \
    --no_spatial_weighting \
    > logs/cfm_cropped_l1_nospatial_n1.log 2>&1 &
