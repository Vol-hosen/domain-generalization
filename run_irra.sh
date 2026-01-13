#!/bin/bash
DATASET_NAME="CUHK-PEDES" # 或者 CUHK-PEDES
DATA_DIR='/mnt/hardisk/wucan/datasets/'
GPU_ID=0               # 指定空闲的 GPU

# --- 运行命令 ---
export CUDA_VISIBLE_DEVICES=$GPU_ID

python train.py \
--name irra-cuhk-kl-v2 \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--num_workers 8 \
--root_dir $DATA_DIR  \
--cons_eps 5e-2 \
--cons_warmup_epochs 5 \
