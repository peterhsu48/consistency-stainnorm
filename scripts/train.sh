#!/bin/bash

mpiexec -n 1 python cm_train.py --camelyon True --training_mode consistency_training \
    --target_ema_mode zero --ema_rate 0 --scale_mode improved \
    --num_epochs 4 --start_epoch 0 \
    --loss_norm pseudo_huber --lr_anneal_steps 0 --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False \
    --dropout 0.0 --batch_size 8 --image_size 96 --lr 0.0001 --num_channels 128 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True \
    --schedule_sampler uniform --use_fp16 False --weight_decay 0.0 --weight_schedule uniform --data_dir ../data \
    --log_path ./log_folder --camelyon_log_file ./log_folder/train.log \
    --split_file ../splits/hospital_4_train_split.txt \
    --sigma_min 0.0

# to resume training:
#   --resume_online_checkpoint [FILENAME]
#   --resume_target_checkpoint [FILENAME]
#   --resume_opt_checkpoint [FILENAME]
#   --start_epoch [EPOCH]    