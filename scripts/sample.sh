#!/bin/bash

mpiexec -n 1 python camelyon_sample.py --batch_size 8 --training_mode consistency_training --sampler multistep \
    --ts 2,2,10 --steps 11 \
    --model_path ./log_folder/online_model_epoch_4.pt \
    --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 \
    --image_size 96 --num_channels 128 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 False --weight_schedule uniform --data_dir ../data \
    --log_path ./log_folder \
    --camelyon_log_file ./log_folder/sample.log \
    --split_file ../splits/hospital_3_test_split.txt \
    --norm_folder ./output_patches \
    --sigma_min 0.0