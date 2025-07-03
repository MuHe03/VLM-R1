#!/bin/bash

# GRPO training script for segmentation task
# This script demonstrates how to use the modified grpo_jsonl_seg.py for segmentation training

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set environment variables
export DEBUG_MODE="true"
export LOG_PATH="./logs/seg_training"

# Create log directory
mkdir -p ./logs

# Run the segmentation training
python src/open_r1/grpo_jsonl_seg.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --data_file_paths "path/to/segmentation_data.jsonl" \
    --image_folders "path/to/images" \
    --output_dir "./outputs/seg_model" \
    --task_type "segmentation" \
    --seg_decoder_path "facebook/mask2former-swin-base" \
    --reward_funcs "seg_accuracy" "format" \
    --reward_method "seg_iou" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --num_train_epochs 3 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 2 \
    --remove_unused_columns False \
    --push_to_hub False \
    --report_to "wandb" \
    --max_pixels 12845056 \
    --min_pixels 3136 \
    --attn_implementation "flash_attention_2" \
    --gradient_checkpointing True \
    --dataloader_pin_memory False \
    --ddp_find_unused_parameters False \
    --ddp_bucket_cap_mb 25 \
    --ddp_broadcast_buffers False \
    --ddp_timeout 1800 \
    --deepspeed "configs/zero3.yaml" \
    --bf16 True 