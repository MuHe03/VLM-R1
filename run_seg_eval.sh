#!/bin/bash

#[DONE:dyzhou:14.07] 分割验证运行脚本 - 一键执行IoU/GIoU/CIoU评估
# Segmentation evaluation script for vlm_seg_full_sft
# This script evaluates the trained segmentation model on Qwen_val.json

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your GPU setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Configuration
MODEL_PATH="saves/custom_seg_2/full/sft/checkpoint-20000"  # Adjust to your model path
VAL_DATA_PATH="../Qwen_val.json"  # Path to your validation data
OUTPUT_DIR="./logs/seg_evaluation"
BATCH_SIZE=1
 
# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "Starting segmentation evaluation..."
echo "Model path: ${MODEL_PATH}"
echo "Validation data: ${VAL_DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
NUM_GPUS=4
# Run evaluation
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
    scripts/eval_segmentation.py \
    --model_path ${MODEL_PATH} \
    --val_data_path ${VAL_DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE}

echo "Evaluation completed. Results saved to ${OUTPUT_DIR}" 