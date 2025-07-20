export CUDA_VISIBLE_DEVICES=0

python scripts/eval_segmentation.py \
    --model_path=saves/custom_seg_2/full/sft/checkpoint-20000/ \
    --val_data_path=../Qwen_val.json 