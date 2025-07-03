PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"

# Change the data_paths and image_folders to your own data
data_paths="/training/shz/dataset/vlm-r1/rec_jsonsl_train/refcoco_train.jsonl:/training/shz/dataset/vlm-r1/rec_jsonsl_train/refcocop_train.jsonl:/training/shz/dataset/vlm-r1/rec_jsonsl_train/refcocog_train.jsonl" 
image_folders="/training/shz/dataset/coco:/training/shz/dataset/coco:/training/shz/dataset/coco"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
SEG_DECODER_PATH="facebook/mask2former-swin-base"
ARROW_CACHE="/tmp/arrow_cache"

export EXP_NAME="Qwen2.5-VL-7B-Instruct-seg" # TODO: change this to your own experiment name
TASK_TYPE="segmentation"
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# MAX_STEPS=1200 # TODO: change this to your own max steps


# export WANDB_DISABLED=true
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12350 \
  src/open_r1/grpo_jsonl_seg.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path "$MODEL_NAME" \
    --seg_decoder_path "$SEG_DECODER_PATH" \
    --data_file_paths "$DATA_FILE" \
    --image_folders "$IMAGE_FOLDER" \
    --arrow_cache_dir "$ARROW_CACHE" \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 10 \
    --num_train_epochs 3 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 500 \
    --max_pixels 262144 \
    --val_split_ratio 0.05 \
    --reward_funcs accuracy format \
    --finetuning_type full \
    --stage sft \
    --overwrite_output_dir true \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 2 \
    --resume_from_checkpoint null \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json

echo "Training completed for ${EXP_NAME}"
