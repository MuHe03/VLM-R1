#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#[DONE:dyzhou:14.07] 分割验证脚本 - 支持GIoU/CIoU两种评估指标
"""
Segmentation evaluation script for vlm_seg_full_sft model
Evaluates the model on Qwen_val.json dataset with GIoU and CIoU metrics
"""

import json
import os
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from typing import List, Dict, Any, Tuple
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import LLaMA-Factory components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llamafactory.model.vlm_seg import QwenVLSegForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor
from pycocotools import mask as maskUtils


def setup_distributed():
    """Setup distributed training environment"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}")
    return local_rank, world_size, rank


def extract_rle_mask(content: str) -> Dict[str, Any]:
    """Extract RLE mask from model output"""
    pattern = r'```json\s*(.*?)\s*```'
    json_match = re.search(pattern, content, re.DOTALL)
    
    if json_match:
        try:
            json_content = json_match.group(1).strip()
            data = json.loads(json_content)
            if isinstance(data, list) and len(data) > 0:
                return data[0].get('rle_mask', {})
        except json.JSONDecodeError:
            pass
    
    return {}


#[DONE:dyzhou:14.07] 实现GIoU计算功能 - 考虑边界框包含关系
def compute_giou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute GIoU (Generalized IoU) between two binary masks"""
    # Compute IoU
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    
    # Compute bounding boxes
    def get_bbox(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return 0, 0, 0, 0
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    
    # Get bounding boxes
    rmin1, rmax1, cmin1, cmax1 = get_bbox(mask1)
    rmin2, rmax2, cmin2, cmax2 = get_bbox(mask2)
    
    # Compute area of bounding box union
    rmin_c = min(rmin1, rmin2)
    rmax_c = max(rmax1, rmax2)
    cmin_c = min(cmin1, cmin2)
    cmax_c = max(cmax1, cmax2)
    
    area_c = (rmax_c - rmin_c + 1) * (cmax_c - cmin_c + 1)
    
    # Compute GIoU
    if area_c == 0:
        return iou
    
    giou = iou - (area_c - union) / area_c
    return giou


#[DONE:dyzhou:14.07] 实现CIoU计算功能 - 综合考虑重叠、位置和形状
def compute_ciou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute CIoU (Complete IoU) between two binary masks"""
    # Compute IoU
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    
    # Compute bounding boxes
    def get_bbox(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return 0, 0, 0, 0
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    
    # Get bounding boxes
    rmin1, rmax1, cmin1, cmax1 = get_bbox(mask1)
    rmin2, rmax2, cmin2, cmax2 = get_bbox(mask2)
    
    # Compute center points
    center1_x = (cmin1 + cmax1) / 2
    center1_y = (rmin1 + rmax1) / 2
    center2_x = (cmin2 + cmax2) / 2
    center2_y = (rmin2 + rmax2) / 2
    
    # Compute distance between centers
    center_distance = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
    
    # Compute diagonal distance of bounding box union
    rmin_c = min(rmin1, rmin2)
    rmax_c = max(rmax1, rmax2)
    cmin_c = min(cmin1, cmin2)
    cmax_c = max(cmax1, cmax2)
    
    diagonal_distance = (rmax_c - rmin_c) ** 2 + (cmax_c - cmin_c) ** 2
    
    # Compute aspect ratio consistency
    def get_aspect_ratio(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return 1.0
        height = np.sum(rows)
        width = np.sum(cols)
        return width / height if height > 0 else 1.0
    
    aspect1 = get_aspect_ratio(mask1)
    aspect2 = get_aspect_ratio(mask2)
    
    # Compute aspect ratio penalty
    v = (4 / (np.pi ** 2)) * (np.arctan(aspect1) - np.arctan(aspect2)) ** 2
    
    # Compute CIoU
    if diagonal_distance == 0:
        return iou
    
    alpha = v / (1 - iou + v + 1e-6)
    ciou = iou - (center_distance / diagonal_distance) - alpha * v
    
    return ciou


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Convert RLE encoding to binary mask"""
    if 'counts' not in rle or 'size' not in rle:
        return np.array([])
    
    try:
        # Convert dict to proper RLE format for pycocotools
        rle_obj = {
            'counts': rle['counts'],
            'size': rle['size']
        }
        mask = maskUtils.decode(rle_obj)
        return mask.astype(bool)
    except Exception as e:
        print(f"Error decoding RLE: {e}")
        return np.array([])


def load_model_and_tokenizer(model_path: str, device: str):
    """Load the segmentation model and tokenizer"""
    print(f"Loading model from {model_path}")
    
    # Load model
    model = QwenVLSegForConditionalGeneration.from_pretrained(
        model_path,
        
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        device_map={"": device},
    )
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, tokenizer, processor


def evaluate_segmentation(
    model_path: str,
    val_data_path: str,
    output_dir: str,
    batch_size: int = 1,
    device_map: str = "cuda:0"
):
    """Evaluate segmentation model on validation data"""
    
    # Setup distributed if needed
    if torch.cuda.device_count() > 1:
        local_rank, world_size, rank = setup_distributed()
        device = f"cuda:{local_rank}"
    else:
        device = device_map
        rank = 0
        world_size = 1
    
    # Load model
    model, tokenizer, processor = load_model_and_tokenizer(model_path, device)
    model.eval()
    
    # Load validation data
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # Split data for distributed evaluation
    per_rank_data = len(val_data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(val_data)
    rank_data = val_data[start_idx:end_idx]
    
    results = []
    total_giou = 0.0
    total_ciou = 0.0
    valid_predictions = 0
    
    print(f"Evaluating {len(rank_data)} samples on rank {rank}")
    
    for idx, sample in enumerate(tqdm(rank_data, desc=f"Rank {rank}")):
        # try:
            # Extract user question and ground truth
            user_content = sample['messages'][0]['content']
            assistant_content = sample['messages'][1]['content']
            image_path = sample['images'][0]
            
            # Extract metadata if available
            metadata = sample.get('metadata', {})
            
            # Load and process image
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                print(f"Image not found: {image_path}")
                continue
            
            # Prepare input
            messages = [
                {"role": "user", "content": user_content}
            ]
            
            # Generate response
            with torch.no_grad():
                inputs = processor(
                    messages=messages,
                    images=image,
                    return_tensors="pt"
                ).to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract predicted mask
            pred_rle = extract_rle_mask(response)
            gt_rle = extract_rle_mask(assistant_content)
            
            # Convert to masks
            pred_mask = rle_to_mask(pred_rle)
            gt_mask = rle_to_mask(gt_rle)
            
            #[DONE:dyzhou:14.07] 计算GIoU和CIoU评估指标
            # Compute metrics
            giou_score = 0.0
            ciou_score = 0.0
            if pred_mask.size > 0 and gt_mask.size > 0:
                if pred_mask.shape == gt_mask.shape:
                    giou_score = compute_giou(pred_mask, gt_mask)
                    ciou_score = compute_ciou(pred_mask, gt_mask)
                    total_giou += giou_score
                    total_ciou += ciou_score
                    valid_predictions += 1
                else:
                    print(f"Mask shape mismatch: pred {pred_mask.shape}, gt {gt_mask.shape}")
            
            #[DONE:dyzhou:14.07] 存储包含GIoU和CIoU指标的结果
            # Store result
            result = {
                'idx': idx,
                'image_path': image_path,
                'question': user_content,
                'ground_truth': assistant_content,
                'prediction': response,
                'pred_rle': pred_rle,
                'gt_rle': gt_rle,
                'giou': giou_score,
                'ciou': ciou_score,
                'valid': pred_mask.size > 0 and gt_mask.size > 0
            }
            results.append(result)
            
        # except Exception as e:
        #     print(f"Error processing sample {idx}: {e}")
        #     continue
    
    # Gather results from all processes
    if world_size > 1:
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, results)
        
        if rank == 0:
            # Combine results from all processes
            all_results = []
            for r in gathered_results:
                if r is not None:
                    all_results.extend(r)
            
            # Calculate overall metrics
            total_giou = sum(r['giou'] for r in all_results if r['valid'])
            total_ciou = sum(r['ciou'] for r in all_results if r['valid'])
            valid_predictions = sum(1 for r in all_results if r['valid'])
            avg_giou = total_giou / valid_predictions if valid_predictions > 0 else 0.0
            avg_ciou = total_ciou / valid_predictions if valid_predictions > 0 else 0.0
            
            print(f"\nOverall Results:")
            print(f"Total samples: {len(all_results)}")
            print(f"Valid predictions: {valid_predictions}")
            print(f"Average GIoU: {avg_giou:.4f}")
            print(f"Average CIoU: {avg_ciou:.4f}")
            
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "segmentation_results.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metrics': {
                        'total_samples': len(all_results),
                        'valid_predictions': valid_predictions,
                        'average_giou': avg_giou,
                        'average_ciou': avg_ciou
                    },
                    'results': all_results
                }, f, indent=2)
            
            print(f"Results saved to {output_path}")
    else:
        # Single process evaluation
        avg_giou = total_giou / valid_predictions if valid_predictions > 0 else 0.0
        avg_ciou = total_ciou / valid_predictions if valid_predictions > 0 else 0.0
        
        print(f"\nResults:")
        print(f"Total samples: {len(results)}")
        print(f"Valid predictions: {valid_predictions}")
        print(f"Average GIoU: {avg_giou:.4f}")
        print(f"Average CIoU: {avg_ciou:.4f}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "segmentation_results.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': {
                    'total_samples': len(results),
                    'valid_predictions': valid_predictions,
                    'average_giou': avg_giou,
                    'average_ciou': avg_ciou
                },
                'results': results
            }, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--val_data_path", type=str, required=True,
                       help="Path to Qwen_val.json")
    parser.add_argument("--output_dir", type=str, default="./logs",
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--device_map", type=str, default="cuda:0",
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    evaluate_segmentation(
        model_path=args.model_path,
        val_data_path=args.val_data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_map=args.device_map
    )


if __name__ == "__main__":
    main() 