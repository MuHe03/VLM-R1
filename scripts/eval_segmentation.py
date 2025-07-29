#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation evaluation script for vlm_seg_full_sft model (logit‑based)
===========================================================================
This version **directly consumes the model's `seg_logits` tensor** instead of
parsing an RLE string from the LLM’s textual response.  It also reads the RLE
annotation from the top‑level `rle_mask` field of the validation JSON, which is
how ReasonSeg val data is structured.

Main additions / changes
-----------------------
* **logits_to_mask()** – turns a raw *seg_logits* tensor into a binary mask **tensor**.
* **mask_to_rle()**   – now accepts a **torch.Tensor** or **np.ndarray**;
                        always converts internally to `uint8` NumPy before
                        calling `pycocotools.mask.encode`.
* **evaluate_segmentation()** –
  * keeps *pred_mask* as a tensor during resizing / post‑processing, avoiding the
    NumPy ↔ Tensor mismatch that triggered the original `unsqueeze` error;
  * converts to NumPy **only once** when computing metrics and saving RLE.

NOTE: Thresholds / post‑processing may need tuning for your dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from pycocotools import mask as maskUtils
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
import torch.nn.functional as F

# Project imports – make sure your PYTHONPATH includes ../src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.data.template import TEMPLATES
from llamafactory.extras.constants import IMAGE_PLACEHOLDER
from llamafactory.model.vlm_seg import QwenVLSegForConditionalGeneration
from llamafactory.data.mask_utils import (
            extract_rle_mask_from_content,
            rle_to_mask,
            process_segmentation_sample
        )

###############################################################################
# Helper functions
###############################################################################

def setup_distributed() -> Tuple[int, int, int]:
    """Initialise torch.distributed (NCCL); return (local_rank, world_size, rank)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
    return local_rank, dist.get_world_size(), dist.get_rank()

###############################################################################
# Mask ↔ RLE utilities
###############################################################################

def logits_to_mask(seg_logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert *seg_logits* → binary **tensor** mask (B, 1, H, W) of type `float` (0/1).

    Args:
        seg_logits: `(B, 1, H, W)` **or** `(B, H, W)` **or** `(B, C, H, W)`.
        threshold:  Sigmoid threshold for binary foreground.
    """
    if seg_logits.dim() == 4:  # (B, C, H, W)
        if seg_logits.size(1) > 1:  # multi‑class: take argmax over classes
            mask = seg_logits.softmax(dim=1).argmax(dim=1, keepdim=True).float()
        else:  # single‑channel logits → probabilities → threshold
            mask = seg_logits.sigmoid().gt(threshold).float()
    elif seg_logits.dim() == 3:  # (B, H, W)
        probs = seg_logits.sigmoid()
        mask = probs.unsqueeze(1).gt(threshold).float()
    else:
        raise ValueError(f"Unsupported seg_logits shape: {seg_logits.shape}")

    return mask  # (B, 1, H, W) float tensor containing 0/1


def mask_to_rle(mask: torch.Tensor | np.ndarray) -> Dict[str, Any]:
    """Binary mask (Tensor **or** ndarray) → COCO RLE dict with UTF‑8 `counts`."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    # bring to 2‑D H×W and uint8
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    mask = (mask > 0).astype(np.uint8)
    rle = maskUtils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

###############################################################################
# Model / tokenizer loader
###############################################################################

def load_model_and_tokenizer(model_path: str, device: str):
    model_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    print(f"Loading model from {model_path}")

    model = QwenVLSegForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=False,
        device_map={"": device},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, tokenizer, processor

###############################################################################
# Core evaluation
###############################################################################

def evaluate_segmentation(
    model_path: str,
    val_data_path: str,
    output_dir: str,
    batch_size: int = 1,
    device_map: str = "cuda:0",
):
    """Run evaluation.  Only batch_size==1 is currently supported."""
    # ---------------------------------------------------------------------
    # Distributed setup ----------------------------------------------------
    # ---------------------------------------------------------------------
    if torch.cuda.device_count() > 1:
        local_rank, world_size, rank = setup_distributed()
        device = f"cuda:{local_rank}"
    else:
        device, world_size, rank = device_map, 1, 0

    # ---------------------------------------------------------------------
    # Model + helper objects ----------------------------------------------
    # ---------------------------------------------------------------------
    model, tokenizer, processor = load_model_and_tokenizer(model_path, device)
    model.eval()

    plugin = get_mm_plugin(name="vlm_seg", image_token="<|image_pad|>", video_token="<|video_pad|>")
    seg_template = TEMPLATES["vlm_seg"]
    system_prompt = seg_template.default_system

    # ---------------------------------------------------------------------
    # Data -----------------------------------------------------------------
    # ---------------------------------------------------------------------
    with open(val_data_path, "r", encoding="utf-8") as fp:
        val_data: List[Dict[str, Any]] = json.load(fp)

    per_rank = len(val_data) // world_size
    start, end = rank * per_rank, (rank + 1) * per_rank if rank < world_size - 1 else len(val_data)
    rank_data = val_data[start:end]

    # ---------------------------------------------------------------------
    # Evaluation loop ------------------------------------------------------
    # ---------------------------------------------------------------------
    results = []
    total_giou = total_ciou = valid = 0.0

    for idx, sample in enumerate(tqdm(rank_data, desc=f"Rank {rank}")):
        user_msg = sample["messages"][0]["content"]
        img_path = sample["images"][0]

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")

        processed_sample = process_segmentation_sample(sample)

        # -----------------------------------------------------------------
        # Build model inputs ---------------------------------------------
        # -----------------------------------------------------------------
        msgs_in = deepcopy(sample["messages"])
        if IMAGE_PLACEHOLDER not in msgs_in[0]["content"]:
            msgs_in[0]["content"] = IMAGE_PLACEHOLDER + " " + msgs_in[0]["content"]

        try:
            processed = plugin.process_messages(msgs_in, [image], [], [], processor)
            prompt_ids, _ = seg_template.encode_oneturn(
                tokenizer=tokenizer,
                messages=processed,
                system=system_prompt,
                tools=None,
            )
            inputs = {
                "input_ids": torch.tensor([prompt_ids], device=device),
                "attention_mask": torch.ones(1, len(prompt_ids), device=device),
            }
            inputs.update(plugin._get_mm_inputs([image], [], [], processor))
        except Exception:
            # Fallback to simple processor call
            prompt = IMAGE_PLACEHOLDER + " " + user_msg
            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)

        # -----------------------------------------------------------------
        # Forward pass ----------------------------------------------------
        # -----------------------------------------------------------------
        with torch.no_grad():
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            out = model(**inputs)
            seg_logits = out.seg_logits  # type: ignore[attr-defined]

        gt_mask = processed_sample["mask"]  # NumPy (H, W) bool
        gt_rle = mask_to_rle(gt_mask)

        # -----------------------------------------------------------------
        # Logits → mask (tensor) & resize if needed -----------------------
        # -----------------------------------------------------------------
        pred_mask_t = logits_to_mask(seg_logits)  # (1, 1, H, W)
        if gt_mask is not None and pred_mask_t.shape[-2:] != gt_mask.shape[-2:]:
            pred_mask_t = F.interpolate(
                pred_mask_t,
                size=gt_mask.shape[-2:],
                mode="nearest",
            )

        # Convert to NumPy once for metrics / RLE
        pred_mask_np = pred_mask_t[0].cpu().numpy().astype(bool)  # (H, W)
        pred_rle = mask_to_rle(pred_mask_np)

        # -----------------------------------------------------------------
        # Metrics ---------------------------------------------------------
        # -----------------------------------------------------------------
        giou = ciou = 0.0
        if pred_mask_np.size > 0 and gt_mask.size > 0 and pred_mask_np.shape == gt_mask.shape:
            giou = compute_giou(pred_mask_np, gt_mask)
            ciou = compute_ciou(pred_mask_np, gt_mask)
            total_giou += giou
            total_ciou += ciou
            valid += 1

        results.append({
            "idx": idx,
            "image_path": img_path,
            "question": user_msg,
            "pred_rle": pred_rle,
            "gt_rle": gt_rle,
            "giou": giou,
            "ciou": ciou,
            "valid": bool(pred_mask_np.size and gt_mask.size),
        })

    # ---------------------------------------------------------------------
    # Distributed gathering -----------------------------------------------
    # ---------------------------------------------------------------------
    if world_size > 1:
        gathered: List[List[Dict[str, Any]]] = [None] * world_size  # type: ignore[assignment]
        dist.all_gather_object(gathered, results)
        if rank == 0:
            results = [item for sub in gathered if sub for item in sub]
            valid = sum(r["valid"] for r in results)
            total_giou = sum(r["giou"] for r in results)
            total_ciou = sum(r["ciou"] for r in results)
    # (else: single GPU – results already aggregated)

    if rank == 0:
        avg_giou = total_giou / valid if valid else 0.0
        avg_ciou = total_ciou / valid if valid else 0.0

        print("\nEvaluation complete")
        print(f"Total: {len(results)} | Valid: {int(valid)} | GIoU: {avg_giou:.4f} | CIoU: {avg_ciou:.4f}")

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "segmentation_results.json")
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump({
                "metrics": {
                    "total_samples": len(results),
                    "valid_predictions": int(valid),
                    "average_giou": avg_giou,
                    "average_ciou": avg_ciou,
                },
                "results": results,
            }, fp, indent=2)
        print(f"Saved: {out_path}")

###############################################################################
# CIoU / GIoU implementations (unchanged)
###############################################################################


def compute_giou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Placeholder – use your existing implementation."""
    # TODO: insert real implementation
    return 0.0


def compute_ciou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Placeholder – use your existing implementation."""
    # TODO: insert real implementation
    return 0.0

###############################################################################
# Entry point
###############################################################################

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--val_data_path", required=True)
    p.add_argument("--output_dir", default="./logs")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device_map", default="cuda:0")
    args = p.parse_args()

    evaluate_segmentation(
        model_path=args.model_path,
        val_data_path=args.val_data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
