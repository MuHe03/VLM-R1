#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data collator for segmentation tasks with RLE mask support.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq

from .collator import SFTDataCollatorWith4DAttentionMask
from .mask_utils import extract_rle_mask_from_content, rle_to_mask

logger = logging.getLogger(__name__)


@dataclass
class SegmentationDataCollator(SFTDataCollatorWith4DAttentionMask):
    """
    Collator that adds four‑dimensional guarantees for `pixel_values`
    and batches optional RLE‑encoded segmentation masks.
    """

    def _ensure_4d(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Make sure a ``pixel_values`` tensor is (B, C, H, W).
        """
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            if tensor.shape[0] in (1, 3):
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.unsqueeze(1)
        return tensor

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("[DEBUG] SegmentationDataCollator called with %d features",
                    len(features))

        cleaned, rle_masks_per_sample = [], []
        for idx, feat in enumerate(features):
            logger.info("[DEBUG] Feature %d keys: %s", idx, list(feat.keys()))

            cleaned.append({k: v for k, v in feat.items() if k != "rle_mask"})
            rle_masks_per_sample.append(feat.get("rle_mask", []))

        batch = super().__call__(cleaned)

        if "pixel_values" in batch:
            batch["pixel_values"] = self._ensure_4d(batch["pixel_values"])

        decoded_masks = []
        has_any_mask = False
        for rle_list in rle_masks_per_sample:
            mask_tensor = None
            if rle_list:
                rle_entry = rle_list[0]
                if isinstance(rle_entry, str):
                    rle_dict = extract_rle_mask_from_content(rle_entry)
                elif isinstance(rle_entry, dict):
                    rle_dict = rle_entry.get("rle_mask", rle_entry)
                else:
                    rle_dict = None

                if rle_dict:
                    mask_np = rle_to_mask(rle_dict)
                    mask_tensor = torch.from_numpy(mask_np).float()
                    has_any_mask = True
            decoded_masks.append(mask_tensor)

        if has_any_mask:
            # Make all masks the same size and stack along batch dim.
            target_h, target_w = next(
                m.shape for m in decoded_masks if m is not None
            )
            padded = []
            for m in decoded_masks:
                if m is None:
                    padded.append(torch.zeros((target_h, target_w)))
                elif m.shape != (target_h, target_w):
                    m = torch.nn.functional.interpolate(
                        m.unsqueeze(0).unsqueeze(0),
                        size=(target_h, target_w),
                        mode="nearest",
                    ).squeeze(0).squeeze(0)
                    padded.append(m)
                else:
                    padded.append(m)
            batch["masks"] = torch.stack(padded)  # (B, H, W)

        return batch
