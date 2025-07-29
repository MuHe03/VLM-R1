#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for handling RLE mask format in segmentation datasets.
"""

import json
import re
import numpy as np
import torch
from typing import Dict, Any, Optional, List
from pycocotools import mask as maskUtils


def extract_rle_mask_from_content(content: str) -> Optional[Dict[str, Any]]:
    """Extract RLE mask from assistant response content.
    
    Args:
        content: Assistant response content containing RLE mask in JSON format
        
    Returns:
        RLE mask dictionary or None if not found
    """
    # Pattern to match JSON code blocks
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
    
    return None


def rle_to_mask(rle: Dict[str, Any]) -> Optional[np.ndarray]:
    """Convert RLE encoding to binary mask.
    
    Args:
        rle: RLE mask dictionary with 'counts' and 'size' fields
        
    Returns:
        Binary mask as numpy array or None if conversion fails
    """
    if not rle or 'counts' not in rle or 'size' not in rle:
        return None
    
    try:
        # Use pycocotools to decode RLE
        mask = maskUtils.decode(rle)
        return mask.astype(np.uint8)
    except Exception:
        return None


def mask_to_tensor(mask: np.ndarray, target_size: Optional[tuple] = None) -> torch.Tensor:
    """Convert numpy mask to torch tensor with optional resizing.
    
    Args:
        mask: Binary mask as numpy array
        target_size: Optional target size (height, width) for resizing
        
    Returns:
        Mask as torch tensor
    """
    mask_tensor = torch.from_numpy(mask).float()
    
    if target_size is not None and mask_tensor.shape != target_size:
        import torch.nn.functional as F
        # Add batch and channel dimensions for interpolation
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(
            mask_tensor, 
            size=target_size, 
            mode='nearest'
        )
        # Remove batch and channel dimensions
        mask_tensor = mask_tensor.squeeze(0).squeeze(0)
    
    return mask_tensor


def process_segmentation_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single segmentation sample from the dataset.
    
    Args:
        sample: Dataset sample containing messages, images, and rle_mask
        
    Returns:
        Processed sample with extracted mask tensor
    """
    processed_sample = sample.copy()
    
    # Extract RLE mask from the sample
    if 'rle_mask' in sample and sample['rle_mask']:
        rle_masks = sample['rle_mask']
        if isinstance(rle_masks, list) and len(rle_masks) > 0:
            # Take the first RLE mask (assuming single mask per sample)
            rle_data = rle_masks[0]
            
            # The RLE data is a JSON string containing the mask
            if isinstance(rle_data, str):
                # Extract RLE mask from JSON string
                rle_mask = extract_rle_mask_from_content(rle_data)
                if rle_mask:
                    # Convert RLE to binary mask
                    mask = rle_to_mask(rle_mask)
                    if mask is not None:
                        processed_sample['mask'] = mask
            elif isinstance(rle_data, dict):
                # If it's already a dict, try to get the RLE mask directly
                rle_mask = rle_data.get('rle_mask', rle_data)
                if rle_mask:
                    mask = rle_to_mask(rle_mask)
                    if mask is not None:
                        processed_sample['mask'] = mask
    
    return processed_sample


def batch_masks(masks: List[np.ndarray], target_size: Optional[tuple] = None) -> torch.Tensor:
    """Batch multiple masks into a single tensor.
    
    Args:
        masks: List of binary masks as numpy arrays
        target_size: Optional target size (height, width) for resizing
        
    Returns:
        Batched masks as torch tensor with shape (batch_size, height, width)
    """
    if not masks:
        return torch.empty(0)
    
    # Convert all masks to tensors with consistent size
    mask_tensors = []
    for mask in masks:
        mask_tensor = mask_to_tensor(mask, target_size)
        mask_tensors.append(mask_tensor)
    
    # Stack into batch
    return torch.stack(mask_tensors, dim=0)
