import torch
import torch.nn.functional as F


def dice_loss(pred_logits, target_masks, smooth=1e-6):
    """
    Compute dice loss for segmentation.
    
    Args:
        pred_logits: Predicted logits from segmentation model
        target_masks: Ground truth masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice loss value
    """
    # Apply softmax to get probabilities
    pred_probs = F.softmax(pred_logits, dim=1)
    
    # Flatten tensors
    pred_flat = pred_probs.view(-1)
    target_flat = target_masks.view(-1)
    
    # Calculate dice coefficient
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    # Return dice loss (1 - dice coefficient)
    return 1 - dice_coeff
