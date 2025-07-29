#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom trainer for VLM segmentation that logs segmentation loss to wandb.
"""

import torch
from typing import Dict, Any, Optional, Union
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput

from ..extras import logging

logger = logging.get_logger(__name__)


class VLMSegmentationTrainer(Trainer):
    """Custom trainer for VLM segmentation that logs segmentation loss separately."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss and extract segmentation loss for logging.
        """
        # Forward pass
        outputs = model(**inputs)
        
        # Extract total loss
        loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None
        
        # Extract segmentation loss for logging
        seg_loss = None
        if hasattr(outputs, 'seg_loss') and outputs.seg_loss is not None:
            seg_loss = outputs.seg_loss
            
            # Log segmentation loss to wandb if available
            if self.state.is_world_process_zero and self.args.report_to and "wandb" in self.args.report_to:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "train/seg_loss": seg_loss.item(),
                            "train/step": self.state.global_step
                        })
                except ImportError:
                    logger.warning("wandb not available for logging segmentation loss")
        
        # Store seg_loss for potential use in logging callback
        if seg_loss is not None:
            self._last_seg_loss = seg_loss.item()
        
        if return_outputs:
            return (loss, outputs) if loss is not None else (torch.tensor(0.0), outputs)
        else:
            return loss if loss is not None else torch.tensor(0.0)
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Override log method to include segmentation loss.
        """
        # Add segmentation loss to logs if available
        if hasattr(self, '_last_seg_loss'):
            logs["seg_loss"] = self._last_seg_loss
        
        # Call parent log method
        super().log(logs)
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Override evaluation loop to handle segmentation loss logging during evaluation.
        """
        # Store original prediction_loss_only
        original_prediction_loss_only = prediction_loss_only
        
        # We need outputs to extract seg_loss, so set prediction_loss_only to False
        prediction_loss_only = False
        
        # Call parent evaluation loop
        eval_output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        return eval_output
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """
        Override to ensure segmentation loss is logged during training.
        """
        # Add segmentation loss to the training logs
        if hasattr(self, '_last_seg_loss'):
            # Log to console
            logger.info(f"Step {self.state.global_step}: seg_loss = {self._last_seg_loss:.6f}")
        
        # Call parent method
        super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)


def get_vlm_seg_trainer(*args, **kwargs) -> VLMSegmentationTrainer:
    """
    Factory function to create VLM segmentation trainer.
    """
    return VLMSegmentationTrainer(*args, **kwargs)
