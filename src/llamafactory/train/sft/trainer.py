# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
#[DEBUG/dyzhou]: import dice_loss from vlm_seg.py
#from open_r1.utils.loss import dice_loss
from ...model.vlm_seg import dice_loss

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


class SegmentationTrainer(CustomSeq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Debug all input keys and shapes
        logger.info(f"[DEBUG] Input keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                logger.info(f"[DEBUG] {key} shape: {value.shape}")
            else:
                logger.info(f"[DEBUG] {key} type: {type(value)}")
        
        pixel_values = inputs.get("pixel_values", None)
        if pixel_values is None:
            pixel_values = inputs.get("images", None)
        
        # Debug pixel_values shape
        if pixel_values is not None:
            logger.info(f"[DEBUG] pixel_values shape: {pixel_values.shape}")
            logger.info(f"[DEBUG] pixel_values type: {type(pixel_values)}")
            
            # Handle unexpected pixel_values shape
            if len(pixel_values.shape) != 4:
                logger.warning(f"[DEBUG] Unexpected pixel_values shape: {pixel_values.shape}, expected 4D tensor")
                # Check if we have the wrong tensor - look for the actual image tensor
                for key, value in inputs.items():
                    if hasattr(value, 'shape') and len(value.shape) == 4:
                        logger.info(f"[DEBUG] Found 4D tensor in key '{key}' with shape {value.shape}")
                        if 'image' in key.lower() or 'pixel' in key.lower():
                            logger.info(f"[DEBUG] Using {key} as pixel_values")
                            pixel_values = value
                            break
        
        masks = inputs.get("masks", None)
        class_labels = inputs.get("class_labels", None)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs.get("labels", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            masks=masks,
            class_labels=class_labels,
            labels=labels,
        )
        loss = outputs.loss
        
        # Extract and log segmentation loss for wandb
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
                        logger.info(f"Step {self.state.global_step}: seg_loss = {seg_loss.item():.6f}")
                except ImportError:
                    logger.warning("wandb not available for logging segmentation loss")
        
        # Store seg_loss for potential use in logging callback
        if seg_loss is not None:
            self._last_seg_loss = seg_loss.item()
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs, start_time=None):
        """Override log method to include segmentation loss."""
        # Add segmentation loss to logs if available
        if hasattr(self, '_last_seg_loss'):
            logs["seg_loss"] = self._last_seg_loss
        
        # Call parent log method with proper signature
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
