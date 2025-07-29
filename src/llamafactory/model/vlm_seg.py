from transformers import Qwen2_5_VLForConditionalGeneration,  Mask2FormerForUniversalSegmentation, Mask2FormerConfig
# [DEBUG/dyzhou]: import ModelOutput from transformers.utils
from dataclasses import dataclass
from transformers.utils import logging, ModelOutput
from torch import nn
import torch
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _pair
from transformers.models.mask2former.modeling_mask2former import \
     Mask2FormerSinePositionEmbedding
# [DEBUG/dyzhou]:
#from open_r1.utils.loss import dice_loss

logger = logging.get_logger(__name__)

def dice_loss(pred_logits, target_masks, smooth=1e-6):
    """
    Compute dice loss for query-based segmentation (Mask2Former style).
    
    Args:
        pred_logits: [B, Q, H, W] - Q queries predicting masks
        target_masks: [B, H, W] - Ground truth binary masks
        smooth: Smoothing factor
    """
    
    B, Q, H, W = pred_logits.shape
    
    # Apply sigmoid to get probabilities for each query
    pred_probs = torch.sigmoid(pred_logits)  # [B, Q, H, W]
    
    # Expand target masks to match queries: [B, H, W] -> [B, Q, H, W]
    target_expanded = target_masks.unsqueeze(1).expand(-1, Q, -1, -1)  # [B, Q, H, W]
    
    # Flatten spatial dimensions
    pred_flat = pred_probs.view(B, Q, -1)  # [B, Q, H*W]
    target_flat = target_expanded.view(B, Q, -1).float()  # [B, Q, H*W]
    
    # Compute dice for each query
    intersection = (pred_flat * target_flat).sum(dim=2)  # [B, Q]
    pred_sum = pred_flat.sum(dim=2)  # [B, Q]
    target_sum = target_flat.sum(dim=2)  # [B, Q]
    
    dice_per_query = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)  # [B, Q]
    
    # Take the best dice score across queries for each batch item
    best_dice, _ = dice_per_query.max(dim=1)  # [B]
    
    # Return dice loss (1 - best dice coefficient)
    return 1 - best_dice.mean()

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """Focal loss for class imbalance"""
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss
    
@dataclass
class QwenVLSeg(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None        # language
    seg_logits: torch.FloatTensor = None    # H×W or queries
    class_logits: torch.FloatTensor = None  # class predictions
    vision_hidden_states: torch.FloatTensor = None
    seg_loss: torch.FloatTensor = None      # segmentation loss for wandb logging
    class_loss: torch.FloatTensor = None    # class loss for potential logging

class QwenVLSegForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, seg_decoder_path=None):
        print(config._name_or_path)
        super().__init__(config)
        
        # Set model type to vlm_seg for proper recognition
        config.model_type = "vlm_seg"
        # Prepare segmentation config
        if seg_decoder_path is None:  # from scratch – build with values in `config`
            seg_decoder_path = "facebook/mask2former-swin-large-ade-semantic"
        seg_cfg = Mask2FormerConfig.from_pretrained(seg_decoder_path)
        # If the training YAML specified different values, honour them
        if hasattr(config, "num_classes") and config.num_classes is not None:
            seg_cfg.num_labels = config.num_classes
        if hasattr(config, "seg_hidden_dim") and config.seg_hidden_dim is not None:
            seg_cfg.hidden_dim = config.seg_hidden_dim


        full_model = Mask2FormerForUniversalSegmentation(seg_cfg)
        self.seg_config = full_model.config
        self.seg_decoder = full_model.model.transformer_module.decoder
        self.class_predictor = full_model.class_predictor

        hid = self.seg_config.hidden_dim
        lm_dim = config.hidden_size 

        self.pyr_convs = nn.ModuleList([
            nn.Identity(),                         # keep stride 16 as is
            nn.Sequential(
                nn.ConvTranspose2d(hid, hid, 2, 2),  # ×2 → stride 8
                nn.GroupNorm(32, hid)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(hid, hid, 2, 2),  # ×2 → stride 4
                nn.GroupNorm(32, hid)
            ),
        ])

        self.pixel_decoder = full_model.model.pixel_level_module.decoder
        for p in self.pixel_decoder.parameters():  # freeze
            p.requires_grad_(False)
        
        self.high_proj = nn.Conv2d(hid, 192, 1, bias=False)
            
        level_inchs = [proj[0].in_channels
                        for proj in self.pixel_decoder.input_projections][::-1]
        self.level_adapters = nn.ModuleList(
            [nn.Conv2d(hid, cin, 1) for cin in level_inchs]
        )
        
        self.seg_token_id = None
        self._init_seg_token()
        
        self.vis_proj   = nn.Linear(lm_dim, hid, bias=False)   # image patches
        self.query_proj = nn.Linear(lm_dim, hid, bias=False)   # [SEG] tokens

        self.query_feat   = nn.Embedding(self.seg_config.num_queries, hid)
        self.query_pos    = nn.Embedding(self.seg_config.num_queries, hid)
        
        self.pos_embedder = Mask2FormerSinePositionEmbedding(
                                num_pos_feats=hid // 2, normalize=True)
        
    def _ensure_query_embeddings(self, hidden_dim):
        """
        Guarantee that `self.seg_decoder` exposes a learnable query tensor.
        Falls back to the parent Mask2Former module or creates a fresh one.
        """
        for name in ("query_embed", "query_feat", "queries_embed"):
            if hasattr(self.seg_decoder, name):
                self.query_source = getattr(self.seg_decoder, name)
                return

        if hasattr(self, "query_embed"):
            self.query_source = self.query_embed 
            return
        
        num_q = getattr(self.seg_config, "num_queries", 100)
        self.query_source = nn.Embedding(num_q, hidden_dim)
        self.register_parameter("vlm_seg_queries", self.query_source.weight)
        
    def _init_seg_token(self):
        """Initialize the SEG token, adding it to the tokenizer and resizing model embeddings if necessary."""
        if hasattr(self, 'get_tokenizer'):
            tokenizer = self.get_tokenizer()
            if tokenizer is not None:
                seg_token = '[SEG]'
                if seg_token not in tokenizer.vocab:
                    logger.info(f"Adding '{seg_token}' to tokenizer vocabulary.")
                    tokenizer.add_special_tokens({'additional_special_tokens': [seg_token]})
                    new_vocab_size = len(tokenizer)
                    self.resize_token_embeddings(new_vocab_size)
                    self.config.vocab_size = new_vocab_size

                self.seg_token_id = tokenizer.convert_tokens_to_ids(seg_token)
                if self.seg_token_id == tokenizer.unk_token_id:
                    logger.info(f"'{seg_token}' token not properly added or found.")
                else:
                    logger.info(f"'{seg_token}' token ID: {self.seg_token_id}")
    
    def get_tokenizer(self):
        """Get the tokenizer associated with this model."""
        try:
            # Try to get tokenizer from the model's config
            from transformers import AutoTokenizer
            model_name = self.config._name_or_path
            return AutoTokenizer.from_pretrained(model_name)
        except:
            logger.info(f"No SEG token Found!")
            return None

    def forward(self, pixel_values=None, input_ids=None, masks=None, class_labels=None, compute_loss: bool | None = None, **kwargs):
        return_dict_flag = kwargs.pop("return_dict", True)
        output_hidden_states_flag = kwargs.pop("output_hidden_states", True)
        base_out = super().forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            output_hidden_states=output_hidden_states_flag,
            return_dict=return_dict_flag,
            **kwargs,
        )
        # print(input_ids)
        
        img_tok = self.config.image_token_id
        lm_dim = self.config.hidden_size 
        hid = self.seg_config.hidden_dim
        vis_mask = (input_ids == img_tok)
        if not vis_mask.any():
            return base_out

        last_hidden = base_out.hidden_states[-1]
        B, L, D = last_hidden.shape
        
        # print("vis_mask: ", vis_mask)

        vis_tokens = last_hidden[vis_mask].view(B, -1, lm_dim)
        vis_tokens = self.vis_proj(vis_tokens)

        num_patches = vis_tokens.size(1)
        # print("num_patches: ", num_patches)
        H = int(round(num_patches ** 0.5))
        while H > 0 and num_patches % H != 0:
            H -= 1
        W = num_patches // H
        
        feats = vis_tokens.transpose(1,2).reshape(B, hid, H, W)

        high_feat = self.high_proj(feats)
        feats_lvls = []
        feats_lvls.append(self.pyr_convs[0](feats))
        feats_lvls.append(self.pyr_convs[1](feats_lvls[0]))
        feats_lvls.append(self.pyr_convs[2](feats_lvls[1]))

        feats_pd = [self.level_adapters[i](feat)
                    for i, feat in enumerate(feats_lvls)]
        feats_pd_full = [high_feat] + feats_pd
        
        
        pd_out = self.pixel_decoder(feats_pd_full)
        mask_features       = pd_out.mask_features
        multi_scale_feats   = pd_out.multi_scale_features 
        feature_size_list   = [feat.shape[-2:] for feat in multi_scale_feats]
        
        multi_stage_pos = [
            self.pos_embedder(f).flatten(2).permute(2, 0, 1)
            for f in multi_scale_feats
        ]

        if self.seg_token_id is not None and (input_ids == self.seg_token_id).any():
            obj_q = last_hidden[(input_ids == self.seg_token_id)].view(B, -1, lm_dim)
            obj_q = self.query_proj(obj_q)
            query_pos = None
            logger.info(f"Using SEG token for query projection.")
        else:
            obj_q = self.query_feat.weight.unsqueeze(0).expand(B, -1, hid)
            query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, B, 1)

        obj_q = obj_q.transpose(0, 1)

        dec_out = self.seg_decoder(
            inputs_embeds=obj_q,
            multi_stage_positional_embeddings=multi_stage_pos,
            pixel_embeddings=mask_features,
            encoder_hidden_states=[
                f.flatten(2).permute(2, 0, 1)
                for f in multi_scale_feats
            ],
            query_position_embeddings=query_pos,
            feature_size_list=feature_size_list,
            return_dict=True,
        )

        class_logits = self.class_predictor(dec_out.last_hidden_state)
        mask_embeds  = self.seg_decoder.mask_predictor.mask_embedder(dec_out.last_hidden_state)
        seg_logits   = torch.einsum("bqc,bchw->bqhw", mask_embeds, mask_features)
        
        # --- DEBUG: log VLM outputs per sample ---
        tokenizer = self.get_tokenizer() if hasattr(self, "get_tokenizer") else None
        for idx in range(base_out.logits.size(0)):
            txt_logits_sample = base_out.logits[idx]
            seg_logits_sample = seg_logits[idx]
            if tokenizer is not None:
                pred_ids = txt_logits_sample.argmax(dim=-1)
                text_out = tokenizer.decode(pred_ids, skip_special_tokens=False)
            else:
                text_out = "<tokenizer unavailable>"
            text_in = tokenizer.decode(input_ids[idx], skip_special_tokens=False)
            # Debug mask and class label shapes
            mask_shape_str = f"masks[{idx}].shape = {masks[idx].shape}" if masks is not None else "masks = None"
            class_shape_str = f"class_labels[{idx}].shape = {class_labels[idx].shape}" if class_labels is not None else "class_labels = None"
            
            logger.info(
                f"[DEBUG] Sample {idx}:\n"
                f"          text_in:  {text_in}\n"
                f"          text_out: {text_out}\n"
                f"          text logits shape: {txt_logits_sample.shape}\n"
                f"          seg logits shape: {seg_logits_sample.shape}\n"
                f"          class logits shape: {class_logits[idx].shape}\n"
                f"          {mask_shape_str}\n"
                f"          {class_shape_str}"
            )

        if pixel_values is not None and seg_logits.shape[-2:] != pixel_values.shape[-2:]:
            seg_logits = F.interpolate(seg_logits, size=pixel_values.shape[-2:],
                                        mode="bilinear", align_corners=False)
        
        # Resize target masks to match segmentation logits resolution if needed
        if masks is not None and seg_logits.shape[-2:] != masks.shape[-2:]:
            # logger.info(f"Resizing target masks from {masks.shape[-2:]} to {seg_logits.shape[-2:]}")
            masks = F.interpolate(masks.unsqueeze(1), size=seg_logits.shape[-2:], 
                                mode="nearest").squeeze(1)

        compute_loss = self.training if compute_loss is None else compute_loss

        # Compute losses separately for logging
        seg_loss = None
        class_loss = None
        
        if compute_loss:
            total_loss = base_out.loss or torch.tensor(0.0, device=seg_logits.device)
            
            # Compute segmentation loss separately
            if masks is not None:
                seg_loss = dice_loss(seg_logits, masks)
                total_loss += seg_loss
            
            # Compute class loss separately
            if class_labels is not None:
                flat_logits = class_logits.view(-1, class_logits.size(-1))
                flat_labels = class_labels.view(-1)
                valid = flat_labels >= 0
                if valid.any():
                    class_loss = focal_loss(flat_logits[valid], flat_labels[valid])
                    total_loss += class_loss
        else:
            total_loss = None

        return QwenVLSeg(
            loss=total_loss,
            logits=base_out.logits,
            seg_logits=seg_logits,
            class_logits=class_logits,
            vision_hidden_states=vis_tokens,
            seg_loss=seg_loss,  # Add segmentation loss for wandb logging
            class_loss=class_loss,  # Add class loss for potential logging
        )

def build_vlm_seg_model(config, *args, **kwargs):
    """Utility to build the VLM-Seg model with optional modes.

    Keyword Args:
        continue_training (bool): If True, load the full model from the
            checkpoint pointed to by ``config._name_or_path`` without
            initializing a fresh segmentation decoder.  This is useful when
            resuming training from a previously saved distributed checkpoint.
        evaluation (bool): If True, the returned model will be put into
            ``eval()`` mode immediately.
    """
    continue_training = kwargs.pop("continue_training", False)
    evaluation = kwargs.pop("evaluation", False)

    if continue_training:
        # Do NOT override `config` so that the checkpoint's own hyper-params
        # (e.g. num_classes, hidden_dim) are kept intact.
        model, load_info = QwenVLSegForConditionalGeneration.from_pretrained(
            config._name_or_path,
            ignore_mismatched_sizes=False,   # strict loading
            output_loading_info=True,
            *args,
            **kwargs,
        )
        miss = load_info.get("missing_keys", [])
        unexp = load_info.get("unexpected_keys", [])
        logger.info(
            f"[VLM-Seg] Loaded checkpoint with {len(miss)} missing and {len(unexp)} unexpected keys."
        )
    else:
        seg_decoder_path = getattr(config, "seg_decoder_path", None)
        if seg_decoder_path is None and "seg_decoder_path" in kwargs:
            seg_decoder_path = kwargs.pop("seg_decoder_path")
        if seg_decoder_path is None:
            seg_decoder_path = "facebook/mask2former-swin-large-ade-semantic"

        model = QwenVLSegForConditionalGeneration.from_pretrained(
            config._name_or_path,
            config=config,
            seg_decoder_path=seg_decoder_path,
            ignore_mismatched_sizes=True,
            *args,
            **kwargs,
        )

    if evaluation:
        model.eval()
    return model