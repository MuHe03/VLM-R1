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

class QwenVLSegForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, seg_decoder_path=None):
        print(config._name_or_path)
        super().__init__(config)
        if seg_decoder_path is None:            # from‑scratch
            seg_cfg = Mask2FormerConfig()       # uses default 100 queries & 1 class
        else:                                   # reuse published hyper‑params only
            seg_cfg = Mask2FormerConfig.from_pretrained(seg_decoder_path)

        full_model = Mask2FormerForUniversalSegmentation(seg_cfg)
        self.seg_config = full_model.config
        self.seg_decoder = full_model.model.transformer_module.decoder
        self.class_predictor = full_model.class_predictor

        hid = self.seg_config.hidden_dim
        lm_dim = config.hidden_size 

        self.pyr_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hid, hid, 1), nn.GroupNorm(32, hid)
            ),
            nn.Sequential(
                nn.Conv2d(hid, hid, 3, stride=2, padding=1, groups=hid),
                nn.Conv2d(hid, hid, 1), nn.GroupNorm(32, hid)
            ),
            nn.Sequential(
                nn.Conv2d(hid, hid, 3, stride=2, padding=1, groups=hid),
                nn.Conv2d(hid, hid, 1), nn.GroupNorm(32, hid)
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

    def forward(self, pixel_values=None, input_ids=None, masks=None, class_labels=None, **kwargs):
        base_out = super().forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        
        img_tok = self.config.image_token_id
        lm_dim = self.config.hidden_size 
        hid = self.seg_config.hidden_dim
        vis_mask = (input_ids == img_tok)
        last_hidden = base_out.hidden_states[-1]
        B, L, D = last_hidden.shape
        
        vis_tokens = last_hidden[vis_mask].view(B, -1, lm_dim)
        vis_tokens = self.vis_proj(vis_tokens)

        num_patches = vis_tokens.size(1)
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
            logger.info(
                f"[DEBUG] Sample {idx}:"
                f"           text logits shape {txt_logits_sample.shape}, "
                f"seg logits shape {seg_logits_sample.shape}, class logits shape {class_logits[idx].shape}"
            )

        if seg_logits.shape[-2:] != pixel_values.shape[-2:]:
            seg_logits = F.interpolate(seg_logits, size=pixel_values.shape[-2:],
                                        mode="bilinear", align_corners=False)

        total_loss = base_out.loss or torch.tensor(0.0, device=seg_logits.device)
        if masks is not None:
            total_loss += dice_loss(seg_logits, masks)
        if class_labels is not None:
            flat_logits = class_logits.view(-1, class_logits.size(-1))
            flat_labels = class_labels.view(-1)
            valid = flat_labels >= 0
            if valid.any():
                total_loss += focal_loss(flat_logits[valid], flat_labels[valid])

        return QwenVLSeg(
            loss=total_loss,
            logits=base_out.logits,
            seg_logits=seg_logits,
            class_logits=class_logits,
            vision_hidden_states=vis_tokens,
        )

def build_vlm_seg_model(config, *args, **kwargs):
    seg_decoder_path = getattr(config, "seg_decoder_path", None)
    if seg_decoder_path is None and "seg_decoder_path" in kwargs:
        seg_decoder_path = kwargs.pop("seg_decoder_path")
    if seg_decoder_path is None:
        seg_decoder_path = "facebook/mask2former-swin-large-ade-semantic"

    print(seg_decoder_path)
    return QwenVLSegForConditionalGeneration.from_pretrained(
        config._name_or_path,
        config=config,
        seg_decoder_path=seg_decoder_path,
        ignore_mismatched_sizes=True,
        *args, **kwargs
    )