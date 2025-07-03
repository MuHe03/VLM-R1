from transformers import Qwen2_5_VLForConditionalGeneration, ModelOutput, Mask2FormerForUniversalSegmentation
from transformers.utils import logging
from torch import nn
import torch
import torch.nn.functional as F
from open_r1.utils.loss import dice_loss

logger = logging.get_logger(__name__)

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

class QwenVLSeg(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None        # language
    seg_logits: torch.FloatTensor = None    # H×W or queries
    class_logits: torch.FloatTensor = None  # class predictions
    vision_hidden_states: torch.FloatTensor = None

class QwenVLSegForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, seg_decoder_path="facebook/mask2former-swin-base"):
        super().__init__(config)
        full_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            seg_decoder_path, ignore_mismatched_sizes=True
        )
        # Extract only the decoder components
        self.seg_decoder = full_model.model.transformer_decoder
        self.class_predictor = full_model.model.class_predictor
        self.mask_embedder = full_model.model.mask_embedder
        self.query_embed = full_model.model.query_embed
        self.input_proj = full_model.model.input_proj
        
        self.seg_config = full_model.config
        
        vlm_hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else config.text_config.hidden_size
        mask2former_hidden_size = self.seg_config.hidden_dim
        
        if vlm_hidden_size != mask2former_hidden_size:
            self.feature_adapter = nn.Linear(vlm_hidden_size, mask2former_hidden_size)
        else:
            self.feature_adapter = nn.Identity()

        # Add feature upsampler to match the expected spatial dimensions
        self.feature_upsampler = nn.Identity()  # For now, use identity. Can be replaced with upsampling layers if needed
        
        # Initialize SEG token ID
        self.seg_token_id = None
        self._init_seg_token()

    def _init_seg_token(self):
        """Initialize the SEG token in the tokenizer vocabulary."""
        if hasattr(self, 'get_tokenizer'):
            tokenizer = self.get_tokenizer()
            if tokenizer is not None:
                self.seg_token_id = tokenizer.convert_tokens_to_ids('<SEG>')
                logger.info(f"SEG token ID: {self.seg_token_id}")
    
    def get_tokenizer(self):
        """Get the tokenizer associated with this model."""
        try:
            # Try to get tokenizer from the model's config
            from transformers import AutoTokenizer
            model_name = self.config._name_or_path
            return AutoTokenizer.from_pretrained(model_name)
        except:
            return None

    def forward(self, pixel_values=None, input_ids=None, masks=None, class_labels=None, **kwargs):
        base_out = super().forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        vision_tokens = base_out.vision_hidden_states[-1]

        B, N, D = vision_tokens.shape
        
        if pixel_values is not None:
            img_h, img_w = pixel_values.shape[-2:]
            patch_size = 14 # according to Qwen2.5-VL paper, patch size is 14x14
            h, w = img_h // patch_size, img_w // patch_size
        else:
            h = w = int(N ** 0.5)
        
        adapted_tokens = self.feature_adapter(vision_tokens)
        feats = adapted_tokens.transpose(1, 2).reshape(B, -1, h, w)
        
        upsampled_feats = self.feature_upsampler(feats)
        
        projected_feats = self.input_proj(upsampled_feats)
        
        if masks is not None:
            target_size = upsampled_feats.shape[-2:]
            if masks.shape[-2:] != target_size:
                masks = F.interpolate(masks.float(), size=target_size, mode='nearest').long()

        # Use SEG token ID from model or kwargs
        seg_token_id = kwargs.get('seg_token_id', self.seg_token_id)
        if seg_token_id is not None and input_ids is not None:
            # find seg token
            seg_positions = (input_ids == seg_token_id).nonzero(as_tuple=True)
            if len(seg_positions[0]) > 0:
                
                text_hidden_states = base_out.hidden_states[-1]
                batch_indices, token_indices = seg_positions
                object_queries = text_hidden_states[batch_indices, token_indices]
                
                
                object_queries_list = []
                for b in range(B):
                    batch_mask = batch_indices == b
                    if batch_mask.any():
                        batch_queries = object_queries[batch_mask]
                        object_queries_list.append(batch_queries)
                    else:
                        object_queries_list.append(self.query_embed.weight[:1])
                        
                
                max_queries = max(q.shape[0] for q in object_queries_list)
                padded_queries = []
                for queries in object_queries_list:
                    if queries.shape[0] < max_queries:
                        padding = self.query_embed.weight[:max_queries - queries.shape[0]]
                        queries = torch.cat([queries, padding], dim=0)
                    padded_queries.append(queries)
                
                object_queries = torch.stack(padded_queries, dim=0)
            else:
                object_queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        else:
            object_queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        
        decoder_outputs = self.seg_decoder(
            inputs_embeds=object_queries,
            encoder_hidden_states=projected_feats.flatten(2).transpose(1, 2),
            return_dict=True
        )
        
        class_predictions = self.class_predictor(decoder_outputs.last_hidden_state)
        mask_embeddings = self.mask_embedder(decoder_outputs.last_hidden_state)
        
        seg_logits = torch.einsum("bqc,bchw->bqhw", mask_embeddings, projected_feats)
        
        if pixel_values is not None:
            original_size = pixel_values.shape[-2:]
            if seg_logits.shape[-2:] != original_size:
                seg_logits = F.interpolate(seg_logits, size=original_size, mode='bilinear', align_corners=False)

        # llm loss
        total_loss = base_out.loss if base_out.loss is not None else torch.tensor(0.0, device=seg_logits.device)
        
        # seg loss
        if masks is not None:
            seg_loss = dice_loss(seg_logits, masks)
            total_loss = total_loss + seg_loss
        
        # class loss
        if class_labels is not None:
            class_logits_flat = class_predictions.view(-1, class_predictions.size(-1))
            class_labels_flat = class_labels.view(-1)
            
            valid_mask = class_labels_flat >= 0
            if valid_mask.any():
                class_loss = focal_loss(
                    class_logits_flat[valid_mask], 
                    class_labels_flat[valid_mask]
                )
                total_loss = total_loss + class_loss

        return QwenVLSeg(
            loss=total_loss,
            logits=base_out.logits,
            seg_logits=seg_logits,
            class_logits=class_predictions,
            vision_hidden_states=base_out.vision_hidden_states,
        )
