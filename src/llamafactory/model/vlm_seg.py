from open_r1.models.vlm_seg import QwenVLSegForConditionalGeneration

def build_vlm_seg_model(config, *args, **kwargs):
    seg_decoder_path = getattr(config, "seg_decoder_path", None)
    if seg_decoder_path is None and "seg_decoder_path" in kwargs:
        seg_decoder_path = kwargs.pop("seg_decoder_path")
    if seg_decoder_path is None:
        seg_decoder_path = "facebook/mask2former-swin-base"
    return QwenVLSegForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        config=config,
        seg_decoder_path=seg_decoder_path,
        *args, **kwargs
    )