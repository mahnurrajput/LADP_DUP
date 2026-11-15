"""A tiny, randomly initialised LlavaForConditionalGeneration for CPU tests.

It runs transformers' real LLaVA code (vision tower -> hidden_states[-2][:, 1:]
-> multi_modal_projector -> masked_scatter into the image-token positions ->
Llama), so tests built on it exercise the actual multimodal interface.
Image 56x56 with patch 14 -> 16 patches (+ CLS).
"""

import torch
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

N_PATCHES = 16
IMAGE_TOKEN = 63
VOCAB = 64
EOS = 2


def tiny_llava(seed: int = 0) -> LlavaForConditionalGeneration:
    torch.manual_seed(seed)
    vc = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                          num_attention_heads=4, image_size=56, patch_size=14, projection_dim=16)
    tc = LlamaConfig(vocab_size=VOCAB, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                     pad_token_id=0, bos_token_id=1, eos_token_id=EOS)
    cfg = LlavaConfig(vision_config=vc, text_config=tc, image_token_index=IMAGE_TOKEN,
                      vision_feature_layer=-2, vision_feature_select_strategy="default",
                      image_seq_length=N_PATCHES, projector_hidden_act="gelu")
    return LlavaForConditionalGeneration(cfg).eval()


def tiny_inputs(seed: int = 1, n_before: int = 3, n_after: int = 5):
    """input_ids = [bos, text…, <image>×16, text…] and matching pixel_values."""
    g = torch.Generator().manual_seed(seed)
    before = torch.randint(3, 60, (n_before,), generator=g)
    after = torch.randint(3, 60, (n_after,), generator=g)
    ids = torch.cat([torch.tensor([1]), before, torch.full((N_PATCHES,), IMAGE_TOKEN), after])[None]
    px = torch.randn(1, 3, 56, 56, generator=g)
    return ids, px
