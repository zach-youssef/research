"""Minimal device patch for MineDojo/NitroGen inference (in-process, no server).

NitroGen's `inference_session.py` hard-codes `cuda` (model.to, tensors, bf16 autocast) and its
vision tower is built with `from_pretrained`, which downloads weights the checkpoint then
overwrites. This module reproduces `load_model` and `_predict_flowmatching` from the pinned commit
with the device as a parameter, and nothing else changed. The upstream code is imported, not copied.
"""
import contextlib
import time

import numpy as np
import torch
from transformers import AutoImageProcessor, SiglipConfig, SiglipVisionModel

from nitrogen.cfg import CkptConfig
from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig


def load_model(ckpt_path: str, siglip_dir: str):
    """Load the checkpoint on CPU. `siglip_dir` holds only config.json and preprocessor_config.json
    of the vision encoder named in the checkpoint; its weights come from the checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = CkptConfig.model_validate(ckpt["ckpt_config"])
    model_cfg, tok_cfg = cfg.model_cfg, cfg.tokenizer_cfg
    assert isinstance(model_cfg, NitroGen_Config) and isinstance(tok_cfg, NitrogenTokenizerConfig)
    assert tok_cfg.game_mapping_cfg is None, "this checkpoint is expected to be game-unconditional"
    vision_name = model_cfg.vision_encoder_name

    # Replace the weight download by construction from the config (weights overwritten below).
    orig = SiglipVisionModel.from_pretrained

    def _from_config(cls, name, *a, **k):
        assert name == vision_name, name
        return cls(SiglipConfig.from_pretrained(siglip_dir).vision_config)

    SiglipVisionModel.from_pretrained = classmethod(_from_config)
    try:
        model = NitroGen(config=model_cfg, game_mapping=None)
    finally:
        SiglipVisionModel.from_pretrained = orig
    img_proc = AutoImageProcessor.from_pretrained(siglip_dir)
    tok_cfg.training = False
    tokenizer = NitrogenTokenizer(tok_cfg)
    model.load_state_dict(ckpt["model"], strict=True)  # raises on any missing or unexpected key
    model.eval()
    tokenizer.eval()
    return model, tokenizer, img_proc, cfg


@contextlib.contextmanager
def cpu_noise():
    """Draw torch.randn on the CPU generator even for device tensors, so that the same seed gives the
    same noise on CPU and MPS (the MPS generator differs)."""
    orig = torch.randn

    def randn(*size, device=None, **kw):
        out = orig(*size, **kw)
        return out.to(device) if device is not None else out

    torch.randn = randn
    try:
        yield
    finally:
        torch.randn = orig


def predict(model, tokenizer, img_proc, frame_rgb: np.ndarray, device: str, seed: int = 0,
            autocast_bf16: bool = False):
    """One action chunk for one RGB uint8 frame (H, W, 3). Mirrors `_predict_flowmatching` with
    context length 1, no game conditioning, cfg_scale 1. Returns (raw tensor on CPU, decoded dict
    of CPU tensors, seconds). Time includes preprocessing, 16 Euler steps and the copy back."""
    t0 = time.perf_counter()
    pixel_values = img_proc([frame_rgb], return_tensors="pt")["pixel_values"]  # (1, 3, 256, 256)
    frames = pixel_values.to(device)
    dropped = torch.zeros((1,), dtype=torch.bool, device=device)
    data = tokenizer.encode({"frames": frames, "dropped_frames": dropped, "game": None})
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, np.ndarray):
            data[k] = torch.tensor(v, device=device).unsqueeze(0)
        else:
            data[k] = [v]
    torch.manual_seed(seed)
    ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if autocast_bf16 else contextlib.nullcontext()
    with torch.inference_mode(), cpu_noise(), ctx:
        out = model.get_action(data, old_layout=False)
        decoded = tokenizer.decode(out)
    raw = out["action_tensor"].float().cpu()
    decoded = {k: v.float().cpu() for k, v in decoded.items()}
    if device == "mps":
        torch.mps.synchronize()
    return raw, decoded, time.perf_counter() - t0
