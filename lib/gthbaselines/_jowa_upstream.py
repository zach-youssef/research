"""Upstream JOWA (github.com/CJReinforce/JOWA at JOWA_COMMIT, GPL-3.0) loaded for inference.

The clone is never edited and never vendored: `check.sh` fetches it under ``runs/`` and
`install` patches it at import time (and `build`, patch 6, while building). The patches, each with its reason (the first four come
from the experiment ``2026-11-24-jowa-mac-smoke``, ``src/jowa_patch.py``):

1. ``gym``: upstream imports gym 0.19, which has no arm64 wheels. Only a type annotation in
   ``envs/world_model_env.py`` touches it, so ``gym`` is aliased to gymnasium.
2. ``envs``: its ``__init__`` imports upstream's Atari env wrapper, multiprocess env and
   gym 0.19 spaces, none of which inference needs; ``envs`` is registered as a bare package
   over the clone's directory so only ``envs.world_model_env`` loads.
3. LPIPS: the tokenizer builds a VGG16 LPIPS loss (training only) whose constructor downloads
   ~0.5 GB of pretrained VGG weights. The same architecture is built without downloading; the
   values come from ``tokenizer.pt``, which stores the ``lpips.*`` keys.
4. Checkpoints lack the ``transformer.blocks.N.attn.mask`` buffers (trained with flash-attn,
   which has no mask buffer); the mask is the deterministic tril built at init, so `load`
   skips those keys and fails on any other missing key.
5. MPS empty slices: ``Head.forward`` runs a head on a zero-length slice for heads that apply
   to no token of a short input; MPS asserts inside relu on an empty tensor, so the head runs
   on one row and the result is cut back to zero rows (the same shape on every device).
6. Orthogonal init: the tokenizer's ``init_tokenizer_weights`` calls ``nn.init.orthogonal_``,
   whose QR (Accelerate's LAPACK, torch 2.14.1 on macOS 26 arm64) corrupts the heap or faults
   for the 512 x 576 weight in most fresh processes (42 of 60 runs of a bare
   ``orthogonal_(torch.empty(512, 576))`` crashed; the other shapes never did). `load`
   overwrites every tokenizer parameter, so `build` leaves those tensors uninitialised instead.

One upstream behaviour is kept but made switchable (`set_q_dropout`): the Q heads are
``utils.Ensemble``s built with functorch's ``combine_state_for_ensemble``, which copies the MLPs
in training mode, so ``.eval()`` never reaches them and their dropout (p = 0.01 on the first
layer, ``config/critic_head``) stays active at evaluation, drawn from torch's global generator.
Upstream's reported scores were played that way.

The upstream modules have generic top-level names (``models``, ``utils``, ``agent``,
``envs``), so `install` puts the clone first on ``sys.path`` and a process that imports this
module can't also import other packages with those names.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

JOWA_COMMIT = "8cb02c5e6768a948c566a9012ac8a4fc420c6970"
"""The upstream commit the adapter is written and tested against (2024-12-02)."""

_installed: Path | None = None


def check_clone(jowa_dir: Path) -> None:
    """Raises ValueError unless ``jowa_dir`` is a git checkout of JOWA_COMMIT."""
    try:
        head = subprocess.run(
            ["git", "-C", str(jowa_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"jowa_dir {jowa_dir}: not a git checkout ({error})") from error
    if head != JOWA_COMMIT:
        raise ValueError(f"jowa_dir {jowa_dir}: HEAD is {head}, the adapter needs {JOWA_COMMIT}")


def install(jowa_dir: Path) -> None:
    """Makes the clone at ``jowa_dir`` (checked to be JOWA_COMMIT) importable, with the
    patches above. Idempotent; a second, different clone in one process is refused."""
    global _installed
    jowa_dir = Path(jowa_dir).resolve()
    if _installed is not None:
        if _installed != jowa_dir:
            raise ValueError(f"JOWA is already loaded from {_installed}, not {jowa_dir}")
        return
    check_clone(jowa_dir)
    src = jowa_dir / "src"
    sys.path.insert(0, str(src))

    import gymnasium

    sys.modules.setdefault("gym", gymnasium)
    envs = types.ModuleType("envs")
    envs.__path__ = [str(src / "envs")]
    sys.modules["envs"] = envs

    import torchvision

    import models.tokenizer.lpips as lpips  # type: ignore[import-not-found]

    lpips.models = types.SimpleNamespace(
        vgg16=lambda pretrained=True: torchvision.models.vgg16(weights=None)
    )
    lpips.LPIPS.load_from_pretrained = lambda self: None

    import models.slicer as slicer  # type: ignore[import-not-found]

    head_forward = slicer.Head.forward

    def forward(self, x, num_steps, prev_steps):
        if x.device.type == "mps" and self.compute_slice(num_steps, prev_steps).numel() == 0:
            return self.head_module(x[:, :1])[..., :0, :]
        return head_forward(self, x, num_steps, prev_steps)

    slicer.Head.forward = forward
    _installed = jowa_dir


def build(jowa_dir: Path, size: str) -> tuple[Any, Any]:
    """The tokenizer and JOWA model of ``size`` (``JOWA_40M``, ``JOWA_150M``), on the CPU,
    from the clone's own Hydra configs, with untrained weights (the tokenizer's orthogonally
    initialised ones uninitialised, patch 6): call `load` before using them."""
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    from action_tokenizer import ATARI_NUM_ACTIONS  # type: ignore[import-not-found]
    from models.jowa_model import JOWAModel  # type: ignore[import-not-found]

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(Path(jowa_dir).resolve() / "config"), version_base="1.3"):
        cfg = compose(
            config_name="eval", overrides=[f"transformer={size}", f"critic_head={size}"]
        )
    GlobalHydra.instance().clear()
    with mock.patch.object(torch.nn.init, "orthogonal_", lambda tensor, *a, **k: tensor):
        tokenizer = instantiate(cfg.tokenizer)
    model = JOWAModel(
        obs_vocab_size=cfg.tokenizer.vocab_size,
        act_vocab_size=ATARI_NUM_ACTIONS,
        config_transformer=instantiate(cfg.transformer),
        config_critic_arch=cfg.critic_head,
        config_critic_train=cfg.action,
        device=torch.device("cpu"),
    )
    return tokenizer, model


def load(tokenizer: Any, model: Any, tokenizer_file: Path, model_file: Path) -> list[str]:
    """Upstream ``Agent.load`` with the ``attn.mask`` buffers (patch 4) skipped; returns the
    skipped keys. Raises KeyError for any other key missing from a checkpoint."""
    import torch

    def find(name: str, checkpoint: dict[str, Any]) -> str:
        for prefix in ("_orig_mod.module.", "", "module.", "_orig_mod."):
            if prefix + name in checkpoint:
                return prefix + name
        raise KeyError(f"{name} isn't in the checkpoint")

    # The files' SHA-256s are checked before this, so weights_only=True only guards against
    # a pickle that isn't a plain state dict.
    saved = torch.load(tokenizer_file, map_location="cpu", weights_only=True)
    state = tokenizer.state_dict()
    for name in state:
        state[name] = saved[find(name, saved)]
    tokenizer.load_state_dict(state)

    skipped = []
    saved = torch.load(model_file, map_location="cpu", weights_only=True)
    state = model.state_dict()
    for name in state:
        if name.endswith(".attn.mask"):
            skipped.append(name)
            continue
        state[name] = saved[find(name, saved)]
    model.load_state_dict(state, strict=False)
    return skipped


def set_q_dropout(model: Any, active: bool) -> int:
    """Turns the Q heads' dropout (see above) on, as upstream evaluates, or off (eval mode).
    Returns how many ensembles it set; raises if the model has none."""
    from utils import Ensemble  # type: ignore[import-not-found]

    count = 0
    for module in model.modules():
        if isinstance(module, Ensemble):
            module.vmap.__wrapped__.stateless_model.train(active)
            count += 1
    if count == 0:
        raise ValueError("no utils.Ensemble in the model: upstream changed")
    return count
