"""Runtime patches that let upstream JOWA (CJReinforce/JOWA @ 8cb02c5) run on an Apple-silicon Mac.

Nothing in the upstream clone is edited. Patches, each with its reason:
1. `gym` -> `gymnasium` alias: upstream imports gym 0.19 (no arm64 wheels for atari-py); only
   `gym.spaces`, `gym.Env` and an unused `gym.make` path are touched.
2. `pygame` stub: game/keymap.py imports pygame for key constants only (interactive play).
3. LPIPS: the tokenizer builds a VGG16 LPIPS loss (training only) whose constructor downloads
   ~0.5 GB of pretrained VGG weights. Build the same architecture without downloading; the
   values come from tokenizer.pt, which stores the lpips.* keys.
4. Environment: `create_atari_environment` (gym 0.19 `{Game}NoFrameskip-v4` + atari-py) is
   replaced by gymnasium `ALE/{Game}-v5` (frameskip 1, no sticky actions, minimal action set),
   wrapped to the gym 0.19 API so upstream's own AtariPreprocessing, action mapping and AgentEnv
   run unchanged.
5. Checkpoint loading: upstream's Agent.load raises KeyError for `transformer.blocks.N.attn.mask`,
   the causal-mask buffer that exists only on the non-FlashAttention path (the checkpoints were
   trained with flash-attn, so they lack it; the mask is the deterministic tril built at init).
   `load_checkpoint` is Agent.load's logic with those buffers skipped and reported.
Everything else (agent, models, planning; the CUDA-only autocast is a disabled no-op without
CUDA) is upstream code.
"""
import sys
import types


def install(jowa_src: str):
    sys.path.insert(0, jowa_src)
    import ale_py  # noqa: F401  (registers ALE/* with gymnasium)
    import gymnasium
    sys.modules["gym"] = gymnasium
    sys.modules["gym.spaces"] = gymnasium.spaces
    sys.modules["gym.spaces.box"] = gymnasium.spaces.box

    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return 0
    stub = _Stub("pygame")
    stub.__file__ = "<pygame stub>"
    sys.modules["pygame"] = stub

    import torchvision
    import models.tokenizer.lpips as lp
    lp.models = types.SimpleNamespace(
        vgg16=lambda pretrained=True: torchvision.models.vgg16(weights=None))
    lp.LPIPS.load_from_pretrained = lambda self: None

    import envs.atari_env_wrapper as aw
    state = {"seed": 0}

    class _GymShim:
        """gymnasium unwrapped ALE env exposed with the gym 0.19 API."""

        def __init__(self, game, seed):
            self.env = gymnasium.make(
                f"ALE/{game}-v5", frameskip=1, repeat_action_probability=0.0,
                full_action_space=False, obs_type="grayscale").unwrapped
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            self.ale = self.env.ale
            self.reward_range = (-float("inf"), float("inf"))
            self.metadata = {}
            self._seed = seed
            self._first = True

        def reset(self):
            obs, _ = self.env.reset(seed=self._seed if self._first else None)
            self._first = False
            return obs

        def step(self, action):
            obs, rew, term, trunc, info = self.env.step(int(action))
            return obs, rew, bool(term or trunc), info

        def close(self):
            self.env.close()

    def create_atari_environment(game_name, sticky_actions=True):
        assert not sticky_actions
        return aw.AtariPreprocessing(_GymShim(game_name, state["seed"]))

    aw.create_atari_environment = create_atari_environment
    return state


def load_checkpoint(agent, path, size):
    """Agent.load with the attn.mask buffers (built at init, absent from checkpoints) skipped.
    Returns the list of skipped keys; any other missing key raises KeyError."""
    import os
    import torch

    def find(name, ckpt):
        for prefix in ["_orig_mod.module.", "", "module.", "_orig_mod."]:
            if prefix + name in ckpt:
                return prefix + name
        raise KeyError(f"{name} not found in ckpt.")

    skipped = []
    ck_tok = torch.load(os.path.join(path, "tokenizer.pt"), map_location="cpu")
    d = agent.tokenizer.state_dict()
    for name in d:
        d[name] = ck_tok[find(name, ck_tok)]
    agent.tokenizer.load_state_dict(d)
    ck = torch.load(os.path.join(path, size + ".pt"), map_location="cpu")
    d = agent.jowa_model.state_dict()
    for name in d:
        if name.endswith(".attn.mask"):
            skipped.append(name)
            continue
        d[name] = ck[find(name, ck)]
    agent.jowa_model.load_state_dict(d, strict=False)
    return skipped
