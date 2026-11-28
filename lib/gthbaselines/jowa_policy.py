"""JOWA (Cheng et al. 2025, ``cheng2025jointly`` in kb/) as a ``gtharness.policies.Policy``.

    gth evaluate DATA --protocol P --policy gthbaselines.jowa_policy:JowaPolicy \\
        --policy-kwargs '{"jowa_dir": ..., "weights_dir": ..., "size": "JOWA_40M", ...}' --jobs 1

Every setting that changes what the policy plays is a required keyword (README.md lists them
with the values that reproduce the paper's evaluation). Upstream JOWA, the model code, is
loaded from a pinned clone (`_jowa_upstream`); the weights from the README's Drive files,
checked by SHA-256 (`jowa_weights`).

What the policy does per step, as upstream's ``Agent.act`` and ``AgentEnv.step`` do:

- it encodes the newest observation (84x84 gray, scaled to [0, 1]) into 36 VQ tokens and
  appends them, after the previous step's action token, to a token context of the last
  ``context_steps`` steps (upstream's ``num_given_steps``, at most the model's 8 blocks);
  JOWA stacks no frames, so it ignores the older observations the protocol passes;
- with probability ``epsilon`` (upstream: 0.001, Table 16) it plays a uniformly random valid
  action, drawn from the policy's own seeded generator rather than torch's global one;
- otherwise it plays the greedy action of the Q head (C51 expectation, averaged over the
  ensemble) or, with ``planning="beam_search"``, the first action of upstream's beam search in
  the world model, over the valid actions. Upstream's world model masks its actions with the
  task game's minimal set whatever the agent's mask, so beam search takes
  ``valid_actions="task_game"`` only.

Precision: upstream evaluates under ``torch.cuda.amp.autocast(dtype=bfloat16)``
(``config/eval.yaml``), which is bfloat16 on CUDA and a no-op elsewhere; ``precision`` makes
the choice explicit on every device: ``bfloat16_autocast`` (``torch.autocast`` on the
policy's device: on the CPU and MPS the tokenizer runs in bfloat16, checked) or ``float32``.

The Q heads' dropout (p = 0.01) stays active in upstream's evaluation, so its greedy action
is stochastic; ``q_dropout="upstream"`` keeps that, with torch's generator seeded at each
seeded reset, and ``"off"`` makes the heads deterministic (`_jowa_upstream.set_q_dropout`).

The valid actions are JOWA's 18 (ALE's full set, in ALE's order) or, with
``valid_actions="task_game"``, the ALE minimal set of ``task_game`` (upstream's mask,
Table 17). ``task_game`` also picks the task embedding: upstream's training game index, so a
game JOWA never saw has to borrow one (upstream itself maps its held-out games onto similar
training games for this).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gthbaselines import _jowa_upstream, jowa_weights

ATARI_ACTIONS = 18
PLANNING = ("greedy", "beam_search")
VALID_ACTIONS = ("all", "task_game")
Q_DROPOUT = ("upstream", "off")
DEVICES = ("cpu", "mps")
PRECISIONS = ("bfloat16_autocast", "float32")


class JowaPolicy:
    """JOWA-40M or -150M playing one game with a fixed task embedding.

    Required keywords:
        policy_seed: the protocol's; JOWA has one released checkpoint, so every seed plays
            the same weights (the runner then reads several seeds as sampling seeds).
        jowa_dir: a clone of github.com/CJReinforce/JOWA at `_jowa_upstream.JOWA_COMMIT`.
        weights_dir: holds ``tokenizer.pt`` and ``<size>.pt`` (`jowa_weights`).
        size: ``JOWA_40M`` or ``JOWA_150M``.
        device: ``cpu`` or ``mps`` (MPS needs the Mac GPU slot).
        precision: ``bfloat16_autocast`` (upstream's, on CUDA) or ``float32``.
        threads: torch's CPU threads in this process (`torch.set_num_threads`); with
            ``--jobs N`` the processes share the cores.
        task_game: one of upstream's ``GAME_NAMES`` (``Assault``); its task embedding is used.
        valid_actions: ``all`` (18) or ``task_game`` (that game's ALE minimal set).
        context_steps: steps of token context, 1 to 8 (upstream's ``num_given_steps``). The
            reported scores use a per-game value, ``buffer_size`` in upstream's
            ``eval/JOWA_<size>_raw_scores.json`` (Assault: 3 for 40M, 4 for 150M); 8 is
            only ``config/eval.yaml``'s default.
        epsilon: the probability of a random valid action each step (upstream: 0.001).
        q_dropout: ``upstream`` (the Q heads' dropout active, as upstream evaluates; torch's
            global generator is seeded from the reset seed) or ``off`` (deterministic heads).
        planning: ``greedy`` or ``beam_search``; beam search takes ``beam_width`` and
            ``horizon`` (upstream's per-game values are in its ``eval/*_raw_scores.json``)
            and about 1-2 s a step on the Mac (the jowa smoke), so it is for the PC.

    `act` returns ALE action numbers (0..17), so the protocol's action set is ``atari_full``
    (`obs_requirements`); the runner maps them onto the game's pad (``GameConfig.from_atari``).
    """

    def __init__(
        self,
        *,
        policy_seed: int,
        jowa_dir: str,
        weights_dir: str,
        size: str,
        device: str,
        precision: str,
        threads: int,
        task_game: str,
        valid_actions: str,
        context_steps: int,
        epsilon: float,
        q_dropout: str,
        planning: str,
        beam_width: int | None = None,
        horizon: int | None = None,
    ) -> None:
        _choice("size", size, jowa_weights.SIZES)
        _choice("device", device, DEVICES)
        _choice("precision", precision, PRECISIONS)
        _choice("valid_actions", valid_actions, VALID_ACTIONS)
        _choice("planning", planning, PLANNING)
        _choice("q_dropout", q_dropout, Q_DROPOUT)
        if not isinstance(threads, int) or threads < 1:
            raise ValueError(f"threads must be an integer >= 1, not {threads!r}")
        if not isinstance(context_steps, int) or not 1 <= context_steps <= 8:
            raise ValueError(f"context_steps must be an integer in 1..8, not {context_steps!r}")
        if not 0 <= epsilon < 1:
            raise ValueError(f"epsilon must be in [0, 1), not {epsilon!r}")
        if planning == "beam_search":
            if valid_actions != "task_game":
                raise ValueError(
                    "beam_search needs valid_actions='task_game': upstream's world model"
                    " plans over the task game's minimal set only"
                )
            for name, value in (("beam_width", beam_width), ("horizon", horizon)):
                if not isinstance(value, int) or value < 1:
                    raise ValueError(f"beam_search needs {name}, an integer >= 1, not {value!r}")
            if horizon >= 8:
                raise ValueError(
                    f"horizon ({horizon}) must be below 8: upstream's beam search dreams"
                    " within the model's 8 blocks"
                )
        elif beam_width is not None or horizon is not None:
            raise ValueError("beam_width and horizon are for planning='beam_search' only")

        import torch

        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("device 'mps': torch reports MPS unavailable")
        torch.set_num_threads(threads)
        jowa = Path(jowa_dir)
        _jowa_upstream.install(jowa)
        from action_tokenizer import (  # type: ignore[import-not-found]
            GAME_NAMES,
            LIMITED_ACTION_TO_FULL_ACTION,
        )
        from agent import Agent  # type: ignore[import-not-found]

        _choice("task_game", task_game, tuple(GAME_NAMES))
        tokenizer_file, _ = jowa_weights.verified(Path(weights_dir), "tokenizer.pt")
        model_file, model_sha = jowa_weights.verified(Path(weights_dir), f"{size}.pt")
        tokenizer, model = _jowa_upstream.build(jowa, size)
        _jowa_upstream.load(tokenizer, model, tokenizer_file, model_file)

        self._device = torch.device(device)
        tokenizer.to(self._device)
        model.to(self._device)
        # Plain tensor attributes (not buffers), which .to() leaves on the CPU.
        for attr in ("atoms_support", "log_alpha"):
            if hasattr(model, attr):
                setattr(model, attr, getattr(model, attr).to(self._device))
        self._agent = Agent(
            tokenizer,
            model,
            GAME_NAMES.index(task_game),
            num_envs=1,
            dtype=torch.bfloat16 if precision == "bfloat16_autocast" else torch.float32,
            buffer_size=context_steps,
            device=self._device,
            should_plan="beam_search" if planning == "beam_search" else False,
            beam_width=beam_width or 1,
            horizon=horizon or 0,
        ).eval()
        _jowa_upstream.set_q_dropout(model, q_dropout == "upstream")
        if valid_actions == "all":
            self._agent.action_mask = torch.ones(
                1, ATARI_ACTIONS, dtype=torch.bool, device=self._device
            )
            self._valid = np.arange(ATARI_ACTIONS)
        else:
            self._valid = np.sort(LIMITED_ACTION_TO_FULL_ACTION[task_game])
        self._torch = torch
        self._planning = planning
        self._epsilon = float(epsilon)
        self._q_dropout = q_dropout
        self._autocast = precision == "bfloat16_autocast"
        self._rng: np.random.Generator | None = None
        self._last: int | None = None

        from gtharness.policies import ObsRequirements

        plan = f"beam{beam_width}x{horizon}" if planning == "beam_search" else "greedy"
        self.id = (
            f"jowa-{size.split('_')[1].lower()}-{plan}-ctx{context_steps}-eps{epsilon:g}"
            f"-task{task_game}-{valid_actions}-qdrop{q_dropout}"
            f"-{'bf16' if precision == 'bfloat16_autocast' else 'fp32'}-{device}-v1"
        )
        self.checkpoint_sha256 = model_sha
        self.obs_requirements = ObsRequirements(
            size=(84, 84),
            grayscale=True,
            maxpool=True,
            frame_stack=None,  # JOWA reads the newest observation; its context is tokens
            frameskip=4,
            action_set="atari_full",
        )

    def reset(self, seed: int | None, game: Any) -> None:
        """Starts an episode: clears the token context; ``seed`` reseeds the epsilon draws
        and, with ``q_dropout="upstream"``, torch's global generator (the dropout masks)."""
        if seed is not None:
            self._rng = np.random.Generator(np.random.PCG64(seed))
            if self._q_dropout == "upstream":
                self._torch.manual_seed(seed)
        elif self._rng is None:
            raise ValueError("the first reset needs a seed")
        if game.n_actions != ATARI_ACTIONS:
            raise ValueError(
                f"{game.game_id}: {game.n_actions} actions; JOWA plays ALE's {ATARI_ACTIONS}"
                " (action set atari_full)"
            )
        self._agent.reset(1)
        self._last = None

    def act(self, obs: np.ndarray) -> int:
        torch = self._torch
        agent = self._agent
        if self._rng is None:
            raise ValueError("act before the first reset")
        if obs.ndim != 4 or obs.shape[1:] != (84, 84, 1) or obs.dtype != np.uint8:
            raise ValueError(f"obs must be uint8[stack, 84, 84, 1], not {obs.dtype}{obs.shape}")
        with torch.no_grad(), torch.autocast(
            self._device.type, dtype=torch.bfloat16, enabled=self._autocast
        ):
            frame = torch.from_numpy(np.ascontiguousarray(obs[-1, :, :, 0]))
            x = frame.to(self._device, torch.float32).div(255).view(1, 1, 84, 84)
            if self._last is not None:
                agent.update_memory(torch.tensor([[self._last]], device=self._device))
            agent.update_memory(agent.tokenizer.encode(x, should_preprocess=True).tokens)
            if self._rng.random() < self._epsilon:
                action = int(self._rng.choice(self._valid))
            elif self._planning == "beam_search":
                action = int(agent.plan_beam_search().view(-1)[0])
            else:
                out = agent.jowa_model(
                    agent.token_buffer[:, : agent.used_token_idx], agent.env_token
                )
                action = int(agent.choose_action(out.logits_q)[0])
        self._last = action
        return action


def _choice(name: str, value: object, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}, not {value!r}")
