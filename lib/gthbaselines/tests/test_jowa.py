"""The JOWA adapter's tests (run by ../check.sh, which sets the GTHB_* variables): argument
refusals, the policy with the real JOWA-40M weights on the CPU, and the testcore smoke."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from gthbaselines import jowa_smoke, jowa_weights
from gthbaselines.jowa_policy import ATARI_ACTIONS, JowaPolicy


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is unset: run lib/gthbaselines/check.sh")
    return value


def kwargs(**changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "policy_seed": 0,
        "jowa_dir": env("GTHB_JOWA_DIR"),
        "weights_dir": env("GTHB_JOWA_WEIGHTS"),
        "size": "JOWA_40M",
        "device": "cpu",
        "precision": "float32",
        "threads": 2,
        "task_game": "Assault",
        "valid_actions": "all",
        "context_steps": 8,
        "epsilon": 0.0,
        "q_dropout": "off",
        "planning": "greedy",
    }
    base.update(changes)
    return base


def game_info(game_id: str = "testcore"):
    from gtharness.policies import GameInfo

    # As the runner makes it under atari_full: ALE's 18 actions.
    return GameInfo(game_id, ATARI_ACTIONS, ("shot", "bomb", "aux1", "aux2"))


def frames(seed: int, n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (4, 84, 84, 1), dtype=np.uint8) for _ in range(n)]


def play(policy: JowaPolicy, seed: int, obs: list[np.ndarray]) -> list[int]:
    policy.reset(seed, game_info())
    return [policy.act(o) for o in obs]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"size": "JOWA_70M"}, "size must be one of"),
        ({"device": "cuda"}, "device must be one of"),
        ({"planning": "mcts"}, "planning must be one of"),
        ({"valid_actions": "minimal"}, "valid_actions must be one of"),
        ({"threads": 0}, "threads must be"),
        ({"context_steps": 9}, "context_steps must be"),
        ({"epsilon": 1.0}, "epsilon must be"),
        ({"precision": "float16"}, "precision must be one of"),
        ({"beam_width": 2}, "for planning='beam_search' only"),
        ({"planning": "beam_search", "beam_width": 2, "horizon": 1}, "valid_actions='task_game'"),
        (
            {"planning": "beam_search", "valid_actions": "task_game", "horizon": 2},
            "beam_search needs beam_width",
        ),
        (
            {"planning": "beam_search", "valid_actions": "task_game", "beam_width": 2, "horizon": 8},
            "must be below 8",
        ),
        ({"q_dropout": "on"}, "q_dropout must be one of"),
        ({"task_game": "Gradius"}, "task_game must be one of"),
    ],
)
def test_refuses_bad_arguments(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        JowaPolicy(**kwargs(**changes))


def test_every_setting_is_required() -> None:
    for name in kwargs():
        if name == "policy_seed":
            continue
        partial = kwargs()
        del partial[name]
        with pytest.raises(TypeError, match=name):
            JowaPolicy(**partial)


@pytest.fixture(scope="module")
def greedy() -> JowaPolicy:
    return JowaPolicy(**kwargs())


def test_identity_and_obs_requirements(greedy: JowaPolicy) -> None:
    from gtharness.policies import check_identity

    check_identity(greedy, "JowaPolicy")
    assert greedy.checkpoint_sha256 == jowa_weights.FILES["JOWA_40M.pt"][0]
    assert greedy.id == "jowa-40m-greedy-ctx8-eps0-taskAssault-all-qdropoff-fp32-cpu-v1"
    req = greedy.obs_requirements
    assert (req.size, req.grayscale, req.maxpool, req.frameskip) == ((84, 84), True, True, 4)
    assert req.frame_stack is None and req.action_set == "atari_full"


def test_greedy_is_deterministic_and_reset_clears_the_context(greedy: JowaPolicy) -> None:
    obs = frames(1, 12)  # past the 8-step context, so the buffer slides
    first = play(greedy, 0, obs)
    assert all(0 <= a < ATARI_ACTIONS for a in first)
    assert play(greedy, 5, obs) == first  # greedy with epsilon 0: the seed doesn't matter
    assert greedy._agent.used_token_idx <= 8 * 37
    agent = greedy._agent
    tokens = agent.token_buffer[:, : agent.used_token_idx]
    with greedy._torch.no_grad():
        q = [agent.jowa_model(tokens, agent.env_token).logits_q for _ in range(4)]
    assert all(greedy._torch.equal(q[0], other) for other in q[1:])


def test_act_before_reset_is_refused() -> None:
    policy = JowaPolicy(**kwargs())
    with pytest.raises(ValueError, match="before the first reset"):
        policy.act(frames(0, 1)[0])


def test_matches_upstreams_agent_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same frames through upstream's Agent.act, then AgentEnv.step's update_memory of the
    action, play the same actions (epsilon 0 on both sides, deterministic heads)."""
    import torch
    from action_tokenizer import GAME_NAMES  # type: ignore[import-not-found]
    from agent import Agent  # type: ignore[import-not-found]

    policy = JowaPolicy(**kwargs(valid_actions="task_game"))
    obs = frames(9, 12)
    ours = play(policy, 0, obs)
    upstream = Agent(
        policy._agent.tokenizer,
        policy._agent.jowa_model,
        GAME_NAMES.index("Assault"),
        num_envs=1,
        dtype=torch.float32,
        buffer_size=8,
        device=torch.device("cpu"),
        should_plan=False,
    ).eval()
    rand = torch.rand
    monkeypatch.setattr(  # upstream's epsilon draw: never below 0.001
        torch, "rand", lambda *size, **kw: torch.ones(*size) if size == (1,) else rand(*size, **kw)
    )
    theirs = []
    for o in obs:
        x = torch.from_numpy(o[-1, :, :, 0].copy()).float().div(255).view(1, 1, 84, 84)
        act = upstream.act(x)
        upstream.update_memory(act.view(-1, 1))
        theirs.append(int(act[0]))
    assert ours == theirs


def test_beam_search_plays_within_the_mask() -> None:
    from action_tokenizer import LIMITED_ACTION_TO_FULL_ACTION  # type: ignore[import-not-found]

    policy = JowaPolicy(
        **kwargs(planning="beam_search", beam_width=2, horizon=1, valid_actions="task_game")
    )
    assert policy.id.startswith("jowa-40m-beam2x1-")
    actions = play(policy, 0, frames(10, 2))
    assert set(actions) <= set(LIMITED_ACTION_TO_FULL_ACTION["Assault"])


def test_bfloat16_autocast_plays() -> None:
    policy = JowaPolicy(**kwargs(precision="bfloat16_autocast"))
    assert "-bf16-cpu-" in policy.id
    dtypes = set()
    conv = policy._agent.tokenizer.encoder.conv_in
    conv.register_forward_hook(lambda module, args, out: dtypes.add(out.dtype))
    actions = play(policy, 0, frames(11, 10))
    assert dtypes == {policy._torch.bfloat16}
    assert all(0 <= a < ATARI_ACTIONS for a in actions)


def test_only_the_newest_frame_counts(greedy: JowaPolicy) -> None:
    obs = frames(2, 6)
    first = play(greedy, 0, obs)
    noisy = [np.concatenate([np.zeros_like(f[:3]), f[-1:]]) for f in obs]
    assert play(greedy, 0, noisy) == first


def test_upstream_q_dropout_follows_the_reset_seed() -> None:
    policy = JowaPolicy(**kwargs(q_dropout="upstream"))
    obs = frames(6, 30)
    a, b = play(policy, 7, obs), play(policy, 7, obs)
    assert a == b
    # The dropout is live: one context, many forward passes, different Q values.
    agent = policy._agent
    tokens = agent.token_buffer[:, : agent.used_token_idx]
    with policy._torch.no_grad():
        q = [agent.jowa_model(tokens, agent.env_token).logits_q for _ in range(4)]
    assert any(not policy._torch.equal(q[0], other) for other in q[1:])


def test_epsilon_draws_follow_the_reset_seed() -> None:
    policy = JowaPolicy(**kwargs(epsilon=0.9))
    obs = frames(3, 20)
    a, b, c = play(policy, 7, obs), play(policy, 7, obs), play(policy, 8, obs)
    assert a == b and a != c


def test_task_game_mask_limits_the_actions() -> None:
    from action_tokenizer import LIMITED_ACTION_TO_FULL_ACTION  # type: ignore[import-not-found]

    policy = JowaPolicy(**kwargs(valid_actions="task_game", epsilon=0.5))
    allowed = set(LIMITED_ACTION_TO_FULL_ACTION["Assault"])
    assert len(allowed) < ATARI_ACTIONS
    assert set(play(policy, 0, frames(4, 40))) <= allowed


def test_reset_refuses_flat_actions() -> None:
    from gtharness.policies import GameInfo

    policy = JowaPolicy(**kwargs())
    with pytest.raises(ValueError, match="JOWA plays ALE's 18"):
        policy.reset(0, GameInfo("testcore", 144, ("shot",)))


def test_testcore_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    status = jowa_smoke.main(
        [
            "--out", str(tmp_path / "smoke"),
            "--jowa-dir", env("GTHB_JOWA_DIR"),
            "--weights-dir", env("GTHB_JOWA_WEIGHTS"),
            "--core-path", env("GTHB_TESTCORE"),
            "--size", "JOWA_40M",
            "--device", "cpu",
            "--precision", "float32",
            "--steps", "50",
            "--threads", "2",
        ]
    )
    assert status == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["checked"] and summary["truncated"] and summary["steps"] > 40
    assert summary["policy_id"].startswith("jowa-40m-greedy-ctx8-eps0.001-taskAssault")
    protocol = (tmp_path / "smoke" / "protocol.toml").read_text(encoding="utf-8")
    assert 'action_set = "atari_full"' in protocol and "frameskip = 4" in protocol
