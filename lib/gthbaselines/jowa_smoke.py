"""The JOWA adapter's smoke run: ``gth evaluate`` on the harness's test core under jowa2025.

    python -m gthbaselines.jowa_smoke --out DIR --jowa-dir J --weights-dir W --core-path LIB \\
        --size JOWA_40M --device cpu --precision float32 --steps 200 --threads 4

The protocol is the harness's ``jowa2025`` preset with only what the test core needs changed:
the game (``testcore``), a cap of ``--steps`` agent steps, and one episode. The ``atari_full``
action set (the runner maps JOWA's ALE actions onto the pad), frame skip, max-pool, 84x84 gray
area-resized observations, sticky actions and no-ops stay jowa2025's. The test core isn't an
Atari game, so the run checks plumbing (loading, the observation path, the action map, the
results files), not play: the task embedding and mask are Assault's, with all 18 actions
valid.

It writes DIR/data (the test core's content and stand-in BIOS, as the harness's
tests/conftest.py writes them), DIR/protocol.toml and DIR/results (the run), and prints one
JSON line: the episode's score, steps, the policy's ms per step and the run's wall time.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import zipfile
from pathlib import Path

# The test core's files, as the harness's tests/conftest.py (TESTCORE_CONTENT, TESTCORE_BIOS)
# writes them; games/testcore/game.toml pins their SHA-1s, so a drift fails the run's checks.
TESTCORE_CONTENT = b"gth-testcore content: seeds the test core's PRNG\n"
TESTCORE_BIOS = b"gth-testcore stand-in BIOS: the test core reads none\n"
FRAMESKIP = 4


def write_testcore_data(root: Path) -> tuple[Path, Path]:
    """The data directory (BIOS under system/) and the ROM directory for the test core."""
    roms = root / "roms"
    roms.mkdir(parents=True)
    (roms / "testcore.gthc").write_bytes(TESTCORE_CONTENT)
    with zipfile.ZipFile(roms / "gth_testcore_parent.zip", "w", zipfile.ZIP_DEFLATED) as parent:
        parent.writestr("parent.bin", TESTCORE_BIOS)
    system = root / "data" / "system"
    (system / "gth").mkdir(parents=True)
    (system / "gth_testcore_bios.bin").write_bytes(TESTCORE_BIOS)
    with zipfile.ZipFile(system / "gth" / "gth_testcore_bios.zip", "w") as bios:
        bios.writestr("bios.bin", TESTCORE_BIOS)
    return root / "data", roms


def smoke_protocol(steps: int) -> str:
    """jowa2025 on the test core, capped at ``steps`` agent steps (canonical text)."""
    from gtharness.protocol import PROTOCOLS_DIR, load

    base = load(PROTOCOLS_DIR / "jowa2025.toml")
    protocol = dataclasses.replace(
        base,
        games=("testcore",),
        max_episode_frames=steps * FRAMESKIP,
        episodes_per_seed=1,
    )
    return protocol.expand().text


def policy_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "jowa_dir": str(args.jowa_dir),
        "weights_dir": str(args.weights_dir),
        "size": args.size,
        "device": args.device,
        "precision": args.precision,
        "threads": args.threads,
        "task_game": "Assault",
        "valid_actions": "all",
        "context_steps": 8,
        "epsilon": 0.001,
        "q_dropout": "upstream",
        "planning": "greedy",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jowa-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--core-path", type=Path, required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args(argv)

    from gtharness.cli import main as gth
    from gtharness.evaluate import check_results

    data, roms = write_testcore_data(args.out)
    os.environ["GTH_ROM_DIR"] = str(roms)
    protocol = args.out / "protocol.toml"
    protocol.write_text(smoke_protocol(args.steps), encoding="utf-8")
    results = args.out / "results"
    status = gth(
        [
            "evaluate",
            str(data),
            "--protocol",
            str(protocol),
            "--policy",
            "gthbaselines.jowa_policy:JowaPolicy",
            "--policy-kwargs",
            json.dumps(policy_kwargs(args)),
            "--core-path",
            str(args.core_path),
            "--out",
            str(results),
            "--jobs",
            "1",
        ]
    )
    if status != 0:
        print(f"gth evaluate exited {status}", file=sys.stderr)
        return 1
    checked = check_results(results)
    run = json.loads((results / "run.json").read_text(encoding="utf-8"))
    (episode,) = [json.loads(line) for line in (results / "episodes.jsonl").open()]
    print(
        json.dumps(
            {
                "device": args.device,
        "precision": args.precision,
                "size": args.size,
                "policy_id": episode["policy_id"],
                "score": episode["score"],
                "steps": episode["steps"],
                "terminated": episode["terminated"],
                "truncated": episode["truncated"],
                "mapped_slots_used": episode["mapped_slots_used"],
                "policy_ms_per_step": round(run["policy_ms_per_step"], 2),
                "wall_s": round(run["wall_s"], 2),
                "checked": bool(checked),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
