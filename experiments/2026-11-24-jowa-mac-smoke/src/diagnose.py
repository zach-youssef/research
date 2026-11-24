"""CPU-only diagnostics for the JOWA port (run: `run.sh --diagnose`). Not part of the plan's
criterion; it checks what the 32-cell run cannot: that the ALE v5 shim's action sets equal
upstream's tables, how well the frozen tokenizer reconstructs the port's frames (and whether a
different grayscale conversion changes that), the per-cell action histograms of greedy play with
the tuned and the default context length, and the parameter counts without training-only modules.
Writes ../diagnostics.json and ../figures/diag_<game>.png."""
import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

CACHE = Path.home() / ".cache" / "jowa"
GAMES = ["Assault", "Seaquest"]
SIZES = ["JOWA_40M", "JOWA_150M"]
SEEDS = [0, 1]
STEPS = 300
FRAME_STEPS = [20, 60, 100, 150, 200, 300, 400, 500]
THREADS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--jowa-dir", required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    exp, jowa = Path(args.exp_dir), Path(args.jowa_dir)
    sys.path.insert(0, str(exp / "src"))
    import jowa_patch
    state = jowa_patch.install(str(jowa / "src"))
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    import action_tokenizer as at
    from agent import Agent
    from envs import AtariEnvWrapper, SingleProcessEnv
    from game import AgentEnv
    from models.jowa_model import JOWAModel
    from utils import set_seed
    torch.set_num_threads(THREADS)
    out = {"torch_threads": torch.get_num_threads()}

    # (a) action meanings of the ALE v5 shim vs upstream's tables
    out["action_meanings"] = {}
    for g in GAMES:
        state["seed"] = 0
        e = AtariEnvWrapper(g).create_env()
        names = list(e._env.environment.env.get_action_meanings())
        ref = list(at._LIMITED_ACTION_SET[g])
        assert names == ref, f"{g}: ALE {names} != upstream {ref}"
        out["action_meanings"][g] = {"ale": names, "upstream": ref, "equal": True}
        e._env.close()

    def load(size):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(jowa / "config"), version_base="1.1"):
            cfg = compose(config_name="eval", overrides=[f"transformer={size}", f"critic_head={size}"])
        tok = instantiate(cfg.tokenizer)
        model = JOWAModel(obs_vocab_size=cfg.tokenizer.vocab_size, act_vocab_size=at.ATARI_NUM_ACTIONS,
                          config_transformer=instantiate(cfg.transformer), config_critic_arch=cfg.critic_head,
                          config_critic_train=cfg.action, device=torch.device("cpu"))
        a = Agent(tok, model, 0, 1, "float32", 8, torch.device("cpu")).eval()
        jowa_patch.load_checkpoint(a, str(CACHE), size)
        return cfg, tok, model

    # (b) parameter counts without training-only modules
    models = {}
    out["params"] = {}
    for size in SIZES:
        cfg, tok, model = load(size)
        models[size] = (cfg, tok, model)
        n = lambda ps: sum(p.numel() for p in ps)
        tok_total = n(tok.parameters())
        tok_lpips = n(p for k, p in tok.named_parameters() if k.startswith("lpips."))
        mod_total = n(model.parameters())
        mod_tgt = n(p for k, p in model.named_parameters() if k.startswith("head_q_target"))
        out["params"][size] = dict(model_total=mod_total, head_q_target=mod_tgt, model_inference=mod_total - mod_tgt,
                                   tokenizer_total=tok_total, tokenizer_lpips=tok_lpips,
                                   tokenizer_encoder_decoder_etc=tok_total - tok_lpips)

    # (c) tokenizer reconstruction on the port's frames, ALE grayscale vs an RGB->gray alternative
    tok = models[SIZES[0]][1].eval()
    fig_dir = exp / "figures"
    fig_dir.mkdir(exist_ok=True)
    out["reconstruction"] = {}

    def recon(frame):  # frame: (84, 84) uint8
        x = torch.from_numpy(frame)[None, None].float() / 255
        with torch.no_grad():
            enc = tok.encode(x, should_preprocess=True)
            rec = tok.decode(enc.z_quantized, should_postprocess=True).clamp(0, 1)
        return x, rec, int(enc.tokens.unique().numel())

    for g in GAMES:
        state["seed"] = 0
        e = AtariEnvWrapper(g).create_env()
        obs = e.reset()
        rng = np.random.RandomState(0)
        rows, orig_imgs, rec_imgs, alt_imgs = [], [], [], []
        for t in range(1, max(FRAME_STEPS) + 1):
            obs, _, d, _ = e.step(int(rng.choice(at.LIMITED_ACTION_TO_FULL_ACTION[g])))
            if d:
                break
            if t in FRAME_STEPS:
                x, rec, ntok = recon(obs[..., 0])
                rgb = e._env.environment.ale.getScreenRGB()
                gray2 = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)
                x2, rec2, _ = recon(gray2)
                rows.append(dict(step=t, recon_mse=float(((rec - x) ** 2).mean()), frame_variance=float(x.var()),
                                 frame_mean=float(x.mean()), frac_pixels_above_0p1=float((x > 0.1).float().mean()),
                                 distinct_tokens=ntok,
                                 alt_gray_recon_mse=float(((rec2 - x2) ** 2).mean()), alt_gray_frame_variance=float(x2.var()),
                                 mean_abs_diff_ale_gray_vs_cv2_gray=float(np.abs(obs[..., 0].astype(float) - gray2).mean())))
                u8 = lambda a: (a.squeeze().numpy() * 255).astype(np.uint8)
                orig_imgs.append(u8(x)); rec_imgs.append(u8(rec)); alt_imgs.append(gray2)
        e._env.close()
        rr = [r["recon_mse"] / r["frame_variance"] for r in rows]
        out["reconstruction"][g] = dict(frames=rows, mean_recon_mse_over_variance=float(np.mean(rr)),
                                        mean_alt_recon_mse_over_variance=float(np.mean(
                                            [r["alt_gray_recon_mse"] / r["alt_gray_frame_variance"] for r in rows])))
        # montage: rows = original (ALE grayscale), reconstruction, RGB->gray alternative; x3 nearest upscale
        mont = np.concatenate([np.concatenate(rowimgs, axis=1) for rowimgs in (orig_imgs, rec_imgs, alt_imgs)], axis=0)
        Image.fromarray(mont).resize((mont.shape[1] * 2, mont.shape[0] * 2), Image.NEAREST).save(fig_dir / f"diag_{g.lower()}.png")

    # (d) greedy cells with action histograms: default context 8 and the repo's tuned per-game context
    plan_hp = {s: json.load(open(jowa / "eval" / f"{s}_raw_scores.json")) for s in SIZES}
    out["greedy"] = []
    for size in SIZES:
        cfg, tok_, model = models[size]
        for g in GAMES:
            names = at._LIMITED_ACTION_SET[g]
            legal_full = at.LIMITED_ACTION_TO_FULL_ACTION[g]
            tuned = int(plan_hp[size][g]["buffer_size"][0])
            for buf in sorted({8, tuned}, reverse=True):
                for seed in SEEDS:
                    state["seed"] = seed
                    set_seed(seed)
                    env = SingleProcessEnv(AtariEnvWrapper(g).create_env)
                    agent = Agent(tok_, model, at.GAME_NAMES.index(g), 1, "float32", buf, torch.device("cpu")).eval()
                    aenv = AgentEnv(agent, env, "atari", False, verbose=False)
                    acts = []
                    orig = agent.act
                    agent.act = lambda o, orig=orig: (lambda a: (acts.append(int(a[0])), a)[1])(orig(o))
                    aenv.reset()
                    steps, done = 0, False
                    while steps < STEPS and not done:
                        _, _, dn, info = aenv.step()
                        done = bool(dn[0]); steps += 1
                    env.env._env.close()
                    hist = Counter(names[list(legal_full).index(a)] for a in acts)
                    out["greedy"].append(dict(size=size, game=g, context=buf, tuned=buf == tuned and buf != 8,
                                              seed=seed, steps=steps, ret=float(info["return"][0]),
                                              clipped_return=float(info["clipped_return"][0]),
                                              action_histogram=dict(hist)))
                    print(json.dumps(out["greedy"][-1]), flush=True)
    (exp / "diagnostics.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote diagnostics.json", flush=True)


if __name__ == "__main__":
    main()
