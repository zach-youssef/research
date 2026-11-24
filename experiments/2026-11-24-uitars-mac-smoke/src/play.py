"""UI-TARS-1.5-7B (4-bit MLX) plays browser 2048 closed-loop. See ../plan.md.

Usage (via run.sh): play.py --runs-dir R --exp-dir E [--smoke]
"""
import argparse, hashlib, http.server, importlib.metadata as md, json, os, platform
import random, resource, socketserver, statistics, subprocess, sys, threading, time
from datetime import date
from pathlib import Path

EXP_ID = "2026-11-24-uitars-mac-smoke"
CLAIM = "wang2025gametars#C8"
MODEL_REPO = "mlx-community/UI-TARS-1.5-7B-4bit"
MODEL_REV = "cca0e360e25322cdf480cf841e301b6d55bce9e0"
MODEL_SHA256 = "75cc35783e9aca2f3a9b21594b01c402219d63978928718bfa16f175153d07d3"
VIEW_W, VIEW_H = 728, 812  # multiples of 28 (smart_resize is the identity); 812 px shows the whole page (board bottom at 774 px)
KEYS = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]
SEEDS = [0, 1, 2]
MAX_STEPS = 30
INSTRUCTION = ("You are playing the browser game 2048. Slide the tiles with the arrow keys (up, down, "
               "left, right; press one arrow key per step with hotkey) so that equal tiles merge, and "
               "reach the highest tile and score you can.")
SEED_JS = """(() => { let a = %d >>> 0; Math.random = function () { a = (a + 0x6D2B79F5) >>> 0;
 let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
 return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();"""


def serve(directory, port=0):
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k): super().__init__(*a, directory=str(directory), **k)
        def log_message(self, *a): pass
    srv = socketserver.TCPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Game:
    def __init__(self, browser, url, seed):
        self.ctx = browser.new_context(viewport={"width": VIEW_W, "height": VIEW_H})
        self.ctx.route("**/*", lambda r: r.continue_() if r.request.url.startswith("http://127.0.0.1") else r.abort())
        self.ctx.add_init_script(SEED_JS % (seed + 1))
        self.page = self.ctx.new_page()
        self.page.goto(url)
        self.page.wait_for_selector('.tile-container .tile')
        self.page.wait_for_timeout(400)

    def state(self):
        s = self.page.evaluate("() => JSON.parse(localStorage.getItem('gameState') || 'null')")
        if s is None:
            return None
        cells = [[(c or {}).get("value", 0) for c in row] for row in s["grid"]["cells"]]
        return {"cells": cells, "score": s["score"], "over": s["over"]}

    def shot(self):
        return self.page.screenshot()

    def press(self, key):
        self.page.keyboard.press(key)
        self.page.wait_for_timeout(300)

    def click(self, fx, fy):
        self.page.mouse.click(fx * VIEW_W, fy * VIEW_H)
        self.page.wait_for_timeout(300)

    def close(self):
        self.ctx.close()


def norm_key(raw):
    k = (raw or "").lower().replace("_", "").replace("-", "").replace(" ", "")
    k = k.replace("arrow", "")
    return {"left": "ArrowLeft", "right": "ArrowRight", "up": "ArrowUp", "down": "ArrowDown"}.get(k)


def execute(game, act, known):
    """Run a parsed UI-TARS action. Returns (parsed_ok, executed_kind)."""
    t = act["action_type"]
    if t not in known:
        return False, "unknown"
    inp = act["action_inputs"]
    if t == "hotkey":
        key = norm_key(inp.get("key") or inp.get("hotkey"))
        if key is None:
            return True, "hotkey_other"
        game.press(key)
        return True, "arrow"
    if t in ("click", "left_double", "right_single"):
        box = json.loads(inp["start_box"]) if "start_box" in inp else None
        if box:
            game.click(box[0], box[1])
        return True, "click"
    if t == "finished":
        return True, "finished"
    return True, "noop"  # wait, scroll, type, drag


def parses_patched(text, parse, known):
    """Diagnostic only (never executed): would the output parse after removing the Qwen2.5-VL box tokens
    (<|box_start|>, <|box_end|>) that mlx-vlm keeps in the text unless skip_special_tokens is set?"""
    t = text.replace("<|box_start|>", "").replace("<|box_end|>", "")
    try:
        acts = parse(t, factor=1000, origin_resized_height=VIEW_H, origin_resized_width=VIEW_W, model_type="qwen25vl")
        return acts[0]["action_type"] in known
    except Exception:
        return False


def peak_mem_gb():
    import mlx.core as mx
    mlx_peak = (mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory()) / 1e9
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    return mlx_peak, rss


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def strip(frames, path, h=200):
    from PIL import Image
    import io
    ims = [Image.open(io.BytesIO(b)).convert("RGB") for b in frames]
    ims = [i.resize((h, h)) for i in ims]
    out = Image.new("RGB", (h * len(ims), h))
    for n, i in enumerate(ims):
        out.paste(i, (n * h, 0))
    out.save(path)


def random_episode(game, seed, max_steps):
    rng = random.Random(seed)
    for _ in range(max_steps):
        s = game.state()
        if s and s["over"]:
            break
        game.press(rng.choice(KEYS))
    return game.state()["score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    runs, exp = Path(a.runs_dir), Path(a.exp_dir)
    out = runs / ("smoke" if a.smoke else "out")
    out.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[:1] if a.smoke else SEEDS
    max_steps = 1 if a.smoke else MAX_STEPS
    rand_steps = 10 if a.smoke else MAX_STEPS
    t_start = time.time()

    from ui_tars.action_parser import parse_action_to_structure_output, smart_resize
    from ui_tars.prompt import COMPUTER_USE_DOUBAO
    from playwright.sync_api import sync_playwright
    import mlx.core as mx
    from mlx_vlm import load, generate, apply_chat_template
    from huggingface_hub import snapshot_download
    assert smart_resize(VIEW_H, VIEW_W) == (VIEW_H, VIEW_W), "screenshot size must be a smart_resize fixed point"
    known = {"click", "left_double", "right_single", "drag", "hotkey", "type", "scroll", "wait", "finished"}

    # weights provenance (pinned revision; hash checked in full mode, size only in smoke)
    mpath = Path(snapshot_download(MODEL_REPO, revision=MODEL_REV))
    wfiles = sorted(p for p in mpath.iterdir() if p.suffix == ".safetensors" or p.name == "tokenizer.json")
    hashes = {p.name: (file_sha256(p.resolve()) if not a.smoke else f"size={p.resolve().stat().st_size}") for p in wfiles}
    if not a.smoke:
        assert hashes["model.safetensors"] == MODEL_SHA256, hashes

    t0 = time.time()
    model, processor = load(str(mpath))
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.1f}s", flush=True)

    srv = serve(runs / "2048")
    url = f"http://127.0.0.1:{srv.server_address[1]}/index.html"
    steps_log, episodes = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        for seed in seeds:
            g = Game(browser, url, seed)
            frames = []
            for step in range(max_steps):
                s0 = g.state()
                if s0 and s0["over"]:
                    break
                png = g.shot()
                frames.append(png)
                img_path = out / "current.png"
                img_path.write_bytes(png)
                prompt = COMPUTER_USE_DOUBAO.format(language="English", instruction=INSTRUCTION)
                formatted = apply_chat_template(processor, model.config, prompt, num_images=1)
                t1 = time.time()
                res = generate(model, processor, formatted, image=[str(img_path)], max_tokens=300,
                               temperature=0.0, verbose=False, skip_special_tokens=True)  # as the official TGI deploy path decodes
                infer_s = time.time() - t1
                text = res.text if hasattr(res, "text") else str(res)
                parsed, kind, err = False, None, None
                try:
                    acts = parse_action_to_structure_output(text, factor=1000, origin_resized_height=VIEW_H,
                                                            origin_resized_width=VIEW_W, model_type="qwen25vl")
                    parsed, kind = execute(g, acts[0], known)
                    atype = acts[0]["action_type"]
                except Exception as e:  # unparsable output counts as a parse failure
                    err, atype = repr(e)[:200], None
                patched_ok = parses_patched(text, parse_action_to_structure_output, known)
                s1 = g.state()
                changed = bool(s0 and s1 and s0["cells"] != s1["cells"])
                steps_log.append({"seed": seed, "step": step, "infer_s": infer_s, "prompt_tokens": res.prompt_tokens,
                                  "gen_tokens": res.generation_tokens, "prompt_tps": res.prompt_tps,
                                  "gen_tps": res.generation_tps, "parsed": parsed, "parsed_patched": patched_ok, "action": atype, "kind": kind,
                                  "changed": changed, "score": s1["score"] if s1 else None, "text": text, "error": err})
                print(f"seed {seed} step {step}: {infer_s:.1f}s {atype} {kind} changed={changed} score={s1 and s1['score']}", flush=True)
                if kind == "finished":
                    break
            frames.append(g.shot())
            st = g.state()
            episodes.append({"seed": seed, "score": st["score"], "max_tile": max(max(r) for r in st["cells"]),
                             "over": st["over"], "steps": len([x for x in steps_log if x["seed"] == seed])})
            strip(frames[:: max(1, len(frames) // 12)], out / f"strip_seed{seed}.png")
            g.close()
        rand_scores = []
        for seed in seeds:
            g = Game(browser, url, seed)
            rand_scores.append(random_episode(g, seed, rand_steps))
            g.close()
        browser.close()
    srv.shutdown()

    mlx_peak, rss = peak_mem_gb()
    n = len(steps_log)
    infer = [x["infer_s"] for x in steps_log]
    m = lambda v, n_=1, unit="1": {"value": v, "unit": unit, "n": n_}
    mean_model = statistics.mean(e["score"] for e in episodes)
    mean_rand = statistics.mean(rand_scores)
    parse_rate = sum(x["parsed"] for x in steps_log) / n
    change_rate = sum(x["changed"] for x in steps_log) / n
    patched_rate = sum(x["parsed_patched"] for x in steps_log) / n
    click_rate = sum(x["kind"] == "click" or (x["kind"] is None and "Action: click" in x["text"]) for x in steps_log) / n
    arrow_rate = sum(x["kind"] == "arrow" for x in steps_log) / n
    metrics = {
        "parse_rate": m(parse_rate, n), "board_change_rate": m(change_rate, n),
        "arrow_action_rate": m(arrow_rate, n), "click_action_rate": m(click_rate, n),
        "parse_rate_after_box_token_strip": m(patched_rate, n),
        "mean_score_model": m(mean_model, len(episodes)), "mean_score_random": m(mean_rand, len(rand_scores)),
        "ms_per_step_mean": m(1000 * statistics.mean(infer), n, "ms"),
        "ms_per_step_median": m(1000 * statistics.median(infer), n, "ms"),
        "gen_tokens_per_step_mean": m(statistics.mean(x["gen_tokens"] for x in steps_log), n),
        "prompt_tokens_per_step_mean": m(statistics.mean(x["prompt_tokens"] for x in steps_log), n),
        "gen_tokens_per_second_mean": m(statistics.mean(x["gen_tps"] for x in steps_log), n),
        "load_time_s": m(load_s, 1, "s"), "peak_mem_mlx_gb": m(mlx_peak, 1, "GB"), "peak_rss_gb": m(rss, 1, "GB"),
    }
    observed = (f"parse_rate {parse_rate:.3f}, board_change_rate {change_rate:.3f}, mean_score_model "
                f"{mean_model:.1f} vs mean_score_random {mean_rand:.1f} (n={len(episodes)} seeds, {n} steps)")
    if parse_rate >= 0.9 and change_rate >= 0.5 and mean_model > mean_rand: would = "would be supports"
    elif parse_rate < 0.5 or change_rate < 0.1: would = "would be contradicts"
    else: would = "would be inconclusive"
    result = {
        "episodes": episodes, "random_scores": rand_scores, "weights_sha256": hashes,
        "model": {"repo": MODEL_REPO, "revision": MODEL_REV}, "steps": steps_log,
    }
    (out / "play_log.json").write_text(json.dumps(result, indent=1))
    print("METRICS", observed, "|", would, flush=True)
    if a.smoke:
        print(f"smoke ok: wall {time.time() - t_start:.1f}s; first output: {steps_log[0]['text'][:200]!r}")
        return

    def rp(path):
        return subprocess.run(["git", "rev-parse", f"HEAD:experiments/{EXP_ID}/{path}"], cwd=exp, capture_output=True,
                              text=True, check=True).stdout.strip()
    git = lambda *x: subprocess.run(["git", *x], cwd=exp, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool([l for l in git("status", "--porcelain").splitlines()
                  if not any(o in l for o in (f"{EXP_ID}/results.", f"{EXP_ID}/figures"))])
    res_json = {
        "schema": 1, "experiment": EXP_ID, "date": str(date.today()),
        "commits": [{"repo": "research", "sha": git("rev-parse", "HEAD"), "dirty": dirty}],
        "inputs": {"src": rp("src"), "run.sh": rp("run.sh"), "requirements.txt": rp("requirements.txt")},
        "exploratory": True, "tests": [CLAIM],
        "verdicts": {CLAIM: {"verdict": "inconclusive", "evidence": f"exploratory run, capped at inconclusive; {observed}; criterion on these values {would}"}},
        "metrics": metrics, "seeds": seeds, "wall_clock_s": time.time() - t_start,
        "env": {"python": platform.python_version(), "os": f"macOS {platform.mac_ver()[0]} {platform.machine()}",
                "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions()),
                "gpu": "Apple M4 (Metal via MLX), 16 GB unified memory"},
        "provenance": {"model": MODEL_REPO, "revision": MODEL_REV, "weights_sha256": hashes,
                       "episodes": episodes, "random_scores": rand_scores},
    }
    (exp / "results.json").write_text(json.dumps(res_json, indent=2) + "\n")
    print("wrote results.json")


if __name__ == "__main__":
    main()
