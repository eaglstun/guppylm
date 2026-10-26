"""Targeted gray-WebP/E8 hybrid test on the v5 noun-cont model.

Set AQ_QUANT_CKPT=runs/aq-chat-v5-noun-cont/final.pt and AQ_QUANT_OUT
before launch. All byte counts are actual E8 code/scale tensor bytes plus actual
WebP container bytes and fp16 row scales, with fixed tensors counted once.
"""

import json
import io
import os
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.func import functional_call

from quantlib import (D, OUT, SD, SMALL, TARGETS, base_sd, encode_chat,
                      load_dialogues, model_from, pad, reply, score)
from stage19_color_weights import prepare, reconstruct


PAYLOAD = os.environ.get("AQ_HYBRID_E8_PAYLOAD", "runs/quant-e8p/generated-g128-tuned300.pt")
DECODED = os.environ.get("AQ_HYBRID_E8_DECODED", "runs/quant-e8p/generated-g128-probe.pt")
MAX_BYTES = int(os.environ.get("AQ_HYBRID_MAX_BYTES", "3570000"))
payload = torch.load(PAYLOAD, map_location="cpu", weights_only=False)
e8 = torch.load(DECODED, map_location="cpu", weights_only=False)["model"]
assert set(TARGETS) == set(payload["codes"]) == set(payload["scales"])
assert torch.equal(e8["tok_emb.weight"], e8["lm_head.weight"])


def nbytes(t):
    return t.numel() * t.element_size()


e8_target_bytes = {k: nbytes(payload["codes"][k]) + nbytes(payload["scales"][k]) for k in TARGETS}
fixed_bytes = (sum(nbytes(v) for v in payload["small"].values())
               + nbytes(payload["pos_codes"]) + nbytes(payload["pos_scales"])
               + nbytes(payload["base_grid_4bit"]))
e8_bytes = fixed_bytes + sum(e8_target_bytes.values())
assert e8_bytes > 0
webp = {}
for q in (70, 80):
    webp[q] = {}
    for k in TARGETS:
        img, scales, shape = prepare(SD[k], "gray")
        stream = io.BytesIO()
        Image.fromarray(img).save(stream, format="WEBP", quality=q, method=4)
        blob = stream.getvalue()
        pixels = np.asarray(Image.open(io.BytesIO(blob)).convert("L"))
        assert pixels.shape == img.shape
        image_bytes = len(blob)
        w = torch.from_numpy(reconstruct(pixels, scales, "gray", shape).copy())
        webp[q][k] = (w, image_bytes, scales.size * 2,
                      torch.from_numpy((pixels.astype(np.float32) - 128).copy()),
                      torch.from_numpy(scales.copy()), blob)


def evaluate(name, keys, quality, small="tuned", probe=False):
    t0 = time.time()
    sd = {k: v for k, v in e8.items() if k != "lm_head.weight"}
    delta = 0
    qualities = quality if isinstance(quality, dict) else {k: quality for k in keys}
    for k in keys:
        w, image_bytes, scale_bytes = webp[qualities[k]][k][:3]
        sd[k] = w
        delta += image_bytes + scale_bytes - e8_target_bytes[k]
    if small == "original":
        original = base_sd()
        for k in SMALL:
            sd[k] = original[k].half().float()
    bytes_ = e8_bytes + delta
    m = model_from(sd)
    loss = score(m)
    row = {"recipe": name, "quality": quality if isinstance(quality, int) else "mixed",
           "matrix_qualities": qualities, "keys": list(keys),
           "small": small, "bytes": bytes_, "MB": round(bytes_ / 1e6, 4),
           "delta_bytes": delta, "loss": round(loss, 6),
           "seconds": round(time.time() - t0, 2)}
    if probe:
        row["reply"] = reply(m)
    print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, "results.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def save_probe_checkpoint(label, keys, quality, small):
    sd = {k: v for k, v in e8.items() if k != "lm_head.weight"}
    qualities = quality if isinstance(quality, dict) else {k: quality for k in keys}
    for k in keys:
        sd[k] = webp[qualities[k]][k][0]
    if small == "original":
        original = base_sd()
        for k in SMALL:
            sd[k] = original[k].half().float()
    sd["lm_head.weight"] = sd["tok_emb.weight"]
    path = os.path.join(OUT, label + "-probe.pt")
    torch.save({"model": sd, "config": payload["config"]}, path)
    return path


def distill_best(steps=300):
    with open(os.path.join(OUT, "results.jsonl")) as f:
        rows = [json.loads(s) for s in f]
    candidate = min((r for r in rows if r["small"] == "original" and
                     r["bytes"] <= MAX_BYTES), key=lambda r: r["loss"])
    keys = candidate["keys"]
    qualities = candidate["matrix_qualities"]
    initial = {k: v for k, v in e8.items() if k != "lm_head.weight"}
    for k in keys:
        initial[k] = webp[qualities[k]][k][0]
    original = base_sd()
    for k in SMALL:
        initial[k] = original[k].half().float()
    teacher = model_from(base_sd())
    student = model_from(initial)
    student.train()
    for mod in student.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    log_scale = {k: torch.zeros_like(webp[qualities[k]][k][4], requires_grad=True) for k in keys}
    small = {k: initial[k].clone().requires_grad_(True) for k in SMALL}

    def weights(final=False):
        sd = dict(initial)
        for k in SMALL:
            sd[k] = small[k].half().float() if final else small[k]
        for k in keys:
            pixels = webp[qualities[k]][k][3]
            scale = webp[qualities[k]][k][4] * torch.exp(log_scale[k])
            if final:
                scale = scale.half().float()
            sd[k] = pixels * scale
        sd["lm_head.weight"] = sd["tok_emb.weight"]
        return sd

    train_rows = load_dialogues([os.path.join(D, "dialogues.jsonl")])
    encoded = [encode_chat(__import__("quantlib").tok, r, 512) for r in train_rows]
    rng = np.random.default_rng(1)
    opt = torch.optim.Adam([{"params": list(log_scale.values()), "lr": 2e-3},
                            {"params": list(small.values()), "lr": 2e-4}])
    t0 = time.time()
    milestones = []
    for step in range(1, steps + 1):
        warm = min(1.0, step / 50)
        lr_factor = warm * .5 * (1 + np.cos(np.pi * step / steps))
        opt.param_groups[0]["lr"] = 2e-3 * lr_factor
        opt.param_groups[1]["lr"] = 2e-4 * lr_factor
        X, _ = pad([encoded[i] for i in rng.choice(len(encoded), 8, replace=False)], 512)
        keep = X != 0
        with torch.no_grad():
            target = F.log_softmax(teacher(X)[0][keep], -1)
        got = F.log_softmax(functional_call(student, weights(), (X,))[0][keep], -1)
        loss = F.kl_div(got, target, log_target=True, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"tune step {step}: KL {loss.item():.4f}, elapsed {time.time()-t0:.0f}s", flush=True)
        if step in (100, 200, 300):
            snapshot = weights(final=True)
            snapshot_loss = score(model_from(snapshot))
            torch.save({"model": snapshot, "config": payload["config"]},
                       os.path.join(OUT, f"distilled-hybrid-step{step}-probe.pt"))
            milestone = {"step": step, "held_loss": round(snapshot_loss, 6),
                         "kl_batch": round(loss.item(), 6)}
            milestones.append(milestone)
            print("MILESTONE " + json.dumps(milestone), flush=True)
    final = weights(final=True)
    held = score(model_from(final))
    out = {"recipe": "distilled " + candidate["recipe"], "source": candidate["recipe"],
           "steps": steps, "bytes": candidate["bytes"], "MB": candidate["MB"],
           "loss": round(held, 6), "baseline_loss": candidate["loss"],
           "elapsed": round(time.time()-t0, 1), "milestones": milestones}
    print("DISTILLED " + json.dumps(out), flush=True)
    with open(os.path.join(OUT, "results.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")
    torch.save({"model": final, "config": payload["config"]},
               os.path.join(OUT, "distilled-hybrid-probe.pt"))
    encoded_webp = {k: {"bytes": webp[qualities[k]][k][5],
                        "scale": (webp[qualities[k]][k][4] * torch.exp(log_scale[k])).detach().half(),
                        "shape": tuple(SD[k].shape), "quality": qualities[k]} for k in keys}
    packed = {"webp": encoded_webp,
              "e8_codes": {k: v for k, v in payload["codes"].items() if k not in keys},
              "e8_scales": {k: v for k, v in payload["scales"].items() if k not in keys},
              "small": {k: v.detach().half() for k, v in small.items()},
              "pos_codes": payload["pos_codes"], "pos_scales": payload["pos_scales"],
              "base_grid_4bit": payload["base_grid_4bit"],
              "config": payload["config"], "rot": payload["rot"],
              "rotation_seed": payload["rotation_seed"],
              "rotation_block": payload["rotation_block"],
              "scale_group": payload["scale_group"], "codebook": payload["codebook"],
              "logical_bytes": candidate["bytes"], "held_loss": held}
    torch.save(packed, os.path.join(OUT, "distilled-hybrid-payload.pt"))
    return out


def pack_milestone(step):
    template = torch.load(os.path.join(OUT, "distilled-hybrid-payload.pt"),
                          map_location="cpu", weights_only=False)
    full = torch.load(os.path.join(OUT, f"distilled-hybrid-step{step}-probe.pt"),
                      map_location="cpu", weights_only=False)["model"]
    for k, w in template["webp"].items():
        q = webp[w["quality"]][k][3]
        nz = (q != 0).float().argmax(dim=1, keepdim=True)
        pix = q.gather(1, nz)
        assert torch.all(pix != 0), k
        inferred = (full[k].gather(1, nz) / pix).half()
        assert torch.equal(q * inferred.float(), full[k]), k
        w["scale"] = inferred
    template["small"] = {k: full[k].detach().half() for k in SMALL}
    with open(os.path.join(OUT, "results.jsonl")) as f:
        rows = [json.loads(x) for x in f]
    result = next(r for r in reversed(rows) if r.get("milestones"))
    template["held_loss"] = next(x["held_loss"] for x in result["milestones"] if x["step"] == step)
    path = os.path.join(OUT, f"distilled-hybrid-step{step}-payload.pt")
    torch.save(template, path)
    return path


def verify_packed(stem="distilled-hybrid"):
    from stage15_e8p import B, decode
    from stage1_scalar import rotations
    path = os.path.join(OUT, stem + "-payload.pt")
    p = torch.load(path, map_location="cpu", weights_only=False)
    full = torch.load(os.path.join(OUT, stem + "-probe.pt"),
                      map_location="cpu", weights_only=False)["model"]
    nib = torch.empty(256, 8, dtype=torch.uint8)
    nib[:, 0::2] = p["base_grid_4bit"] & 15
    nib[:, 1::2] = p["base_grid_4bit"] >> 4
    grid = (nib.float() - 8) / 2
    assert torch.equal(grid, B)
    got = {}
    for k in TARGETS:
        if k in p["webp"]:
            w = p["webp"][k]
            pixels = np.asarray(Image.open(io.BytesIO(w["bytes"])).convert("L"))
            got[k] = torch.from_numpy((pixels.astype(np.float32) - 128).copy()) * w["scale"].float()
        else:
            codes = p["e8_codes"][k]
            group = p["scale_group"] or SD[k].shape[1]
            Q = decode(codes, grid).reshape(SD[k].shape)
            Q = Q * p["e8_scales"][k].float().repeat_interleave(group, dim=1)
            U, R = rotations(k, "hadamard")
            got[k] = U @ Q @ R.T
    got["lm_head.weight"] = got["tok_emb.weight"]
    for k, v in p["small"].items():
        got[k] = v.float()
    got["pos_emb.weight"] = p["pos_codes"].float() * p["pos_scales"].float()[:, None]
    maximum = max((got[k] - full[k]).abs().max().item() for k in full)
    actual = (sum(len(v["bytes"]) + nbytes(v["scale"]) for v in p["webp"].values())
              + sum(nbytes(v) for v in p["e8_codes"].values())
              + sum(nbytes(v) for v in p["e8_scales"].values())
              + sum(nbytes(v) for v in p["small"].values())
              + nbytes(p["pos_codes"]) + nbytes(p["pos_scales"])
              + nbytes(p["base_grid_4bit"]))
    assert actual == p["logical_bytes"], (actual, p["logical_bytes"])
    assert maximum < 1e-5, maximum
    measured_loss = score(model_from(got))
    print(json.dumps({"decode_max_abs_difference": maximum,
                      "logical_bytes": actual, "torch_container_bytes": os.path.getsize(path),
                      "decoded_loss": measured_loss}), flush=True)


def pack_zip(stem="distilled-hybrid"):
    p = torch.load(os.path.join(OUT, stem + "-payload.pt"),
                   map_location="cpu", weights_only=False)
    dest = os.path.join(OUT, stem + "-payload.zip")
    manifest = {k: p[k] for k in ("config", "rot", "rotation_seed", "rotation_block",
                                   "scale_group", "codebook", "logical_bytes", "held_loss")}
    manifest["tensors"] = {}
    manifest["webp"] = {}
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as z:
        def add_tensor(category, key, t):
            member = f"{category}/{key}.bin"
            a = t.detach().contiguous().numpy()
            z.writestr(member, a.tobytes())
            manifest["tensors"][category + "/" + key] = {
                "member": member, "dtype": str(a.dtype), "shape": list(a.shape)}
        for category in ("e8_codes", "e8_scales", "small"):
            for k, v in p[category].items():
                add_tensor(category, k, v)
        for k in ("pos_codes", "pos_scales", "base_grid_4bit"):
            add_tensor("fixed", k, p[k])
        for k, w in p["webp"].items():
            name = f"webp/{k}.webp"
            z.writestr(name, w["bytes"])
            add_tensor("webp_scales", k, w["scale"])
            manifest["webp"][k] = {"member": name, "shape": list(w["shape"]),
                                   "quality": w["quality"]}
        z.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
    with zipfile.ZipFile(dest) as z:
        total = sum(x.file_size for x in z.infolist() if x.filename != "manifest.json")
        assert total == p["logical_bytes"], (total, p["logical_bytes"])
        assert z.testzip() is None
    print(json.dumps({"zip": dest, "file_bytes": os.path.getsize(dest),
                      "payload_bytes": total}), flush=True)


if __name__ == "__main__":
    byte_manifest = {"e8_bytes": e8_bytes, "fixed_bytes": fixed_bytes,
                     "fixed_parts": {"small_fp16": sum(nbytes(v) for v in payload["small"].values()),
                                     "pos_codes": nbytes(payload["pos_codes"]),
                                     "pos_scales": nbytes(payload["pos_scales"]),
                                     "base_grid": nbytes(payload["base_grid_4bit"])},
                     "webp": {str(q): {k: {"image_bytes": webp[q][k][1],
                                             "scale_bytes": webp[q][k][2],
                                             "e8_bytes": e8_target_bytes[k]}
                                         for k in TARGETS} for q in (70, 80)}}
    with open(os.path.join(OUT, "byte-manifest.json"), "w") as f:
        json.dump(byte_manifest, f, indent=2)
    if "--manifest" in sys.argv:
        sys.exit(0)
    if "--distill" in sys.argv:
        distill_best()
        sys.exit(0)
    if "--verify" in sys.argv:
        verify_packed()
        sys.exit(0)
    if "--pack-step100" in sys.argv:
        print(pack_milestone(100), flush=True)
        verify_packed("distilled-hybrid-step100")
        pack_zip("distilled-hybrid-step100")
        sys.exit(0)
    if "--pack-zip" in sys.argv:
        pack_zip()
        sys.exit(0)
    if "--refine" in sys.argv:
        core = (["tok_emb.weight"] +
                [k for k in TARGETS if ".attn.out." in k or ".ffn.up." in k])
        base = evaluate("refine base q80", core, 80, small="original")
        options = [k for k in TARGETS if k not in core]
        trials = []
        for k in options:
            qualities = {x: 80 for x in core}
            qualities[k] = 70
            trial = evaluate("add " + k + " q70", core + [k], qualities, small="original")
            trials.append(trial)
        feasible = [r for r in trials if r["bytes"] <= MAX_BYTES]
        best = min(feasible, key=lambda r: r["loss"])
        row = evaluate("best single + second", best["keys"], best["matrix_qualities"], small="original")
        remaining = [k for k in options if k not in best["keys"]]
        seconds = []
        for k in remaining:
            qualities = dict(best["matrix_qualities"])
            qualities[k] = 70
            expected = best["bytes"] + webp[70][k][1] + webp[70][k][2] - e8_target_bytes[k]
            if expected <= MAX_BYTES:
                seconds.append(evaluate("second add " + k + " q70", best["keys"] + [k], qualities, small="original"))
        all_rows = [base] + trials + seconds
        best_final = min((r for r in all_rows if r["bytes"] <= MAX_BYTES), key=lambda r: r["loss"])
        print("BEST_FEASIBLE " + json.dumps(best_final), flush=True)
        save_probe_checkpoint("best-hybrid", best_final["keys"], best_final["matrix_qualities"], "original")
        save_probe_checkpoint("small-hybrid", core, 70, "original")
        sys.exit(0)
    rows = []
    rows.append(evaluate("E8 baseline", (), 80))
    recipes = {"embedding": ["tok_emb.weight"],
               "qkv family": [k for k in TARGETS if ".attn.qkv." in k],
               "attention output family": [k for k in TARGETS if ".attn.out." in k],
               "ffn up family": [k for k in TARGETS if ".ffn.up." in k],
               "ffn down family": [k for k in TARGETS if ".ffn.down." in k],
               "layer0": [k for k in TARGETS if k.startswith("blocks.0.")]}
    for q in (70, 80):
        for name, keys in recipes.items():
            rows.append(evaluate(name, keys, q))
    # Greedy among disjoint whole-family choices. Two steps suffice to test
    # interaction without a combinatorial sweep.
    family = list(recipes.items())[:-1]
    for q in (70, 80):
        singles = [r for r in rows if r["quality"] == q and r["recipe"] in dict(family)]
        baseline = rows[0]["loss"]
        ranked = sorted(singles, key=lambda r: (baseline - r["loss"]) / max(r["delta_bytes"], 1), reverse=True)
        chosen = []
        for i, r in enumerate(ranked[:3], 1):
            chosen.extend(r["keys"])
            rows.append(evaluate("greedy " + "+".join(x["recipe"] for x in ranked[:i]), chosen, q))
    # Compare coadapted E8 small tensors against original fp16 values for the
    # best useful hybrid at each quality.
    for q in (70, 80):
        candidates = [r for r in rows if r["quality"] == q and r["delta_bytes"] > 0]
        best = min(candidates, key=lambda r: r["loss"])
        rows.append(evaluate("original small control: " + best["recipe"], best["keys"], q, small="original"))
