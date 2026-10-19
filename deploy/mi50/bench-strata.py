#!/usr/bin/env python3
"""Strata speed comparison: same prompts, same params, one server at a time.

Usage: bench-strata.py <base_url> <tag>
Writes JSON results to $BENCH_OUT_DIR/bench-<tag>.json (default: the current directory).
"""
import json
import os
import random
import statistics
import string
import sys
import time

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
OUT = os.path.join(os.environ.get("BENCH_OUT_DIR", "."), "bench-%s.json" % TAG)

UNIT = "The quick brown fox jumps over the lazy dog and then it rests for a while. "


def make_prompt(approx_tokens):
    """A fresh prompt each call: a random nonce keeps the prefix cache from serving it."""
    nonce = "".join(random.choices(string.ascii_lowercase, k=16))
    n = max(1, approx_tokens // 13)
    return "[%s] %s" % (nonce, UNIT * n)


def call(prompt, max_tokens):
    t0 = time.time()
    r = requests.post(BASE + "/v1/chat/completions", json={
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }, timeout=3600)
    r.raise_for_status()
    wall = time.time() - t0
    d = r.json()
    return d.get("timings", {}), d.get("usage", {}), wall


def scenario(name, approx_tokens, max_tokens, repeats):
    rows = []
    for i in range(repeats):
        prompt = make_prompt(approx_tokens)
        t, u, wall = call(prompt, max_tokens)
        row = {
            "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
            "prompt_per_second": t.get("prompt_per_second"),
            "predicted_n": t.get("predicted_n"),
            "predicted_per_second": t.get("predicted_per_second"),
            "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
            "wall_s": round(wall, 2),
        }
        rows.append(row)
        print("  %-8s run%d prompt_n=%s cache=%s prefill=%s tok/s  decode=%s tok/s  wall=%.1fs" % (
            name, i, row["prompt_n"], row["cache_n"], row["prompt_per_second"],
            row["predicted_per_second"], wall), flush=True)
    def med(k):
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return round(statistics.median(v), 2) if v else None
    return {
        "name": name, "approx_tokens": approx_tokens, "max_tokens": max_tokens,
        "repeats": repeats, "rows": rows,
        "median_prompt_per_second": med("prompt_per_second"),
        "median_predicted_per_second": med("predicted_per_second"),
    }


def main():
    print("== %s ==" % TAG, flush=True)
    print("warmup ...", flush=True)
    call(make_prompt(64), 16)

    results = {"tag": TAG, "base": BASE, "scenarios": []}
    for name, ptok, mtok, reps in [
        ("prefill4k", 4096, 1, 3),
        ("decode", 512, 256, 3),
    ]:
        print("%s ..." % name, flush=True)
        results["scenarios"].append(scenario(name, ptok, mtok, reps))

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
