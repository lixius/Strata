#!/usr/bin/env python3
"""Fixed-prompt decode comparison: the SAME prompt text and generation length on both
servers, so the MTP draft acceptance is driven by the same content.

Usage: bench-decode-fixed.py <base_url> <tag>
Writes JSON results to $BENCH_OUT_DIR/bench-fixed-<tag>.json (default: the current directory).
"""
import json
import os
import statistics
import sys
import time

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
HDR = {}
if os.environ.get("STRATA_API_KEY"):
    HDR["Authorization"] = "Bearer " + os.environ["STRATA_API_KEY"]
TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
OUT = os.path.join(os.environ.get("BENCH_OUT_DIR", "."), "bench-fixed-%s.json" % TAG)

# The same prompt every time: a code-writing task, so the model generates a long deterministic answer.
PROMPT = (
    "Write a Python function that computes the nth Fibonacci number three ways: "
    "iteratively, with memoized recursion, and with matrix exponentiation. "
    "Include docstrings and a short example for each."
)
MAX_TOKENS = 1024


def call(prompt, max_tokens):
    t0 = time.time()
    r = requests.post(BASE + "/v1/chat/completions", json={
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }, headers=HDR, timeout=3600)
    r.raise_for_status()
    d = r.json()
    return d.get("timings", {}), time.time() - t0


def main():
    print("== fixed-prompt decode: %s ==" % TAG, flush=True)
    rows = []
    for i in range(3):
        t, wall = call(PROMPT, MAX_TOKENS)
        dn, da = t.get("draft_n") or 0, t.get("draft_n_accepted") or 0
        row = {
            "run": i,
            "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
            "predicted_n": t.get("predicted_n"),
            "prompt_per_second": t.get("prompt_per_second"),
            "predicted_per_second": t.get("predicted_per_second"),
            "predicted_per_token_ms": t.get("predicted_per_token_ms"),
            "draft_n": dn, "draft_accepted": da,
            "acceptance": round(da / dn, 3) if dn else None,
            "wall_s": round(wall, 2),
        }
        rows.append(row)
        print("  run%d prompt_n=%s cache=%s prefill=%s decode=%s tok/s (%.2f ms/tok) acc=%s wall=%.1fs" % (
            i, row["prompt_n"], row["cache_n"], row["prompt_per_second"],
            row["predicted_per_second"], row["predicted_per_token_ms"] or 0,
            row["acceptance"], wall), flush=True)

    def med(k):
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return round(statistics.median(v), 3) if v else None

    res = {"tag": TAG, "base": BASE, "max_tokens": MAX_TOKENS, "rows": rows,
           "median_decode": med("predicted_per_second"),
           "median_ms_per_token": med("predicted_per_token_ms"),
           "median_acceptance": med("acceptance"),
           "median_prefill": med("prompt_per_second")}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("median: decode=%s tok/s, %s ms/tok, acceptance=%s, prefill=%s" % (
        res["median_decode"], res["median_ms_per_token"], res["median_acceptance"], res["median_prefill"]))
    print("wrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
