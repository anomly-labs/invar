#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
fake_llama.py — a stand-in for llama.cpp's `llama-cli`, used by the INVAR unit
suite so the inference / parse / re-execution code paths are testable WITHOUT a
real model or GPU. It emits llama.cpp-shaped stdout that invar.worldline.run_inference
knows how to parse: a `> <prompt>` echo line, a deterministic generation, then a
`[ Prompt: ... t/s ]` stats line whose numbers vary (as real llama.cpp's do).

Determinism: generation is sha1(prompt|seed) — same (prompt, seed) → same text,
which is exactly the property the "llamacpp-pinned-reexec-v0" profile relies on,
so re-execution verification ACCEPTS. Env hooks let tests drive the failure paths:

  FAKE_LLAMA_FAIL=1        exit non-zero with a stderr message (llama.cpp failed)
  FAKE_LLAMA_NOECHO=1      omit the prompt echo line (parse-error path)
  FAKE_LLAMA_FLAKY=<file>  append an incrementing counter to the generation, so the
                           SAME (prompt, seed) yields DIFFERENT text across runs
                           (drives the "re-execution output digest differs" REJECT)
"""
import hashlib
import os
import sys


def _arg(flag, default=None):
    a = sys.argv
    return a[a.index(flag) + 1] if flag in a and a.index(flag) + 1 < len(a) else default


def main():
    if os.environ.get("FAKE_LLAMA_FAIL"):
        sys.stderr.write("fake llama: forced failure (FAKE_LLAMA_FAIL)\n")
        return 3

    if "--list-devices" in sys.argv:
        sys.stdout.write("Available devices:\n  CUDA0: Fake GPU 9000 (1 MiB, 1 MiB free)\n")
        return 0

    prompt = _arg("-p", "")
    seed = _arg("--seed", "0")
    n = int(_arg("-n", "128") or "128")

    gen = "The answer is " + hashlib.sha1(f"{prompt}|{seed}".encode()).hexdigest()[:12]
    flaky = os.environ.get("FAKE_LLAMA_FLAKY")
    if flaky:
        c = 0
        if os.path.exists(flaky):
            c = int(open(flaky).read() or "0")
        c += 1
        open(flaky, "w").write(str(c))
        gen += f" [run {c}]"
    # crude token budget: keep the generation to ~n whitespace tokens
    gen = " ".join(gen.split()[:max(1, n)])

    out = sys.stdout
    out.write("main: build = fake (llama.cpp stand-in)\n")
    out.write("system_info: n_threads = " + (_arg("-t", "4") or "4") + "\n")
    if not os.environ.get("FAKE_LLAMA_NOECHO"):
        out.write("> " + prompt + "\n")
    out.write(gen + "\n")
    # stats line: numbers deliberately vary run-to-run, like real llama.cpp
    jitter = hashlib.sha1(os.urandom(8)).hexdigest()[:4]
    out.write(f"\n[ Prompt: 7 tokens | eval 3.{jitter} t/s ]\n")
    out.write("llama_perf_context_print: done\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
