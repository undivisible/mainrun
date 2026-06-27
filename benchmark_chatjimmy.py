#!/usr/bin/env python3
"""Benchmark the ChatJimmy API: measures tokens/sec, TTFT, total time
across short/medium/long prompts and streaming vs non-streaming."""

import json
import time
import sys
import requests

URL = "https://chatjimmy.ai/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.1-8B"

PROMPTS = {
    "short": "What is 2+2?",
    "medium": "Explain how a CPU works in a few paragraphs.",
    "long": (
        "Write a detailed essay about the history of computing, covering "
        "the abacus, mechanical calculators, vacuum tubes, transistors, "
        "integrated circuits, microprocessors, personal computers, the "
        "internet, and modern AI. Include at least 5 paragraphs."
    ),
}


def build_body(prompt, streaming):
    return {
        "messages": [{"role": "user", "content": prompt}],
        "chatOptions": {
            "selectedModel": MODEL,
            "systemPrompt": "",
            "topK": 8,
        },
        "attachment": None,
        "stream": streaming,
    }


def extract_stats(text):
    start = text.rfind("<|stats|>")
    end = text.rfind("<|/stats|>")
    if start == -1 or end == -1:
        return None
    raw = text[start + len("<|stats|>"):end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def run_streaming(prompt, label):
    body = build_body(prompt, True)
    t0 = time.perf_counter()
    ttft = None
    full_text = []
    try:
        resp = requests.post(URL, headers=HEADERS, json=body, stream=True, timeout=120)
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                full_text.append(chunk)
    except requests.RequestException as e:
        print(f"  [streaming/{label}] ERROR: {e}")
        return
    total = time.perf_counter() - t0
    text = "".join(full_text)
    stats = extract_stats(text)
    report("streaming", label, ttft, total, text, stats)


def run_non_streaming(prompt, label):
    body = build_body(prompt, False)
    t0 = time.perf_counter()
    try:
        resp = requests.post(URL, headers=HEADERS, json=body, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [non-streaming/{label}] ERROR: {e}")
        return
    total = time.perf_counter() - t0
    ttft = total  # no streaming, TTFT == total
    text = resp.text
    stats = extract_stats(text)
    report("non-streaming", label, ttft, total, text, stats)


def report(mode, label, ttft, total, text, stats):
    # Strip stats sentinel for token counting from text
    clean = text
    s = text.find("<|stats|>")
    if s != -1:
        clean = text[:s]
    gen_tokens = None
    tps = None
    prefill_rate = None
    server_ttft = None
    server_total = None
    if stats:
        for k in ("decode_rate", "tokensPerSecond", "tokens_per_second", "tps"):
            if k in stats:
                tps = stats[k]
                break
        for k in ("decode_tokens", "generationTokens", "generatedTokens", "completionTokens", "outputTokens"):
            if k in stats:
                gen_tokens = stats[k]
                break
        prefill_rate = stats.get("prefill_rate")
        server_ttft = stats.get("ttft")
        server_total = stats.get("total_time")
    # Fallback: estimate tokens from text (~4 chars/token)
    if gen_tokens is None:
        gen_tokens = max(1, len(clean) // 4)

    print(f"\n=== {mode} / {label} ===")
    print(f"  Client TTFT:        {ttft:.3f}s" if ttft else "  Client TTFT:        N/A")
    print(f"  Client total time:  {total:.3f}s")
    if server_ttft is not None:
        print(f"  Server TTFT:        {server_ttft*1000:.2f} ms")
    if server_total is not None:
        print(f"  Server gen time:    {server_total*1000:.2f} ms")
    print(f"  Generated tokens:   {gen_tokens}")
    if tps is not None:
        print(f"  Tokens/sec (decode): {tps:.2f}")
    if prefill_rate is not None:
        print(f"  Prefill tok/sec:    {prefill_rate:.2f}")
    print(f"  Response len:       {len(clean)} chars")
    print(f"  Preview:            {clean[:120].strip()!r}")


def main():
    print(f"ChatJimmy API benchmark — {URL} (model: {MODEL})")
    for label, prompt in PROMPTS.items():
        print(f"\n----- Prompt: {label} ({len(prompt)} chars) -----")
        run_streaming(prompt, label)
        run_non_streaming(prompt, label)


if __name__ == "__main__":
    main()
