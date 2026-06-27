#!/usr/bin/env python3
"""Benchmark training and inference speed: PyTorch vs MLX-Python vs C++/MLX."""
import json, time, sys, os, subprocess, argparse

def benchmark_pytorch(data_path, tokenizer_path, steps=200, batch_size=32, block_size=256):
    """Benchmark PyTorch training speed."""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mainrun'))
    from model import GPT, GPTConfig

    with open(data_path) as f:
        d = json.load(f)
    train_ids = torch.tensor(d['train_ids'], dtype=torch.long)
    val_ids = torch.tensor(d['val_ids'], dtype=torch.long)
    val_char_count = d['val_char_count']

    cfg = GPTConfig(
        vocab_size=24000, block_size=block_size, n_layer=8, n_head=9, d_model=576,
        dropout=0.15, eos_id=1, qk_norm=True, rope_theta=1000.0
    )
    device = torch.device('mps')
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    from torch.optim._muon import Muon
    muon_params, adamw_params = model.get_optimizer_param_groups()
    muon_opt = Muon(muon_params, lr=0.02, weight_decay=0.1, momentum=0.95)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=5e-4, weight_decay=0.1, betas=(0.9, 0.95))

    max_steps = steps
    warmup = max(1, int(max_steps * 0.05))
    decay_start = max_steps - max(1, int(max_steps * 0.20))
    decay_len = max(1, max_steps - decay_start)
    def lam(step):
        if step < warmup: return (step+1)/warmup
        elif step < decay_start: return 1.0
        else: return max(0.0, 1.0 - (step - decay_start + 1) / decay_len)

    ptr = 0
    span = block_size * batch_size + 1
    model.train()

    # Warmup steps (not counted)
    for step in range(min(5, steps)):
        lr = 0.02 * lam(step)
        for g in muon_opt.param_groups: g['lr'] = lr
        for g in adamw_opt.param_groups: g['lr'] = 5e-4 * lam(step)
        if ptr + span >= len(train_ids): ptr = 0
        batch = train_ids[ptr:ptr+span].to(device)
        x = batch[:-1].view(batch_size, block_size)
        y = batch[1:].view(batch_size, block_size)
        ptr += block_size * batch_size
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon_opt.step(); adamw_opt.step()
        muon_opt.zero_grad(); adamw_opt.zero_grad()
    torch.mps.synchronize()

    # Timed steps
    t0 = time.time()
    total_loss = 0.0
    for step in range(steps):
        lr = 0.02 * lam(step)
        for g in muon_opt.param_groups: g['lr'] = lr
        for g in adamw_opt.param_groups: g['lr'] = 5e-4 * lam(step)
        if ptr + span >= len(train_ids): ptr = 0
        batch = train_ids[ptr:ptr+span].to(device)
        x = batch[:-1].view(batch_size, block_size)
        y = batch[1:].view(batch_size, block_size)
        ptr += block_size * batch_size
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon_opt.step(); adamw_opt.step()
        muon_opt.zero_grad(); adamw_opt.zero_grad()
        total_loss += loss.item()
    torch.mps.synchronize()
    elapsed = time.time() - t0

    return {
        'framework': 'PyTorch (MPS)',
        'params': n_params,
        'steps': steps,
        'elapsed_s': elapsed,
        'ms_per_step': elapsed / steps * 1000,
        'steps_per_sec': steps / elapsed,
        'avg_train_loss': total_loss / steps,
    }


def benchmark_mlx_python(data_path, tokenizer_path, steps=200, batch_size=32, block_size=256):
    """Benchmark MLX Python training speed."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np

    with open(data_path) as f:
        d = json.load(f)
    train_ids = mx.array(d['train_ids'], dtype=mx.uint32)
    val_ids = mx.array(d['val_ids'], dtype=mx.uint32)

    # Build model matching our C++ architecture
    vocab_size = 24000
    n_layer, n_head, d_model = 8, 9, 576
    head_dim = d_model // n_head
    hidden = int(8.0/3.0 * d_model)
    dropout = 0.15
    eos_id = 1
    rope_theta = 1000.0

    # Simple GPT model in MLX
    class MLXGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.ln_f_w = mx.ones((d_model,))
            self.blocks = []
            for _ in range(n_layer):
                b = {
                    'attn_norm_w': mx.ones((d_model,)),
                    'qkv_w': mx.random.normal((3*d_model, d_model)) * 0.02,
                    'q_norm_w': mx.ones((head_dim,)),
                    'k_norm_w': mx.ones((head_dim,)),
                    'proj_w': mx.random.normal((d_model, d_model)) * (0.02 / (2*n_layer)**0.5),
                    'mlp_norm_w': mx.ones((d_model,)),
                    'w_gate': mx.random.normal((hidden, d_model)) * 0.02,
                    'w_up': mx.random.normal((hidden, d_model)) * 0.02,
                    'w_out': mx.random.normal((d_model, hidden)) * (0.02 / (2*n_layer)**0.5),
                }
                self.blocks.append(b)

        def rmsnorm(self, x, w):
            ms = mx.mean(x*x, axis=-1, keepdims=True)
            return w * mx.rsqrt(ms + 1e-5) * x

        def __call__(self, idx, train=False):
            B, T = idx.shape
            x = self.token_emb(idx)
            if train: x = mx.dropout(x, dropout)

            # RoPE
            inv_freq = 1.0 / (rope_theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
            freqs = mx.outer(mx.arange(T, dtype=mx.float32), inv_freq)
            emb = mx.concatenate([freqs, freqs], axis=-1)
            cos_v = mx.cos(emb).reshape(1, 1, T, head_dim)
            sin_v = mx.sin(emb).reshape(1, 1, T, head_dim)

            # Mask
            eos = (idx == eos_id).astype(mx.float32)
            seg = mx.cumsum(eos, axis=-1) - eos
            same_seg = mx.equal(seg[:, :, None], seg[:, None, :])
            causal = mx.tril(mx.ones((T, T), dtype=mx.bool_))
            allowed = mx.logical_and(causal, same_seg)
            mask = mx.where(allowed, 0.0, -1e9)
            mask = mask[None, None, :, :]

            scale = 1.0 / (head_dim ** 0.5)
            for b in self.blocks:
                h = self.rmsnorm(x, b['attn_norm_w'])
                h_flat = h.reshape(B*T, d_model)
                qkv = (h_flat @ b['qkv_w'].T).reshape(B, T, 3*d_model)
                q, k, v = mx.split(qkv, 3, axis=-1)
                q = q.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
                k = k.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
                v = v.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)

                # RoPE
                qh, ql = q[..., :head_dim//2], q[..., head_dim//2:]
                q = mx.concatenate([-ql, qh], axis=-1)
                q = q * cos_v + mx.concatenate([qh, ql], axis=-1) * sin_v
                kh, kl = k[..., :head_dim//2], k[..., head_dim//2:]
                k = mx.concatenate([-kl, kh], axis=-1)
                k = k * cos_v + mx.concatenate([kh, kl], axis=-1) * sin_v

                # QK norm
                q = self.rmsnorm(q, b['q_norm_w'])
                k = self.rmsnorm(k, b['k_norm_w'])

                scores = (q @ k.transpose(0, 1, 3, 2)) * scale + mask
                weights = mx.softmax(scores, axis=-1)
                if train: weights = mx.dropout(weights, dropout)
                attn = weights @ v
                attn = attn.transpose(0, 2, 1, 3).reshape(B*T, d_model)
                attn = (attn @ b['proj_w'].T).reshape(B, T, d_model)
                if train: attn = mx.dropout(attn, dropout)

                m = self.rmsnorm(x, b['mlp_norm_w'])
                m_flat = m.reshape(B*T, d_model)
                gate = (m_flat @ b['w_gate'].T).reshape(B, T, hidden)
                up = (m_flat @ b['w_up'].T).reshape(B, T, hidden)
                silu = gate * mx.sigmoid(gate)
                out = (silu * up).reshape(B*T, hidden) @ b['w_out'].T
                out = out.reshape(B, T, d_model)
                if train: out = mx.dropout(out, dropout)
                x = x + attn + out

            x = self.rmsnorm(x, self.ln_f_w)
            logits = x.reshape(B*T, d_model) @ self.token_emb.weight.T
            return logits.reshape(B, T, vocab_size)

    model = MLXGPT()
    n_params = sum(p.size for p in model.parameters() if hasattr(p, 'size'))

    def loss_fn(idx, targets):
        logits = model(idx, train=True)
        lse = mx.logsumexp(logits, axis=-1, keepdims=True)
        log_probs = logits - lse
        picked = mx.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
        return -mx.mean(picked)

    def step_fn(idx, targets):
        loss, grads = mx.value_and_grad(loss_fn)(idx, targets)
        # Simple SGD update (not Muon, just for speed comparison)
        for p, g in zip(model.parameters(), grads):
            if hasattr(p, 'shape') and len(p.shape) >= 1:
                p -= 0.01 * g
        return loss

    # Compile the step function
    step_fn_compiled = mx.compile(step_fn)

    ptr = 0
    span = block_size * batch_size + 1

    # Warmup
    for _ in range(5):
        if ptr + span >= len(train_ids): ptr = 0
        batch = train_ids[ptr:ptr+span]
        x = batch[:-1].reshape(batch_size, block_size)
        y = batch[1:].reshape(batch_size, block_size)
        ptr += block_size * batch_size
        loss = step_fn_compiled(x, y)
        mx.eval(loss)

    # Timed
    t0 = time.time()
    total_loss = 0.0
    for s in range(steps):
        if ptr + span >= len(train_ids): ptr = 0
        batch = train_ids[ptr:ptr+span]
        x = batch[:-1].reshape(batch_size, block_size)
        y = batch[1:].reshape(batch_size, block_size)
        ptr += block_size * batch_size
        loss = step_fn_compiled(x, y)
        mx.eval(loss)
        total_loss += float(loss)
    elapsed = time.time() - t0

    return {
        'framework': 'MLX Python (compiled)',
        'params': n_params,
        'steps': steps,
        'elapsed_s': elapsed,
        'ms_per_step': elapsed / steps * 1000,
        'steps_per_sec': steps / elapsed,
        'avg_train_loss': total_loss / steps,
    }


def benchmark_cpp_inference(checkpoint_path, tokenizer_path, prompt, max_tokens=100):
    """Benchmark C++/MLX inference with KV cache."""
    result = subprocess.run(
        ['./build/train', 'infer', '--checkpoint', checkpoint_path,
         '--tokenizer', tokenizer_path, '--prompt', prompt,
         '--max-tokens', str(max_tokens), '--benchmark'],
        capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), 'mlx-cpp')
    )
    output = result.stdout + result.stderr
    # Parse output
    info = {}
    for line in output.split('\n'):
        if 'Prompt tokens:' in line: info['prompt_tokens'] = int(line.split(':')[1].strip())
        elif 'Generated:' in line: info['generated'] = int(line.split(':')[1].strip())
        elif 'Prefill:' in line: info['prefill_ms'] = float(line.split(':')[1].replace('ms','').strip())
        elif 'Total:' in line: info['total_s'] = float(line.split(':')[1].replace('s','').strip())
        elif 'Tokens/sec:' in line: info['tps'] = float(line.split(':')[1].strip())
    info['framework'] = 'C++/MLX (KV cache)'
    return info


def benchmark_pytorch_inference(data_path, tokenizer_path, prompt, max_tokens=100):
    """Benchmark PyTorch inference (no KV cache, full forward each step)."""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mainrun'))
    from model import GPT, GPTConfig

    cfg = GPTConfig(
        vocab_size=24000, block_size=256, n_layer=8, n_head=9, d_model=576,
        dropout=0.0, eos_id=1, qk_norm=True, rope_theta=1000.0
    )
    device = torch.device('mps')
    model = GPT(cfg).to(device)
    model.eval()

    # Tokenize
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    tokens = tok.encode(prompt).ids
    if not tokens: tokens = [1]
    prompt_tokens = len(tokens)

    V = cfg.vocab_size
    block_size = cfg.block_size

    # Warmup
    with torch.no_grad():
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(idx, None)

    # Timed
    t0 = time.time()
    generated = 0
    with torch.no_grad():
        for _ in range(max_tokens):
            if len(tokens) > block_size:
                tokens = tokens[-block_size:]
            idx = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(idx, None)
            next_token = logits[0, -1].argmax().item()
            tokens.append(next_token)
            generated += 1
            if next_token == cfg.eos_id:
                break
    torch.mps.synchronize()
    elapsed = time.time() - t0

    return {
        'framework': 'PyTorch (MPS, no KV cache)',
        'prompt_tokens': prompt_tokens,
        'generated': generated,
        'total_s': elapsed,
        'tps': generated / elapsed if elapsed > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--infer-tokens', type=int, default=100)
    parser.add_argument('--prompt', type=str, default='The future of AI is going to be')
    parser.add_argument('--skip-pytorch', action='store_true')
    parser.add_argument('--skip-mlx-python', action='store_true')
    parser.add_argument('--skip-inference', action='store_true')
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), 'tokenized_data_24k.json')
    tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer_24k.json')
    checkpoint = os.path.join(os.path.dirname(__file__), 'mlx-cpp', 'checkpoint.safetensors')

    results = {'training': [], 'inference': []}

    print("=" * 70)
    print("TRAINING BENCHMARK")
    print("=" * 70)

    # C++/MLX
    print("\n[C++/MLX] Running...")
    cpp_result = subprocess.run(
        ['./build/train', '--max-steps', str(args.steps), '--eval-interval', '999'],
        capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), 'mlx-cpp')
    )
    cpp_output = cpp_result.stdout + cpp_result.stderr
    for line in cpp_output.split('\n'):
        if 'ms/step' in line and 'Step' in line:
            parts = line.split('ms/step')[0].strip()
            ms_val = float(parts.split('=')[-1].strip()) if 'elapsed' in parts else None
    # Parse from the last step line
    last_step_line = [l for l in cpp_output.split('\n') if 'ms/step' in l]
    if last_step_line:
        line = last_step_line[-1]
        ms_per_step = float(line.split('ms/step')[0].split(',')[-1].strip())
        step_num = int(line.split('Step')[1].split(':')[0].strip())
        elapsed_s = float(line.split('elapsed,')[0].split(',')[-1].strip().rstrip('s'))
        train_loss = float(line.split('train=')[1].split(',')[0].strip())
        results['training'].append({
            'framework': 'C++/MLX',
            'steps': step_num + 1,
            'ms_per_step': ms_per_step,
            'steps_per_sec': 1000.0 / ms_per_step,
            'elapsed_s': elapsed_s,
            'avg_train_loss': train_loss,
        })
        print(f"  C++/MLX: {ms_per_step:.0f}ms/step ({1000/ms_per_step:.1f} steps/s)")

    # PyTorch
    if not args.skip_pytorch:
        print("\n[PyTorch MPS] Running...")
        try:
            pt_result = benchmark_pytorch(data_path, tokenizer_path, steps=args.steps)
            results['training'].append(pt_result)
            print(f"  PyTorch: {pt_result['ms_per_step']:.0f}ms/step ({pt_result['steps_per_sec']:.1f} steps/s)")
        except Exception as e:
            print(f"  PyTorch failed: {e}")

    # MLX Python
    if not args.skip_mlx_python:
        print("\n[MLX Python] Running...")
        try:
            mlx_result = benchmark_mlx_python(data_path, tokenizer_path, steps=args.steps)
            results['training'].append(mlx_result)
            print(f"  MLX Python: {mlx_result['ms_per_step']:.0f}ms/step ({mlx_result['steps_per_sec']:.1f} steps/s)")
        except Exception as e:
            print(f"  MLX Python failed: {e}")

    # Inference
    if not args.skip_inference and os.path.exists(checkpoint):
        print("\n" + "=" * 70)
        print("INFERENCE BENCHMARK")
        print("=" * 70)

        print("\n[C++/MLX KV cache] Running...")
        try:
            cpp_infer = benchmark_cpp_inference(checkpoint, tokenizer_path, args.prompt, args.infer_tokens)
            results['inference'].append(cpp_infer)
            print(f"  C++/MLX: {cpp_infer.get('tps', 0):.1f} tok/s (prefill {cpp_infer.get('prefill_ms', 0):.1f}ms)")
        except Exception as e:
            print(f"  C++/MLX inference failed: {e}")

        if not args.skip_pytorch:
            print("\n[PyTorch MPS] Running...")
            try:
                pt_infer = benchmark_pytorch_inference(data_path, tokenizer_path, args.prompt, args.infer_tokens)
                results['inference'].append(pt_infer)
                print(f"  PyTorch: {pt_infer['tps']:.1f} tok/s")
            except Exception as e:
                print(f"  PyTorch inference failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results['training']:
        print("\nTraining (ms/step, lower is better):")
        for r in sorted(results['training'], key=lambda x: x['ms_per_step']):
            print(f"  {r['framework']:30s} {r['ms_per_step']:8.0f} ms/step  ({r['steps_per_sec']:.1f} steps/s)")
        fastest = min(results['training'], key=lambda x: x['ms_per_step'])
        print(f"\n  Fastest: {fastest['framework']}")

    if results['inference']:
        print("\nInference (tokens/sec, higher is better):")
        for r in sorted(results['inference'], key=lambda x: -x.get('tps', 0)):
            print(f"  {r['framework']:30s} {r.get('tps', 0):8.1f} tok/s")
        fastest = max(results['inference'], key=lambda x: x.get('tps', 0))
        print(f"\n  Fastest: {fastest['framework']}")

    # Save results
    with open(os.path.join(os.path.dirname(__file__), 'benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")


if __name__ == '__main__':
    main()
