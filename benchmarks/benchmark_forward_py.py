import mlx.core as mx
import mlx.nn as nn
import time

vocab_size, n_layer, n_head, d_model = 24000, 8, 9, 576
head_dim = d_model // n_head
hidden = int(8.0/3.0 * d_model)
block_size, batch_size = 256, 32

class MLXGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.ln_f_w = mx.ones((d_model,))
        self.blocks = []
        for _ in range(n_layer):
            b = dict(
                attn_norm_w=mx.ones((d_model,)),
                qkv_w=mx.random.normal((3*d_model, d_model)) * 0.02,
                q_norm_w=mx.ones((head_dim,)),
                k_norm_w=mx.ones((head_dim,)),
                proj_w=mx.random.normal((d_model, d_model)) * (0.02 / (2*n_layer)**0.5),
                mlp_norm_w=mx.ones((d_model,)),
                w_gate=mx.random.normal((hidden, d_model)) * 0.02,
                w_up=mx.random.normal((hidden, d_model)) * 0.02,
                w_out=mx.random.normal((d_model, hidden)) * (0.02 / (2*n_layer)**0.5),
            )
            self.blocks.append(b)

    def rmsn(self, x, w):
        return mx.fast.rms_norm(x, w, 1e-5)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx)
        eos = (idx == 1).astype(mx.float32)
        seg = mx.cumsum(eos, axis=-1) - eos
        same_seg = mx.equal(seg[:, :, None], seg[:, None, :])
        causal = mx.tril(mx.ones((T, T), dtype=mx.bool_))
        mask = mx.where(causal, 0.0, -1e9)[None, None, :, :]
        scale = 1.0 / (head_dim ** 0.5)
        for b in self.blocks:
            h = self.rmsn(x, b['attn_norm_w'])
            hf = h.reshape(B*T, d_model)
            qkv = (hf @ b['qkv_w'].T).reshape(B, T, 3*d_model)
            q, k, v = mx.split(qkv, 3, axis=-1)
            q = q.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, head_dim, traditional=False, base=1000.0, scale=1.0, offset=0)
            k = mx.fast.rope(k, head_dim, traditional=False, base=1000.0, scale=1.0, offset=0)
            q = self.rmsn(q, b['q_norm_w'])
            k = self.rmsn(k, b['k_norm_w'])
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
            attn = attn.transpose(0, 2, 1, 3).reshape(B*T, d_model)
            attn = (attn @ b['proj_w'].T).reshape(B, T, d_model)
            m = self.rmsn(x, b['mlp_norm_w'])
            mf = m.reshape(B*T, d_model)
            gate = (mf @ b['w_gate'].T).reshape(B, T, hidden)
            up = (mf @ b['w_up'].T).reshape(B, T, hidden)
            out = (gate * mx.sigmoid(gate) * up).reshape(B*T, hidden) @ b['w_out'].T
            x = x + attn + out.reshape(B, T, d_model)
        x = self.rmsn(x, self.ln_f_w)
        return (x.reshape(B*T, d_model) @ self.token_emb.weight.T).reshape(B, T, vocab_size)

model = MLXGPT()

idx = mx.random.randint(0, vocab_size, (batch_size, block_size), dtype=mx.uint32)
forward = mx.compile(model)

for _ in range(5):
    mx.eval(forward(idx))

t0 = time.time()
for _ in range(50):
    mx.eval(forward(idx))
print(f'Python forward only: {(time.time()-t0)/50*1000:.0f}ms')
