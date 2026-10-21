"""NumPy reference implementation of the nanochat-ja forward pass (gpt.py semantics).

Loads the bf16 GGUF directly and prints the top-10 next-token logits for the test
prompt. The WebGPU demo reproduces these exactly (see README).

    python tools/ref.py path/to/nano-ja-sft.gguf
    python tools/ref.py model.gguf 32759,32760,31946,2628,1407,266,32761,32762
"""
import sys
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "nano-ja-sft.gguf"
buf = open(PATH, "rb").read()


def parse(buf):
    o = 0

    def u32():
        nonlocal o
        v = int.from_bytes(buf[o:o + 4], "little"); o += 4; return v

    def u64():
        nonlocal o
        v = int.from_bytes(buf[o:o + 8], "little"); o += 8; return v

    def s():
        nonlocal o
        n = u64(); v = buf[o:o + n].decode("utf-8", "replace"); o += n; return v

    def val(t):
        nonlocal o
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t == 8:
            return s()
        if t == 9:
            at, n = u32(), u64()
            return [val(at) for _ in range(n)]
        size = sizes[t]
        raw = buf[o:o + size]; o += size
        signed = t in (1, 3, 5, 11)
        if t == 6:
            return float(np.frombuffer(raw, "<f4")[0])
        if t == 12:
            return float(np.frombuffer(raw, "<f8")[0])
        if t == 7:
            return bool(raw[0])
        return int.from_bytes(raw, "little", signed=signed)

    assert buf[:4] == b"GGUF"
    o = 8
    nt, nkv = u64(), u64()
    kv = {}
    for _ in range(nkv):
        k = s()
        kv[k] = val(u32())
    tensors = {}
    for _ in range(nt):
        name = s()
        nd = u32()
        dims = [u64() for _ in range(nd)]
        t = u32()
        tensors[name] = (dims, t, u64())
    align = kv.get("general.alignment", 32)
    return kv, tensors, (o + align - 1) // align * align


kv, tensors, data0 = parse(buf)


def load(name):
    dims, t, off = tensors[name]
    n = int(np.prod(dims))
    if t == 0:
        a = np.frombuffer(buf, "<f4", n, data0 + off)
    elif t == 30:  # bf16
        a = (np.frombuffer(buf, "<u2", n, data0 + off).astype(np.uint32) << 16).view(np.float32)
    else:
        raise ValueError(f"{name}: tensor type {t}")
    a = np.array(a, dtype=np.float32)
    return a.reshape(dims[1], dims[0]) if len(dims) == 2 else a


W = {n: load(n) for n in tensors}
NL, D, H, HD, FF = 6, 768, 12, 64, 3072
EPS, THETA = 1e-6, 100000.0
print(f"loaded {PATH}: {len(W)} tensors, arch={kv['general.architecture']}")


def rms(x):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def forward(ids):
    T = len(ids)
    x = W["token_embd.weight"][np.asarray(ids)]
    x = rms(x)
    gate = float(W["smear_lambda.weight"][0]) * (1 / (1 + np.exp(-(x[1:, :24] @ W["smear_gate.weight"]))))
    xs = x.copy()
    xs[1:] = x[1:] + gate[:, None] * x[:-1]
    x, x0 = xs, xs.copy()

    inv = 1.0 / (THETA ** (np.arange(0, 64, 2, dtype=np.float32) / 64))
    ang = np.arange(T, dtype=np.float32)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    xb = None
    for i in range(NL):
        x = float(W["resid_lambda.weight"][i]) * x + float(W["x0_lambda.weight"][i]) * x0
        xn = rms(x)
        q = (xn @ W[f"blk.{i}.attn_q.weight"].T).reshape(T, H, HD)
        k = (xn @ W[f"blk.{i}.attn_k.weight"].T).reshape(T, H, HD)
        v = (xn @ W[f"blk.{i}.attn_v.weight"].T).reshape(T, H, HD)
        if f"value_embeds.{i}.weight" in W:
            g = 3 / (1 + np.exp(-(xn[:, :12] @ W[f"blk.{i}.attn.ve_gate.weight"].T)))
            ve = W[f"value_embeds.{i}.weight"][np.asarray(ids)].reshape(T, H, HD)
            v = v + g[:, :, None] * ve
        for name in ("q", "k"):
            t_ = q if name == "q" else k
            x1, x2 = t_[..., :32], t_[..., 32:]
            y1 = x1 * cos[:, None, :] + x2 * sin[:, None, :]
            y2 = x1 * (-sin[:, None, :]) + x2 * cos[:, None, :]
            t_ = np.concatenate([y1, y2], -1)
            if name == "q":
                q = t_
            else:
                k = t_
        q, k = rms(q) * 1.2, rms(k) * 1.2
        scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(HD)
        scores = np.where(np.triu(np.ones((T, T), dtype=bool), 1)[None], -1e30, scores)
        y = np.einsum("hts,shd->thd", softmax(scores), v).reshape(T, D)
        x = x + y @ W[f"blk.{i}.attn_output.weight"].T
        xn = rms(x)
        h = np.maximum(xn @ W[f"blk.{i}.ffn_up.weight"].T, 0) ** 2
        x = x + h @ W[f"blk.{i}.ffn_down.weight"].T
        if i == NL // 2:
            xb = x.copy()
    x = rms(x - float(W["backout_lambda.weight"][0]) * xb)
    logits = x[-1] @ W["output.weight"].T
    return 15 * np.tanh(logits / 15)


if __name__ == "__main__":
    ids = ([int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2
           else [32759, 32760, 31946, 2628, 1407, 266, 32761, 32762])
    lg = forward(ids)
    top = np.argsort(-lg)[:10]
    print("tokens ", ids)
    print("top10  ", [(int(i), round(float(lg[i]), 4)) for i in top])
