// Pure JS (CPU) implementation of the nanochat-ja forward pass.
// Mirrors tools/ref.py and the WebGPU kernels so results match within f16 noise.
// Same public surface as NanochatGPU: upload / reset / prefill / decode / kvPos.

import { f32Any, f32Tensor } from "./gguf.js";

const D = 768;
const H = 12;
const HD = 64;
const FF = 3072;
const V = 32768;
const NL = 6;
const CTX = 2048;
const MAXT = 512;
const EPS = 1e-6;
const THETA = 100000;

const yieldToUi = () => new Promise((r) => setTimeout(r, 0));

function rmsVec(x, n, scale = 1) {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[i] * x[i];
  const rr = scale / Math.sqrt(ss / n + EPS);
  for (let i = 0; i < n; i++) x[i] *= rr;
}

function rmsHead(x, base, scale = 1) {
  let ss = 0;
  for (let i = 0; i < HD; i++) ss += x[base + i] * x[base + i];
  const rr = scale / Math.sqrt(ss / HD + EPS);
  for (let i = 0; i < HD; i++) x[base + i] *= rr;
}

// A is [out, in] row-major (matches ref.py after reshape(dims[1], dims[0])).
// y = x @ A.T  <=>  y[o] = sum_i x[i] * A[o, i]
function gemv(A, x, y, out, inn) {
  for (let o = 0; o < out; o++) {
    let acc = 0;
    const row = o * inn;
    for (let i = 0; i < inn; i++) acc += x[i] * A[row + i];
    y[o] = acc;
  }
}

export class NanochatCPU {
  constructor() {
    this.kvPos = 0;
    this.backend = "cpu";
  }

  async upload(file, onProgress) {
    const names = [
      "token_embd.weight", "output.weight",
      "resid_lambda.weight", "x0_lambda.weight", "smear_lambda.weight", "backout_lambda.weight", "smear_gate.weight",
    ];
    for (let i = 0; i < NL; i++) {
      names.push(
        `blk.${i}.attn_norm.weight`, `blk.${i}.attn_q.weight`, `blk.${i}.attn_k.weight`, `blk.${i}.attn_v.weight`,
        `blk.${i}.attn_q_norm.weight`, `blk.${i}.attn_k_norm.weight`, `blk.${i}.attn_output.weight`,
        `blk.${i}.ffn_norm.weight`, `blk.${i}.ffn_up.weight`, `blk.${i}.ffn_down.weight`,
      );
    }
    for (let i = 1; i < NL; i += 2) {
      names.push(`value_embeds.${i}.weight`, `blk.${i}.attn.ve_gate.weight`);
    }

    this.w = {};
    for (let n = 0; n < names.length; n++) {
      const name = names[n];
      const { data, dims } = f32Any(file, name);
      // Match ref.py: 2D tensors become (dims[1], dims[0]) so matmul is x @ W.T
      // with W stored row-major as [out, in].
      if (dims.length === 2) {
        // data is stored with dims[0] fastest; reshape to [dims[1], dims[0]]
        // by treating the flat buffer as already contiguous that way after transpose
        // of logical axes. GGUF/ggml stores [n_embd, n_vocab] etc. with first dim
        // fastest → after reshape(dims[1], dims[0]) row o is data[o*dims[0] ..].
        this.w[name] = data; // already in the right order for gemv(out=dims[1], in=dims[0])
        this.w[`${name}__shape`] = [dims[1], dims[0]];
      } else {
        this.w[name] = data;
      }
      if (/attn_k_norm/.test(name)) {
        // same fold as GPU path: exporter put 1.2 only on q_norm
        if (data.every((v) => Math.abs(v - 1) < 1e-3)) {
          for (let i = 0; i < data.length; i++) data[i] *= 1.2;
        }
      }
      if (onProgress) onProgress((n + 1) / names.length, name);
      if (n % 3 === 0) await yieldToUi();
    }

    this.lam = {
      resid: Array.from(f32Tensor(file, "resid_lambda.weight")),
      x0: Array.from(f32Tensor(file, "x0_lambda.weight")),
      smear: f32Tensor(file, "smear_lambda.weight")[0],
      backout: f32Tensor(file, "backout_lambda.weight")[0],
    };

    // KV cache: [layer][pos * D + d]
    this.kCache = Array.from({ length: NL }, () => new Float32Array(CTX * D));
    this.vCache = Array.from({ length: NL }, () => new Float32Array(CTX * D));
    this.prev = new Float32Array(D); // smear previous token (decode path)
    this.kvPos = 0;

    // precomputed RoPE inv frequencies
    this.ropeInv = new Float32Array(HD / 2);
    for (let i = 0; i < HD / 2; i++) this.ropeInv[i] = 1 / Math.pow(THETA, i / (HD / 2));
  }

  reset() {
    this.kvPos = 0;
    this.prev.fill(0);
  }

  // Forward T tokens starting at KV position T0. Returns logits for the last token.
  async forwardRun(ids, T0, T) {
    const W = this.w;
    const emb = W["token_embd.weight"]; // [V, D] stored as reshape → shape [V, D] with row = token
    // token_embd dims in GGUF are typically [D, V]; after reshape(dims[1], dims[0]) → [V, D]
    const embShape = W["token_embd.weight__shape"]; // [V, D]

    // x[t*D + d]
    const x = new Float32Array(T * D);
    const x0 = new Float32Array(T * D);

    for (let t = 0; t < T; t++) {
      const tok = ids[t];
      const base = tok * D;
      for (let d = 0; d < D; d++) x[t * D + d] = emb[base + d];
    }
    // parameterless RMS after embed
    for (let t = 0; t < T; t++) rmsVec(x.subarray(t * D, (t + 1) * D), D);

    // smear
    const smearGate = W["smear_gate.weight"]; // [24]
    const smearLam = this.lam.smear;
    const xs = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      let acc = 0;
      for (let j = 0; j < 24; j++) acc += smearGate[j] * x[t * D + j];
      const g = smearLam / (1 + Math.exp(-acc));
      for (let d = 0; d < D; d++) {
        let v = x[t * D + d];
        if (T > 1) {
          if (t > 0) v += g * x[(t - 1) * D + d];
        } else {
          v += g * this.prev[d];
        }
        xs[t * D + d] = v;
      }
      if (t === T - 1) {
        for (let d = 0; d < D; d++) this.prev[d] = x[t * D + d];
      }
    }
    x.set(xs);
    x0.set(xs);

    let xb = null;

    for (let i = 0; i < NL; i++) {
      if (i > 0 && i % 2 === 0) await yieldToUi();

      const resid = this.lam.resid[i];
      const x0l = this.lam.x0[i];
      for (let j = 0; j < T * D; j++) x[j] = resid * x[j] + x0l * x0[j];

      // attn norm
      const xn = new Float32Array(T * D);
      const anW = W[`blk.${i}.attn_norm.weight`];
      for (let t = 0; t < T; t++) {
        const base = t * D;
        xn.set(x.subarray(base, base + D), base);
        rmsVec(xn.subarray(base, base + D), D);
        for (let d = 0; d < D; d++) xn[base + d] *= anW[d];
      }

      const q = new Float32Array(T * D);
      const k = new Float32Array(T * D);
      const v = new Float32Array(T * D);
      const Wq = W[`blk.${i}.attn_q.weight`];
      const Wk = W[`blk.${i}.attn_k.weight`];
      const Wv = W[`blk.${i}.attn_v.weight`];
      for (let t = 0; t < T; t++) {
        const xb_ = xn.subarray(t * D, (t + 1) * D);
        gemv(Wq, xb_, q.subarray(t * D, (t + 1) * D), D, D);
        gemv(Wk, xb_, k.subarray(t * D, (t + 1) * D), D, D);
        gemv(Wv, xb_, v.subarray(t * D, (t + 1) * D), D, D);
      }

      // value embeds on layers 1,3,5
      if (i % 2 === 1) {
        const veGate = W[`blk.${i}.attn.ve_gate.weight`]; // [H, 12]
        const veTbl = W[`value_embeds.${i}.weight`]; // [V, D]
        for (let t = 0; t < T; t++) {
          const tok = ids[t];
          for (let h = 0; h < H; h++) {
            let acc = 0;
            for (let j = 0; j < 12; j++) acc += veGate[h * 12 + j] * xn[t * D + j];
            const g = 3 / (1 + Math.exp(-acc));
            const veBase = tok * D + h * HD;
            const vBase = t * D + h * HD;
            for (let d = 0; d < HD; d++) v[vBase + d] += g * veTbl[veBase + d];
          }
        }
      }

      // RoPE (half-split)
      for (let t = 0; t < T; t++) {
        const pos = T0 + t;
        for (let h = 0; h < H; h++) {
          const base = t * D + h * HD;
          for (let j = 0; j < HD / 2; j++) {
            const ang = pos * this.ropeInv[j];
            const c = Math.cos(ang);
            const s = Math.sin(ang);
            const x1q = q[base + j];
            const x2q = q[base + j + HD / 2];
            q[base + j] = x1q * c + x2q * s;
            q[base + j + HD / 2] = x2q * c - x1q * s;
            const x1k = k[base + j];
            const x2k = k[base + j + HD / 2];
            k[base + j] = x1k * c + x2k * s;
            k[base + j + HD / 2] = x2k * c - x1k * s;
          }
        }
      }

      // QK RMS norm * weights (k already has 1.2 folded when needed)
      const qnW = W[`blk.${i}.attn_q_norm.weight`];
      const knW = W[`blk.${i}.attn_k_norm.weight`];
      for (let t = 0; t < T; t++) {
        for (let h = 0; h < H; h++) {
          const qb = t * D + h * HD;
          const kb = t * D + h * HD;
          rmsHead(q, qb);
          rmsHead(k, kb);
          for (let d = 0; d < HD; d++) {
            q[qb + d] *= qnW[d];
            k[kb + d] *= knW[d];
          }
        }
      }

      // store KV
      for (let t = 0; t < T; t++) {
        const pos = T0 + t;
        this.kCache[i].set(k.subarray(t * D, (t + 1) * D), pos * D);
        this.vCache[i].set(v.subarray(t * D, (t + 1) * D), pos * D);
      }

      // causal attention
      const attn = new Float32Array(T * D);
      const scale = 1 / Math.sqrt(HD);
      const scores = new Float32Array(CTX); // reuse
      for (let t = 0; t < T; t++) {
        const nkv = T0 + t + 1;
        for (let h = 0; h < H; h++) {
          const qb = t * D + h * HD;
          let mx = -1e30;
          for (let j = 0; j < nkv; j++) {
            let acc = 0;
            const kb = j * D + h * HD;
            for (let d = 0; d < HD; d++) acc += q[qb + d] * this.kCache[i][kb + d];
            scores[j] = acc * scale;
            if (scores[j] > mx) mx = scores[j];
          }
          let sum = 0;
          for (let j = 0; j < nkv; j++) {
            scores[j] = Math.exp(scores[j] - mx);
            sum += scores[j];
          }
          const inv = 1 / sum;
          const ob = t * D + h * HD;
          for (let d = 0; d < HD; d++) attn[ob + d] = 0;
          for (let j = 0; j < nkv; j++) {
            const p = scores[j] * inv;
            const vb = j * D + h * HD;
            for (let d = 0; d < HD; d++) attn[ob + d] += p * this.vCache[i][vb + d];
          }
        }
      }

      // attn output proj + residual
      const Wo = W[`blk.${i}.attn_output.weight`];
      for (let t = 0; t < T; t++) {
        const y = new Float32Array(D);
        gemv(Wo, attn.subarray(t * D, (t + 1) * D), y, D, D);
        for (let d = 0; d < D; d++) x[t * D + d] += y[d];
      }

      // FFN
      const fnW = W[`blk.${i}.ffn_norm.weight`];
      const Wu = W[`blk.${i}.ffn_up.weight`];
      const Wd = W[`blk.${i}.ffn_down.weight`];
      for (let t = 0; t < T; t++) {
        const base = t * D;
        const xn2 = new Float32Array(D);
        xn2.set(x.subarray(base, base + D));
        rmsVec(xn2, D);
        for (let d = 0; d < D; d++) xn2[d] *= fnW[d];
        const h = new Float32Array(FF);
        gemv(Wu, xn2, h, FF, D);
        for (let j = 0; j < FF; j++) {
          const v = Math.max(h[j], 0);
          h[j] = v * v;
        }
        const y = new Float32Array(D);
        gemv(Wd, h, y, D, FF);
        for (let d = 0; d < D; d++) x[base + d] += y[d];
      }

      if (i === (NL >> 1)) {
        xb = x.slice();
      }
    }

    // backout
    const back = this.lam.backout;
    for (let j = 0; j < T * D; j++) x[j] -= back * xb[j];
    for (let t = 0; t < T; t++) rmsVec(x.subarray(t * D, (t + 1) * D), D);

    // logits from last token
    const last = x.subarray((T - 1) * D, T * D);
    const logits = new Float32Array(V);
    gemv(W["output.weight"], last, logits, V, D);

    this.kvPos = T0 + T;
    return logits;
  }

  async prefill(ids) {
    if (ids.length > MAXT) throw new Error(`プロンプトが長すぎます (${ids.length} > ${MAXT})`);
    if (this.kvPos + ids.length > CTX) throw new Error("会話がコンテキスト長を超えました。リセットしてください。");
    const t0 = this.kvPos;
    return this.forwardRun(ids, t0, ids.length);
  }

  async decode(tokenId) {
    if (this.kvPos + 1 > CTX) throw new Error("会話がコンテキスト長を超えました。リセットしてください。");
    const t0 = this.kvPos;
    return this.forwardRun([tokenId], t0, 1);
  }
}
