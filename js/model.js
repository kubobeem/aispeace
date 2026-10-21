// WebGPU implementation of the nanochat-ja forward pass (gpt.py semantics).
import { packedF16, f32Tensor } from "./gguf.js";

const D = 768;      // n_embd
const H = 12;       // heads (kv heads too)
const HD = 64;      // head_dim
const FF = 3072;    // ffn hidden
const V = 32768;    // vocab
const NL = 6;       // layers
const CTX = 2048;   // context
const MAXT = 512;   // max tokens pushed through in one dispatch
const EPS = 1e-6;

function parseShaders(src) {
  const parts = src.split(/\/\/@kernel\s+(\w+)/);
  const prelude = parts[0].replace("//@prelude", "");
  const out = {};
  for (let i = 1; i < parts.length; i += 2) out[parts[i]] = prelude + parts[i + 1];
  return out;
}

export async function createDevice() {
  if (!navigator.gpu) throw new Error("このブラウザは WebGPU 未対応です (Chrome/Edge 113+ など)");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU アダプタが取得できませんでした");
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) => console.error("WebGPU error:", e.error.message));
  device.lost.then((info) => console.log("WebGPU device lost:", info.reason, info.message));
  return { device, info: adapter.info || {} };
}

export class NanochatGPU {
  constructor(device, shaders) {
    this.device = device;
    this.kernels = parseShaders(shaders);
    this.pipes = {};
    this.mods = {};
    for (const [name, src] of Object.entries(this.kernels)) {
      const mod = device.createShaderModule({ code: src, label: name });
      this.mods[name] = mod;
      this.pipes[name] = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: name }, label: name });
    }
    this.uBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uData = new ArrayBuffer(32);
    this.uU32 = new Uint32Array(this.uData);
    this.uF32 = new Float32Array(this.uData);
  }

  async checkShaders() {
    const out = {};
    for (const [name, mod] of Object.entries(this.mods)) {
      const info = await mod.getCompilationInfo();
      out[name] = info.messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
    }
    return out;
  }

  buf(bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 4) * 4), usage });
  }

  writeU({ T = 0, T0 = 0, M = 0, K = 0, a = 0, b = 0, c = 0, d = 0 }) {
    this.uU32[0] = T; this.uU32[1] = T0; this.uU32[2] = M; this.uU32[3] = K;
    this.uF32[4] = a; this.uF32[5] = b; this.uF32[6] = c; this.uF32[7] = d;
    this.device.queue.writeBuffer(this.uBuf, 0, this.uData);
  }

  run(name, u, buffers, gx, gy = 1) {
    this.writeU(u);
    const entries = buffers.map((b, i) => ({ binding: i, resource: { buffer: b } }));
    const bg = this.device.createBindGroup({ layout: this.pipes[name].getBindGroupLayout(0), entries });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  async upload(file, onProgress) {
    const names = [
      "token_embd.weight", "output.weight",
      "resid_lambda.weight", "x0_lambda.weight", "smear_lambda.weight", "backout_lambda.weight", "smear_gate.weight",
    ];
    for (let i = 0; i < NL; i++) {
      names.push(`blk.${i}.attn_norm.weight`, `blk.${i}.attn_q.weight`, `blk.${i}.attn_k.weight`, `blk.${i}.attn_v.weight`,
        `blk.${i}.attn_q_norm.weight`, `blk.${i}.attn_k_norm.weight`, `blk.${i}.attn_output.weight`,
        `blk.${i}.ffn_norm.weight`, `blk.${i}.ffn_up.weight`, `blk.${i}.ffn_down.weight`);
    }
    for (let i = 1; i < NL; i += 2) {
      names.push(`value_embeds.${i}.weight`, `blk.${i}.attn.ve_gate.weight`);
    }

    this.w = {};
    for (let n = 0; n < names.length; n++) {
      const name = names[n];
      const t = file.tensors.get(name);
      if (!t) throw new Error(`GGUF に ${name} がありません (nano-ja 用 GGUF ではありません)`);
      const isVec = t.dims.length === 1;
      const asF32 = isVec || /ve_gate/.test(name); // the vebind kernel reads this one as f32
      let data, bytes;
      if (asF32) {
        data = f32Tensor(file, name);
        bytes = data.byteLength;
        if (/attn_k_norm/.test(name)) {
          // gpt.py multiplies k by 1.2 after QK norm; the exporter folded that
          // into q_norm only, so restore it when the stored weights are plain 1.0.
          if (data.every((v) => Math.abs(v - 1) < 1e-3)) for (let i = 0; i < data.length; i++) data[i] *= 1.2;
        }
      } else {
        data = packedF16(file, name);
        bytes = data.byteLength;
      }
      const b = this.buf(bytes);
      this.device.queue.writeBuffer(b, 0, data);
      this.w[name] = b;
      if (onProgress) onProgress((n + 1) / names.length, name);
      if (n % 4 === 0) await new Promise((r) => setTimeout(r, 0));
    }
    this.lam = {
      resid: Array.from(f32Tensor(file, "resid_lambda.weight")),
      x0: Array.from(f32Tensor(file, "x0_lambda.weight")),
      smear: f32Tensor(file, "smear_lambda.weight")[0],
      backout: f32Tensor(file, "backout_lambda.weight")[0],
    };

    // scratch buffers
    this.idsBuf = this.buf(MAXT * 4);
    this.x = this.buf(MAXT * D * 4);
    this.x0 = this.buf(MAXT * D * 4);
    this.xb = this.buf(MAXT * D * 4);
    this.xn = this.buf(MAXT * D * 4);
    this.q = this.buf(MAXT * D * 4);
    this.k = this.buf(MAXT * D * 4);
    this.v = this.buf(MAXT * D * 4);
    this.attn = this.buf(MAXT * D * 4);
    this.prev = this.buf(D * 4);
    this.gateBuf = this.buf(MAXT * H * 4);
    this.h = this.buf(MAXT * FF * 4);
    this.y768 = this.buf(MAXT * D * 4);
    this.yDown = this.buf(MAXT * D * 4);
    this.dummy = this.buf(D * 4);
    this.xs = this.buf(D * 4);
    this.logits = this.buf(V * 4);
    this.kv = [];
    for (let i = 0; i < NL; i++) {
      this.kv.push({
        k: this.buf(CTX * D * 2),
        v: this.buf(CTX * D * 2),
      });
    }
    this.kvPos = 0;
  }

  reset() {
    this.kvPos = 0;
    this.device.queue.writeBuffer(this.prev, 0, new Float32Array(D));
  }

  _norm(src, dst, weightName, scale = 1) {
    const w = weightName ? this.w[weightName] : this.dummy;
    this.run("rmsnorm", { T: this.T, K: D, a: scale, b: EPS, c: weightName ? 1 : 0 }, [this.uBuf, src, w, dst], this.T);
  }

  async _stage(name, buf, count) {
    if (this.onStage) await this.onStage(name, buf, count);
  }

  // full forward over T tokens starting at KV position T0; last column only is projected
  async forwardRun(ids, T0, T) {
    this.T = T;
    this.device.queue.writeBuffer(this.idsBuf, 0, new Uint32Array(ids));
    const lam = this.lam;
    this.run("embed", { T }, [this.uBuf, this.w["token_embd.weight"], this.idsBuf, this.x0], T);
    await this._stage("embed", this.x0, D * T);
    // parameterless RMS norm right after the lookup (gpt.py line 473). The kernel
    // needs distinct source/dest buffers: binding one buffer twice is undefined.
    this._norm(this.x0, this.q, null);
    await this._stage("emb_norm", this.q, D * T);
    this.run("smear", { T, a: lam.smear }, [this.uBuf, this.q, this.w["smear_gate.weight"], this.prev, this.x], T);
    await this._stage("smear", this.x, D * T);
    this.run("blend", { T, K: D, a: 0, b: 1 }, [this.uBuf, this.x0, this.x], T);

    for (let i = 0; i < NL; i++) {
      this.run("blend", { T, K: D, a: lam.resid[i], b: lam.x0[i] }, [this.uBuf, this.x, this.x0], T);
      await this._stage(`blend${i}`, this.x, D * T);
      this._norm(this.x, this.xn, `blk.${i}.attn_norm.weight`);
      await this._stage(`norm${i}`, this.xn, D * T);

      this.run("gemv", { T, K: D, M: D }, [this.uBuf, this.w[`blk.${i}.attn_q.weight`], this.xn, this.q], D);
      this.run("gemv", { T, K: D, M: D }, [this.uBuf, this.w[`blk.${i}.attn_k.weight`], this.xn, this.k], D);
      this.run("gemv", { T, K: D, M: D }, [this.uBuf, this.w[`blk.${i}.attn_v.weight`], this.xn, this.v], D);
      await this._stage(`q_pre${i}`, this.q, D * T);
      await this._stage(`v_pre${i}`, this.v, D * T);

      if (i % 2 === (NL - 1) % 2) {
        this.run("vebind", { T }, [this.uBuf, this.xn, this.w[`blk.${i}.attn.ve_gate.weight`], this.gateBuf], T);
        await this._stage(`ve_gate${i}`, this.gateBuf, H * T);
        this.run("veadd", { T }, [this.uBuf, this.v, this.w[`value_embeds.${i}.weight`], this.idsBuf, this.gateBuf], H, T);
        await this._stage(`v_post${i}`, this.v, D * T);
      }

      this.run("rope", { T, T0 }, [this.uBuf, this.q], H, T);
      this.run("rope", { T, T0 }, [this.uBuf, this.k], H, T);
      await this._stage(`q_rope${i}`, this.q, D * T);
      this.run("rmsnorm64", { T, a: 1, b: EPS }, [this.uBuf, this.q, this.w[`blk.${i}.attn_q_norm.weight`]], H, T);
      this.run("rmsnorm64", { T, a: 1, b: EPS }, [this.uBuf, this.k, this.w[`blk.${i}.attn_k_norm.weight`]], H, T);
      await this._stage(`q_norm${i}`, this.q, D * T);

      this.run("storekv", { T, T0 }, [this.uBuf, this.k, this.v, this.kv[i].k, this.kv[i].v], T);
      this.run("attention", { T, T0, a: 1 / Math.sqrt(HD) }, [this.uBuf, this.q, this.kv[i].k, this.kv[i].v, this.attn], H, T);
      await this._stage(`attn${i}`, this.attn, D * T);

      this.run("gemv", { T, K: D, M: D }, [this.uBuf, this.w[`blk.${i}.attn_output.weight`], this.attn, this.y768], D);
      this.run("add", { T, K: D, a: 1 }, [this.uBuf, this.x, this.y768], T);
      await this._stage(`x_attn${i}`, this.x, D * T);

      this._norm(this.x, this.xn, `blk.${i}.ffn_norm.weight`);
      this.run("gemv", { T, K: D, M: FF }, [this.uBuf, this.w[`blk.${i}.ffn_up.weight`], this.xn, this.h], FF);
      this.run("relu2", { T, K: FF }, [this.uBuf, this.h], T);
      await this._stage(`ffn_h${i}`, this.h, FF * T);
      this.run("gemv", { T, K: FF, M: D }, [this.uBuf, this.w[`blk.${i}.ffn_down.weight`], this.h, this.yDown], D);
      this.run("add", { T, K: D, a: 1 }, [this.uBuf, this.x, this.yDown], T);
      await this._stage(`x_mlp${i}`, this.x, D * T);
      await this._stage(`x_layer${i}`, this.x, D * T);

      if (i === NL / 2) {
        this.run("blend", { T, K: D, a: 0, b: 1 }, [this.uBuf, this.xb, this.x], T);
        await this._stage("xb", this.xb, D * T);
      }
    }
    this.run("subscaled", { T, a: lam.backout }, [this.uBuf, this.x, this.xb], T);
    await this._stage("x_final", this.x, D * T);
    this._norm(this.x, this.xn, null);
    await this._stage("xn_final", this.xn, D * T);
    this.run("copycol", { T, K: T - 1 }, [this.uBuf, this.xn, this.xs], 1);
    this.run("gemv", { T: 1, K: D, M: V }, [this.uBuf, this.w["output.weight"], this.xs, this.logits], V);
    this.kvPos = T0 + T;
  }

  async readLogits() {
    return this.readBuf(this.logits, V);
  }

  async readBuf(src, count, asU32 = false) {
    const bytes = count * 4;
    const st = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, st, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = asU32 ? new Uint32Array(st.getMappedRange().slice(0)) : new Float32Array(st.getMappedRange().slice(0));
    st.unmap();
    st.destroy();
    return out;
  }

  async prefill(ids) {
    if (ids.length > MAXT) throw new Error(`プロンプトが長すぎます (${ids.length} > ${MAXT})`);
    if (this.kvPos + ids.length > CTX) throw new Error("会話がコンテキスト長を超えました。リセットしてください。");
    const t0 = this.kvPos;
    await this.forwardRun(ids, t0, ids.length);
    return this.readLogits();
  }

  async decode(tokenId) {
    if (this.kvPos + 1 > CTX) throw new Error("会話がコンテキスト長を超えました。リセットしてください。");
    const t0 = this.kvPos;
    await this.forwardRun([tokenId], t0, 1);
    return this.readLogits();
  }
}

export function finalizeLogits(raw) {
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = 15 * Math.tanh(raw[i] / 15);
  return out;
}

export function sampleToken(logits, temperature, topK, rng = Math.random) {
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;
  let idx = Array.from({ length: scaled.length }, (_, i) => i);
  idx.sort((a, b) => scaled[b] - scaled[a]);
  const keep = idx.slice(0, Math.max(1, Math.min(topK, idx.length)));
  let max = -Infinity;
  for (const i of keep) max = Math.max(max, scaled[i]);
  let sum = 0;
  const probs = keep.map((i) => {
    const p = Math.exp(scaled[i] - max);
    sum += p;
    return p;
  });
  let r = rng() * sum;
  for (let k = 0; k < keep.length; k++) {
    r -= probs[k];
    if (r <= 0) return keep[k];
  }
  return keep[keep.length - 1];
}
