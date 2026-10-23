// GGUF reader for the nano-ja demo. Handles the two tensor types this file uses:
// f32 (0) and bf16 (30). Weights are packed as u32 pairs of f16 for WebGPU
// (unpack2x16float), which avoids requiring the shader-f16 feature.
// CPU path loads the same tensors as Float32Array.

const GGML_F32 = 0;
const GGML_BF16 = 30;

export function parseGGUF(buffer) {
  const dv = new DataView(buffer);
  let o = 0;
  const u32 = () => { const v = dv.getUint32(o, true); o += 4; return v; };
  const u64 = () => { const v = Number(dv.getBigUint64(o, true)); o += 8; return v; };
  const str = () => {
    const n = u64();
    const s = new TextDecoder().decode(new Uint8Array(buffer, o, n));
    o += n;
    return s;
  };
  const val = (t) => {
    switch (t) {
      case 0: { const v = dv.getUint8(o); o += 1; return v; }
      case 1: { const v = dv.getInt8(o); o += 1; return v; }
      case 2: { const v = dv.getUint16(o, true); o += 2; return v; }
      case 3: { const v = dv.getInt16(o, true); o += 2; return v; }
      case 4: { const v = dv.getUint32(o, true); o += 4; return v; }
      case 5: { const v = dv.getInt32(o, true); o += 4; return v; }
      case 6: { const v = dv.getFloat32(o, true); o += 4; return v; }
      case 7: { const v = !!dv.getUint8(o); o += 1; return v; }
      case 8: return str();
      case 9: {
        const at = u32();
        const n = u64();
        const arr = [];
        for (let i = 0; i < n; i++) arr.push(val(at));
        return arr;
      }
      case 10: return u64();
      case 11: { const v = Number(dv.getBigInt64(o, true)); o += 8; return v; }
      case 12: { const v = dv.getFloat64(o, true); o += 8; return v; }
    }
    throw new Error(`GGUF: unknown metadata type ${t} at ${o}`);
  };

  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
  if (magic !== "GGUF") throw new Error("not a GGUF file");
  o = 8;
  const nTensors = u64();
  const nKV = u64();

  const kv = {};
  for (let i = 0; i < nKV; i++) {
    const key = str();
    const type = u32();
    kv[key] = val(type);
  }

  const tensors = new Map();
  for (let i = 0; i < nTensors; i++) {
    const name = str();
    const nd = u32();
    const dims = [];
    for (let d = 0; d < nd; d++) dims.push(u64());
    const type = u32();
    const offset = u64();
    tensors.set(name, { name, dims, type, offset });
  }

  const align = kv["general.alignment"] || 32;
  const dataStart = Math.ceil(o / align) * align;
  return { kv, tensors, dataStart, buffer };
}

// bf16 -> f16 lookup over all 16-bit patterns (one table, reused by every tensor)
let bf16ToF16 = null;
function bf16Table() {
  if (bf16ToF16) return bf16ToF16;
  const t = new Uint16Array(65536);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let i = 0; i < 65536; i++) {
    u32[0] = i << 16;
    const x = u32[0];
    const sign = (x >>> 16) & 0x8000;
    const exp = (x >>> 23) & 0xff;
    const mant = x & 0x7fffff;
    if (exp === 0xff) { t[i] = sign | (mant ? 0x7e00 : 0x7c00); continue; }
    let e = exp - 127 + 15;
    if (e >= 0x1f) { t[i] = sign | 0x7c00; continue; }
    if (e <= 0) {
      if (e < -10) { t[i] = sign; continue; }
      const m = (mant | 0x800000) >> (1 - e);
      t[i] = sign | ((m + 0x1000) >> 13);
      continue;
    }
    let m = mant + 0x1000;
    if (m & 0x800000) { e += 1; m = 0; if (e >= 0x1f) { t[i] = sign | 0x7c00; continue; } }
    t[i] = sign | (e << 10) | ((m >> 13) & 0x3ff);
  }
  bf16ToF16 = t;
  return t;
}

// Returns a Uint32Array of packed f16 pairs (little-endian: lo | hi<<16).
export function packedF16(file, name) {
  const t = file.tensors.get(name);
  if (!t) throw new Error(`missing tensor ${name}`);
  const n = t.dims.reduce((a, b) => a * b, 1);
  if (n % 2) throw new Error(`odd element count for ${name}`);
  const out = new Uint32Array(n / 2);
  if (t.type === GGML_BF16) {
    const tbl = bf16Table();
    const u16 = new Uint16Array(file.buffer, file.dataStart + t.offset, n);
    for (let i = 0, j = 0; i < n; i += 2, j++) {
      out[j] = tbl[u16[i]] | (tbl[u16[i + 1]] << 16);
    }
  } else if (t.type === GGML_F32) {
    // only small f32 tensors take this path; going through the bf16 table keeps it simple
    const tbl = bf16Table();
    const u32v = new Uint32Array(file.buffer, file.dataStart + t.offset, n);
    for (let i = 0, j = 0; i < n; i += 2, j++) {
      out[j] = tbl[u32v[i] >>> 16] | (tbl[u32v[i + 1] >>> 16] << 16);
    }
  } else {
    throw new Error(`unsupported tensor type ${t.type} for ${name}`);
  }
  return out;
}

export function f32Tensor(file, name) {
  const t = file.tensors.get(name);
  if (!t) throw new Error(`missing tensor ${name}`);
  const n = t.dims.reduce((a, b) => a * b, 1);
  if (t.type !== GGML_F32) throw new Error(`tensor ${name} is not f32`);
  return new Float32Array(file.buffer.slice(file.dataStart + t.offset, file.dataStart + t.offset + n * 4));
}

// Load any supported tensor as Float32Array (CPU path). Layout matches GGUF storage
// order (row-major over dims[0] as the fastest-changing axis). 2D tensors keep
// shape [dims[0], dims[1]] in memory; callers that need (out, in) reshape like ref.py.
export function f32Any(file, name) {
  const t = file.tensors.get(name);
  if (!t) throw new Error(`missing tensor ${name}`);
  const n = t.dims.reduce((a, b) => a * b, 1);
  const out = new Float32Array(n);
  if (t.type === GGML_F32) {
    out.set(new Float32Array(file.buffer, file.dataStart + t.offset, n));
  } else if (t.type === GGML_BF16) {
    const u16 = new Uint16Array(file.buffer, file.dataStart + t.offset, n);
    const tmp = new Float32Array(1);
    const u32 = new Uint32Array(tmp.buffer);
    for (let i = 0; i < n; i++) {
      u32[0] = u16[i] << 16;
      out[i] = tmp[0];
    }
  } else {
    throw new Error(`unsupported tensor type ${t.type} for ${name}`);
  }
  return { data: out, dims: t.dims.slice() };
}
