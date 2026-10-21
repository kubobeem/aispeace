// nano-ja WebGPU kernels. Weights live in storage buffers as packed f16 pairs
// (array<u32>, decoded with unpack2x16float) so the demo does not need shader-f16.
// Activations are f32 buffers laid out [D, T] (channel-major, T contiguous).
// The loader splits this file on "//@kernel <entry>" and compiles one module per
// kernel, because WGSL forbids duplicate @group/@binding pairs inside one module.

//@prelude
struct U {
  T: u32,
  T0: u32,
  M: u32,
  K: u32,
  a: f32,
  b: f32,
  c: f32,
  d: f32,
};
const RED: u32 = 128u;

//@kernel gemv
// y[m, t] = sum_k A[m, k] * x[k, t]  (A: packed f16 [M, K])
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> A: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> shRow: array<f32, 3072>;
var<workgroup> shRed: array<f32, 128>;

@compute @workgroup_size(128)
fn gemv(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let nK = u.K;
  let nT = u.T;
  let m = wid.x;
  let half = nK / 2u;
  var i = lid.x;
  loop {
    if (i >= half) { break; }
    let v = unpack2x16float(A[m * half + i]);
    shRow[i * 2u] = v.x;
    shRow[i * 2u + 1u] = v.y;
    i += RED;
  }
  workgroupBarrier();
  for (var t = 0u; t < nT; t += 1u) {
    var acc = 0.0;
    var k = lid.x;
    loop {
      if (k >= nK) { break; }
      acc = acc + shRow[k] * X[k * nT + t];
      k += RED;
    }
    shRed[lid.x] = acc;
    workgroupBarrier();
    var s = RED / 2u;
    loop {
      if (s == 0u) { break; }
      if (lid.x < s) { shRed[lid.x] = shRed[lid.x] + shRed[lid.x + s]; }
      workgroupBarrier();
      s = s / 2u;
    }
    if (lid.x == 0u) { Y[m * nT + t] = shRed[0]; }
    workgroupBarrier();
  }
}

//@kernel embed
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> tbl: array<u32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> emb: array<f32>;

@compute @workgroup_size(128)
fn embed(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nT = u.T;
  let tok = ids[t];
  var i = lid.x;
  loop {
    if (i >= 384u) { break; }
    let v = unpack2x16float(tbl[tok * 384u + i]);
    emb[(i * 2u) * nT + t] = v.x;
    emb[(i * 2u + 1u) * nT + t] = v.y;
    i += RED;
  }
}

//@kernel smear
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> smIn: array<f32>;
@group(0) @binding(2) var<storage, read> smW: array<f32>;
@group(0) @binding(3) var<storage, read_write> smPrev: array<f32>;
@group(0) @binding(4) var<storage, read_write> smOut: array<f32>;
var<workgroup> shGate: f32;

@compute @workgroup_size(128)
fn smear(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nT = u.T;
  if (lid.x == 0u) {
    var acc = 0.0;
    for (var j = 0u; j < 24u; j += 1u) { acc = acc + smW[j] * smIn[j * nT + t]; }
    shGate = u.a / (1.0 + exp(-acc));
  }
  workgroupBarrier();
  let g = shGate;
  var d = lid.x;
  loop {
    if (d >= 768u) { break; }
    var v = smIn[d * nT + t];
    if (nT > 1u) {
      if (t > 0u) { v = v + g * smIn[d * nT + t - 1u]; }
    } else {
      v = v + g * smPrev[d];
    }
    smOut[d * nT + t] = v;
    if (t == nT - 1u) { smPrev[d] = smIn[d * nT + t]; }
    d += RED;
  }
}

//@kernel blend
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> blX: array<f32>;
@group(0) @binding(2) var<storage, read> blX0: array<f32>;

@compute @workgroup_size(128)
fn blend(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nT = u.T;
  let nK = u.K;
  var d = lid.x;
  loop {
    if (d >= nK) { break; }
    let i = d * nT + t;
    blX[i] = u.a * blX[i] + u.b * blX0[i];
    d += RED;
  }
}

//@kernel rmsnorm
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> nmX: array<f32>;
@group(0) @binding(2) var<storage, read> nmW: array<f32>;
@group(0) @binding(3) var<storage, read_write> nmY: array<f32>;
var<workgroup> shRed: array<f32, 128>;

@compute @workgroup_size(128)
fn rmsnorm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nT = u.T;
  let nK = u.K;
  var ss = 0.0;
  var d = lid.x;
  loop {
    if (d >= nK) { break; }
    let v = nmX[d * nT + t];
    ss = ss + v * v;
    d += RED;
  }
  shRed[lid.x] = ss;
  workgroupBarrier();
  var s = RED / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { shRed[lid.x] = shRed[lid.x] + shRed[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let rr = inverseSqrt(shRed[0] / f32(nK) + u.b);
  d = lid.x;
  loop {
    if (d >= nK) { break; }
    let i = d * nT + t;
    var w = 1.0;
    if (u.c > 0.5) { w = nmW[d]; }
    nmY[i] = nmX[i] * rr * w * u.a;
    d += RED;
  }
}

//@kernel rmsnorm64
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> qkX: array<f32>;
@group(0) @binding(2) var<storage, read> qkW: array<f32>;
var<workgroup> shRed: array<f32, 64>;

@compute @workgroup_size(64)
fn rmsnorm64(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = wid.y;
  let nT = u.T;
  let i = (h * 64u + lid.x) * nT + t;
  let v = qkX[i];
  shRed[lid.x] = v * v;
  workgroupBarrier();
  var s = 32u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { shRed[lid.x] = shRed[lid.x] + shRed[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let rr = inverseSqrt(shRed[0] / 64.0 + u.b);
  qkX[i] = v * rr * qkW[lid.x] * u.a;
}

//@kernel rope
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> rpX: array<f32>;

@compute @workgroup_size(32)
fn rope(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = wid.y;
  let nT = u.T;
  let pos = f32(u.T0 + t);
  let inv = 1.0 / pow(100000.0, f32(lid.x) / 32.0);
  let ang = pos * inv;
  let c = cos(ang);
  let s = sin(ang);
  let i1 = (h * 64u + lid.x) * nT + t;
  let i2 = i1 + 32u * nT;
  let x1 = rpX[i1];
  let x2 = rpX[i2];
  rpX[i1] = x1 * c + x2 * s;
  rpX[i2] = x2 * c - x1 * s;
}

//@kernel storekv
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> skK: array<f32>;
@group(0) @binding(2) var<storage, read> skV: array<f32>;
@group(0) @binding(3) var<storage, read_write> skKC: array<u32>;
@group(0) @binding(4) var<storage, read_write> skVC: array<u32>;

@compute @workgroup_size(128)
fn storekv(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nT = u.T;
  let pos = u.T0 + t;
  var p = lid.x;
  loop {
    if (p >= 384u) { break; }
    let d0 = p * 2u;
    let kc = pack2x16float(vec2<f32>(skK[d0 * nT + t], skK[(d0 + 1u) * nT + t]));
    let vc = pack2x16float(vec2<f32>(skV[d0 * nT + t], skV[(d0 + 1u) * nT + t]));
    skKC[pos * 384u + p] = kc;
    skVC[pos * 384u + p] = vc;
    p += RED;
  }
}

//@kernel attention
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> atQ: array<f32>;
@group(0) @binding(2) var<storage, read> atKC: array<u32>;
@group(0) @binding(3) var<storage, read> atVC: array<u32>;
@group(0) @binding(4) var<storage, read_write> atO: array<f32>;
var<workgroup> atS: array<f32, 2048>;
var<workgroup> shRed: array<f32, 64>;

@compute @workgroup_size(64)
fn attention(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = wid.y;
  let nT = u.T;
  let nkv = u.T0 + t + 1u;
  var j = lid.x;
  loop {
    if (j >= nkv) { break; }
    var acc = 0.0;
    for (var d = 0u; d < 32u; d += 1u) {
      let kv = unpack2x16float(atKC[j * 384u + h * 32u + d]);
      acc = acc + atQ[(h * 64u + d * 2u) * nT + t] * kv.x
                + atQ[(h * 64u + d * 2u + 1u) * nT + t] * kv.y;
    }
    atS[j] = acc * u.a;
    j += 64u;
  }
  workgroupBarrier();
  var mx = -1.0e30;
  j = lid.x;
  loop { if (j >= nkv) { break; } mx = max(mx, atS[j]); j += 64u; }
  shRed[lid.x] = mx;
  workgroupBarrier();
  var s = 32u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { shRed[lid.x] = max(shRed[lid.x], shRed[lid.x + s]); }
    workgroupBarrier();
    s = s / 2u;
  }
  let m = shRed[0];
  var sum = 0.0;
  j = lid.x;
  loop { if (j >= nkv) { break; } sum = sum + exp(atS[j] - m); j += 64u; }
  workgroupBarrier();
  shRed[lid.x] = sum;
  workgroupBarrier();
  s = 32u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { shRed[lid.x] = shRed[lid.x] + shRed[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let inv = 1.0 / shRed[0];
  let d = lid.x;
  var outv = 0.0;
  var jj = 0u;
  loop {
    if (jj >= nkv) { break; }
    let p = exp(atS[jj] - m) * inv;
    let pc = jj * 384u + h * 32u + (d / 2u);
    if ((d & 1u) == 0u) { outv = outv + p * unpack2x16float(atVC[pc]).x; }
    else { outv = outv + p * unpack2x16float(atVC[pc]).y; }
    jj += 1u;
  }
  atO[(h * 64u + d) * nT + t] = outv;
}

//@kernel vebind
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> vgX: array<f32>;
@group(0) @binding(2) var<storage, read> vgW: array<f32>;
@group(0) @binding(3) var<storage, read_write> vgG: array<f32>;

@compute @workgroup_size(32)
fn vebind(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nT = u.T;
  let h = lid.x;
  if (h < 12u) {
    var acc = 0.0;
    for (var j = 0u; j < 12u; j += 1u) { acc = acc + vgW[h * 12u + j] * vgX[j * nT + t]; }
    vgG[h * nT + t] = 3.0 / (1.0 + exp(-acc));
  }
}

//@kernel veadd
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> vaV: array<f32>;
@group(0) @binding(2) var<storage, read> vaTable: array<u32>;
@group(0) @binding(3) var<storage, read> vaIds: array<u32>;
@group(0) @binding(4) var<storage, read> vaG: array<f32>;

@compute @workgroup_size(32)
fn veadd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = wid.y;
  let nT = u.T;
  let d0 = lid.x * 2u;
  let tok = vaIds[t];
  let v = unpack2x16float(vaTable[tok * 384u + h * 32u + lid.x]);
  let g = vaG[h * nT + t];
  let i1 = (h * 64u + d0) * nT + t;
  vaV[i1] = vaV[i1] + g * v.x;
  vaV[i1 + nT] = vaV[i1 + nT] + g * v.y;
}

//@kernel relu2
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> rlH: array<f32>;

@compute @workgroup_size(128)
fn relu2(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nT = u.T;
  let nK = u.K;
  var d = lid.x;
  loop {
    if (d >= nK) { break; }
    let i = d * nT + t;
    let v = max(rlH[i], 0.0);
    rlH[i] = v * v;
    d += RED;
  }
}

//@kernel add
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> adX: array<f32>;
@group(0) @binding(2) var<storage, read> adY: array<f32>;

@compute @workgroup_size(128)
fn add(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nT = u.T;
  let nK = u.K;
  var d = lid.x;
  loop {
    if (d >= nK) { break; }
    let i = d * nT + t;
    adX[i] = adX[i] + adY[i] * u.a;
    d += RED;
  }
}

//@kernel subscaled
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> sbX: array<f32>;
@group(0) @binding(2) var<storage, read> sbB: array<f32>;

@compute @workgroup_size(128)
fn subscaled(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nT = u.T;
  var d = lid.x;
  loop {
    if (d >= 768u) { break; }
    let i = d * nT + t;
    sbX[i] = sbX[i] - u.a * sbB[i];
    d += RED;
  }
}

//@kernel copycol
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> ccSrc: array<f32>;
@group(0) @binding(2) var<storage, read_write> ccDst: array<f32>;

@compute @workgroup_size(128)
fn copycol(@builtin(local_invocation_id) lid: vec3<u32>) {
  let nT = u.T;
  let col = u.K;
  var d = lid.x;
  loop {
    if (d >= 768u) { break; }
    ccDst[d] = ccSrc[d * nT + col];
    d += RED;
  }
}
