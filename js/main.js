import { parseGGUF } from "./gguf.js";
import { Tokenizer } from "./tokenizer.js";
import { createDevice, NanochatGPU, finalizeLogits, sampleToken } from "./model.js";

const $ = (id) => document.getElementById(id);
const chat = $("chat");

let model = null;
let tok = null;
let conversation = [];
let busy = false;
let firstLatency = 0;

function setStatus(text, err = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("err", err);
}

function setProgress(frac) {
  $("prog").style.width = `${Math.round(frac * 100)}%`;
}

async function fetchGGUF(url, token, onProgress) {
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const res = await fetch(url, { headers });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(total ? got / total : 0, got, total);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

async function loadModel(buffer, label) {
  try {
    setStatus("GGUF を解析中…");
    setProgress(0.02);
    const file = parseGGUF(buffer);
    const arch = file.kv["general.architecture"];
    if (arch !== "nanochat") throw new Error(`アーキテクチャが違います: ${arch}`);
    tok = new Tokenizer(file.kv);
    const { device, info } = await createDevice();
    const shaders = await (await fetch("./js/kernels.wgsl")).text();
    const gpu = new NanochatGPU(device, shaders);
    setStatus("重みを GPU に転送中…");
    await gpu.upload(file, (frac, name) => {
      setProgress(0.05 + frac * 0.93);
      if (frac * 100 % 10 < 0.5) setStatus(`重みを GPU に転送中… ${name}`);
    });
    // warm up so the first user message is not penalised by pipeline creation
    setStatus("ウォームアップ中…");
    await gpu.prefill([tok.bos]);
    gpu.reset();
    setProgress(1);
    model = gpu;
    conversation = [tok.bos];
    $("overlay").classList.add("hidden");
    $("input").disabled = false;
    $("send").disabled = false;
    $("input").focus();
    $("meta").textContent = `${info.description || info.vendor || "WebGPU"} · ${(buffer.byteLength / 1048576).toFixed(0)} MB · ${label}`;
    setStatus("準備完了");
  } catch (e) {
    console.error(e);
    setStatus(`読み込み失敗: ${e.message}`, true);
    setProgress(0);
  }
}

function addMessage(role, text = "") {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "You" : "AI";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.append(who, bubble);
  chat.append(wrap);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

async function generate(userText) {
  if (busy || !model) return;
  busy = true;
  $("send").disabled = true;
  $("input").disabled = true;
  addMessage("user", userText);
  const bubble = addMessage("bot");
  bubble.classList.add("typing");

  const temp = Number($("temp").value);
  const topK = Number($("topk").value);
  const maxTokens = Number($("maxt").value);
  const assistantEnd = tok.special("<|assistant_end|>");

  const newIds = [tok.special("<|user_start|>"), ...tok.encode(userText), tok.special("<|user_end|>")];
  conversation.push(...newIds);
  const promptIds = [...newIds, tok.special("<|assistant_start|>")];

  try {
    const t0 = performance.now();
    let raw = await model.prefill(promptIds);
    conversation.push(tok.special("<|assistant_start|>"));
    firstLatency = performance.now() - t0;
    let text = "";
    let produced = 0;
    const genStart = performance.now();
    for (let i = 0; i < maxTokens; i++) {
      const id = sampleToken(finalizeLogits(raw), temp, topK);
      if (id === assistantEnd) { conversation.push(id); break; }
      produced++;
      conversation.push(id);
      text += tok.decode([id]);
      bubble.textContent = text;
      chat.scrollTop = chat.scrollHeight;
      if (produced >= maxTokens) { conversation.push(assistantEnd); break; }
      raw = await model.decode(id);
    }
    const dt = (performance.now() - genStart) / 1000;
    bubble.classList.remove("typing");
    $("meta").textContent =
      `${produced} tok · ${(produced / dt).toFixed(1)} tok/s · 初回 ${Math.round(firstLatency)} ms · kv ${model.kvPos}/2048`;
  } catch (e) {
    console.error(e);
    bubble.classList.remove("typing");
    bubble.textContent = `エラー: ${e.message}`;
  } finally {
    busy = false;
    $("send").disabled = false;
    $("input").disabled = false;
    $("input").focus();
  }
}

function resetChat() {
  if (!model) return;
  model.reset();
  conversation = [tok.bos];
  chat.querySelectorAll(".msg").forEach((n) => n.remove());
  $("meta").textContent = "会話をリセットしました";
}

// ---- wiring ----
$("drop").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  setStatus("ファイルを読み込み中…");
  await loadModel(await f.arrayBuffer(), f.name);
});
for (const ev of ["dragover", "dragenter"]) {
  $("drop").addEventListener(ev, (e) => { e.preventDefault(); $("drop").classList.add("over"); });
}
for (const ev of ["dragleave", "drop"]) {
  $("drop").addEventListener(ev, (e) => { e.preventDefault(); $("drop").classList.remove("over"); });
}
$("drop").addEventListener("drop", async (e) => {
  const f = e.dataTransfer.files[0];
  if (!f) return;
  setStatus("ファイルを読み込み中…");
  await loadModel(await f.arrayBuffer(), f.name);
});
$("fetch").addEventListener("click", async () => {
  const url = $("url").value.trim();
  if (!url) return;
  const token = $("token").value.trim();
  try {
    const buf = await fetchGGUF(url, token, (frac, got, total) => {
      setProgress(frac * 0.05);
      setStatus(`ダウンロード ${(got / 1048576).toFixed(0)} / ${(total / 1048576).toFixed(0)} MB`);
    });
    await loadModel(buf, "HF");
  } catch (e) {
    setStatus(`ダウンロード失敗: ${e.message}`, true);
  }
});

$("send").addEventListener("click", () => {
  const v = $("input").value.trim();
  if (!v) return;
  $("input").value = "";
  $("input").style.height = "auto";
  generate(v);
});
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("send").click();
  }
});
$("input").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
});
$("reset").addEventListener("click", resetChat);
$("samples").addEventListener("click", (e) => {
  const p = e.target.dataset && e.target.dataset.p;
  if (p) generate(p);
});
for (const [id, out] of [["temp", "tempv"], ["topk", "topkv"], ["maxt", "maxtv"]]) {
  $(id).addEventListener("input", () => { $(out).textContent = $(id).value; });
}

// deterministic self-check: /?test=<gguf-url>[&token=..] prints the top-10 logits
// for the verified prompt so they can be compared against the NumPy reference.
const params = new URLSearchParams(location.search);
if (params.has("test")) {
  (async () => {
    try {
      const url = params.get("test");
      const buf = await fetchGGUF(url, params.get("token") || "", () => {});
      await loadModel(buf, "autotest");
      const ids = params.get("ids")
        ? params.get("ids").split(",").map(Number)
        : [32759, 32760, 31946, 2628, 1407, 266, 32761, 32762];
      const t0 = performance.now();
      const raw = await model.prefill(ids);
      const ms = performance.now() - t0;
      const lg = finalizeLogits(raw);
      const top = Array.from(lg.keys()).sort((a, b) => lg[b] - lg[a]).slice(0, 10);
      const payload = {
        ok: true,
        ms: Math.round(ms),
        tokens: ids,
        top10: top.map((i) => [i, Number(lg[i].toFixed(4))]),
        text: top.slice(0, 1).map((i) => tok.decode([i])),
      };
      $("result").textContent = JSON.stringify(payload);
      document.title = `OK ${top.join(",")}`;
    } catch (e) {
      $("result").textContent = JSON.stringify({ ok: false, error: String(e && e.message || e) });
      document.title = `FAIL ${e.message}`;
    }
  })();
}

window.__demo = {
  loadModel,
  get model() { return model; },
  get tok() { return tok; },
  finalizeLogits,
  sampleToken,
};
