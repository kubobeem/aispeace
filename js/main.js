import { parseGGUF } from "./gguf.js";
import { Tokenizer } from "./tokenizer.js";
import { createDevice, NanochatGPU, finalizeLogits, sampleToken } from "./model.js";

const $ = (id) => document.getElementById(id);
const chat = $("chat");

// The demo only ever runs this one model. Loaded silently in the background:
// ?model=<url> wins, then ./model.gguf next to index.html, then the HF copy
// (private repo: pass ?token=... once or set localStorage.hf_token).
const HF_GGUF = "https://huggingface.co/kubobeem/nano-ja-sft-kubobeem/resolve/main/nano-ja-sft.gguf";
const params = new URLSearchParams(location.search);
const token = params.get("token") || localStorage.getItem("hf_token") || "";
const sources = [params.get("model"), "./model.gguf", HF_GGUF].filter(Boolean);

let model = null;
let tok = null;
let conversation = [];
let busy = false;

const meta = (text) => { $("meta").textContent = text; };

async function loadModel(device, info, buffer) {
  const file = parseGGUF(buffer);
  const arch = file.kv["general.architecture"];
  if (arch !== "nanochat") throw new Error(`アーキテクチャが違います: ${arch}`);
  tok = new Tokenizer(file.kv);
  const shaders = await (await fetch("./js/kernels.wgsl")).text();
  const gpu = new NanochatGPU(device, shaders);
  await gpu.upload(file);
  await gpu.prefill([tok.bos]); // warm up so the first message is not penalised by pipeline creation
  gpu.reset();
  model = gpu;
  conversation = [tok.bos];
  $("input").disabled = false;
  $("send").disabled = false;
  $("input").focus();
  meta(`${info.description || info.vendor || "WebGPU"} · bf16 ${(buffer.byteLength / 1048576).toFixed(0)} MB`);
  return gpu;
}

async function boot() {
  let device, info;
  try {
    ({ device, info } = await createDevice());
  } catch (e) {
    meta(e.message);
    return;
  }
  let last;
  for (const src of sources) {
    try {
      const res = await fetch(src, token && src.startsWith("http") ? { headers: { Authorization: `Bearer ${token}` } } : undefined);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await loadModel(device, info, await res.arrayBuffer());
      return;
    } catch (e) {
      last = e;
      console.warn(`model from ${src}: ${e.message}`);
    }
  }
  meta(`モデルを読み込めません (${last ? last.message : "no source"})`);
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
  const assistantStart = tok.special("<|assistant_start|>");
  const assistantEnd = tok.special("<|assistant_end|>");

  const newIds = [tok.special("<|user_start|>"), ...tok.encode(userText), tok.special("<|user_end|>"), assistantStart];
  conversation.push(...newIds);

  try {
    const t0 = performance.now();
    let raw = await model.prefill(newIds);
    const firstLatency = performance.now() - t0;
    let text = "";
    let produced = 0;
    const genStart = performance.now();
    for (let i = 0; i < maxTokens; i++) {
      const id = sampleToken(finalizeLogits(raw), temp, topK);
      produced++;
      conversation.push(id);
      if (id === assistantEnd) break;
      text += tok.decode([id]);
      bubble.textContent = text;
      chat.scrollTop = chat.scrollHeight;
      if (produced >= maxTokens) { conversation.push(assistantEnd); break; }
      raw = await model.decode(id);
    }
    const dt = (performance.now() - genStart) / 1000;
    bubble.classList.remove("typing");
    meta(`${produced} tok · ${(produced / dt).toFixed(1)} tok/s · 初回 ${Math.round(firstLatency)} ms · kv ${model.kvPos}/2048`);
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
  meta("会話をリセットしました");
}

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

// deterministic self-check: /?test=<gguf-url> prints the top-10 logits for the
// verified prompt so they can be compared against tools/ref.py.
if (params.has("test")) {
  (async () => {
    try {
      const url = params.get("test");
      const { device, info } = await createDevice();
      const res = await fetch(url, token && url.startsWith("http") ? { headers: { Authorization: `Bearer ${token}` } } : undefined);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const gpu = await loadModel(device, info, await res.arrayBuffer());
      const ids = params.get("ids")
        ? params.get("ids").split(",").map(Number)
        : [32759, 32760, 31946, 2628, 1407, 266, 32761, 32762];
      const t0 = performance.now();
      const raw = await gpu.prefill(ids);
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
      $("result").textContent = JSON.stringify({ ok: false, error: String((e && e.message) || e) });
      document.title = `FAIL ${e.message}`;
    }
  })();
} else {
  boot();
}

window.__demo = {
  loadModel,
  get model() { return model; },
  get tok() { return tok; },
  finalizeLogits,
  sampleToken,
};
