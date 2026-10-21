// Drives an already-running Chrome (--remote-debugging-port=9222) to the demo's
// self-check page and prints the result, so the WebGPU output can be diffed
// against tools/ref.py.
//
//   chrome --remote-debugging-port=9222 --user-data-dir=%TEMP%\wgpu about:blank
//   node tools/verify.mjs http://localhost:8000/?test=/model.gguf
const [, , url = "http://localhost:8000/?test=/model.gguf"] = process.argv;

const list = await (await fetch("http://127.0.0.1:9222/json/list")).json();
const page = list.find((t) => t.type === "page");
if (!page) throw new Error("no page target; is Chrome running with --remote-debugging-port=9222 ?");

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const send = (method, params = {}) => new Promise((res) => {
  const mid = ++id;
  pending.set(mid, { res });
  ws.send(JSON.stringify({ id: mid, method, params }));
});
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id).res(m.result); pending.delete(m.id); }
  else if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") {
    console.log("[console error]", m.params.args.map((a) => a.value ?? a.description ?? "").join(" "));
  }
});
await new Promise((r) => ws.addEventListener("open", r));
await send("Runtime.enable");
await send("Page.enable");
await send("Network.enable");
await send("Network.setCacheDisabled", { cacheDisabled: true });
await send("Page.navigate", { url });

const deadline = Date.now() + 600000;
let title = "";
while (Date.now() < deadline) {
  const r = await send("Runtime.evaluate", { expression: "document.title", returnByValue: true });
  title = r?.result?.value || "";
  if (title.startsWith("OK") || title.startsWith("FAIL")) break;
  await new Promise((r) => setTimeout(r, 1500));
}
const res = await send("Runtime.evaluate", { expression: "document.getElementById('result').textContent", returnByValue: true });
console.log(title);
console.log(res?.result?.value);
ws.close();
