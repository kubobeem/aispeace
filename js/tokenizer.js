// Byte-level BPE (GPT-2 style) built from the GGUF tokenizer arrays.

function buildByteMap() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const byteToChar = new Map();
  const charToByte = new Map();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCodePoint(cs[i]);
    byteToChar.set(bs[i], ch);
    charToByte.set(ch, bs[i]);
  }
  return { byteToChar, charToByte };
}

export class Tokenizer {
  constructor(kv) {
    this.tokens = kv["tokenizer.ggml.tokens"];
    this.types = kv["tokenizer.ggml.token_type"];
    this.merges = kv["tokenizer.ggml.merges"];
    this.bos = kv["tokenizer.ggml.bos_token_id"];
    this.eos = kv["tokenizer.ggml.eos_token_id"];
    const { byteToChar, charToByte } = buildByteMap();
    this.byteToChar = byteToChar;
    this.charToByte = charToByte;

    this.ranks = new Map();
    this.merges.forEach((m, i) => {
      const [a, b] = m.split(" ");
      this.ranks.set(a + b, i);
    });
    this.tokenToId = new Map();
    this.specials = new Map();
    for (let i = 0; i < this.tokens.length; i++) {
      const t = this.tokens[i];
      if (this.types && this.types[i] === 3) this.specials.set(t, i);
      else this.tokenToId.set(t, i);
    }
    this.decoder = new TextDecoder("utf-8", { fatal: false });
  }

  // text -> token ids (no specials)
  encode(text) {
    const bytes = new TextEncoder().encode(text);
    let chars = new Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) chars[i] = this.byteToChar.get(bytes[i]);
    const merged = this._bpe(chars);
    const ids = new Array(merged.length);
    for (let i = 0; i < merged.length; i++) {
      const id = this.tokenToId.get(merged[i]);
      if (id === undefined) throw new Error(`tokenizer: unknown piece ${JSON.stringify(merged[i])}`);
      ids[i] = id;
    }
    return ids;
  }

  _bpe(sym) {
    while (sym.length > 1) {
      let best = Infinity;
      let at = -1;
      for (let i = 0; i < sym.length - 1; i++) {
        const r = this.ranks.get(sym[i] + sym[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      const a = sym[at], b = sym[at + 1];
      const out = [];
      for (let i = 0; i < sym.length;) {
        if (i < sym.length - 1 && sym[i] === a && sym[i + 1] === b) { out.push(a + b); i += 2; }
        else { out.push(sym[i]); i++; }
      }
      sym = out;
    }
    return sym;
  }

  // token id -> raw bytes (specials produce nothing)
  bytesOf(id) {
    const t = this.tokens[id];
    if (this.types && this.types[id] === 3) return null;
    const out = new Uint8Array(t.length);
    for (let i = 0; i < t.length; i++) {
      const b = this.charToByte.get(t[i]);
      out[i] = b === undefined ? 63 : b; // '?'
    }
    return out;
  }

  decodeChunks(ids) {
    // decode pieces separately so multi-byte chars split across tokens still render
    const parts = [];
    for (const id of ids) {
      const b = this.bytesOf(id);
      if (b) parts.push(this.decoder.decode(b));
    }
    return parts;
  }

  decode(ids) {
    let out = "";
    for (const id of ids) {
      const b = this.bytesOf(id);
      if (b) out += this.decoder.decode(b);
    }
    return out;
  }

  special(name) {
    const id = this.specials.get(name);
    if (id === undefined) throw new Error(`missing special token ${name}`);
    return id;
  }
}
