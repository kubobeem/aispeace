# aispeace — nano-ja WebGPU demo

[nanochat-ja](https://huggingface.co/kubobeem/nano-ja-sft-kubobeem) (49M パラメータの日本語 SFT モデル) を
**ブラウザ内だけで** 動かす WebGPU デモです。サーバーもビルドもフレームワークも不要 — 素の JS + WGSL だけ。

- 重み (bf16 GGUF, 322 MB) を読み込み、推論はすべて GPU 上で実行
- 量子化なし、重みは f16 として GPU に常駐
- テキストは端末外に送信されません

```
index.html             UI (チャット)
js/main.js             読み込み・生成ループ・サンプリング
js/model.js            WebGPU のモデル実装 (バッファ / ディスパッチ / forward)
js/kernels.wgsl        WGSL カーネル (gemv, attention, RoPE, RMSNorm, ...)
js/gguf.js             GGUF パーサ + bf16→f16 変換
js/tokenizer.js        byte-level BPE (GGUF の語彙から構築)
tools/ref.py           NumPy リファレンス実装 (検証用)
tools/verify.mjs       WebGPU 出力と上記の突合 (CDP 経由)
```

## 動かす

ES modules と `fetch` を使うので `file://` では動きません。静的サーバー経由で開いてください。

```sh
python -m http.server 8000
# → http://localhost:8000/
```

初回に GGUF を読み込みます。方法は 2 つ:

1. **GGUF をドラッグ&ドロップ** — ローカルに置いた `nano-ja-sft.gguf` をそのまま
2. **URL から取得** — 公開 URL、または Hugging Face の private リポジトリの resolve URL + HF トークン

重みのリポジトリ (`kubobeem/nano-ja-sft-kubobeem`, `kubobeem/nano-ja-sft-weight`) は
現在整理中のため private にしています。トークンはその場で `Authorization: Bearer` に
だけ使い、保存も送信もしません。

GitHub Pages に置く場合も同じです (Settings → Pages → main / root)。ページ自体は重みを含みません。

## アーキテクチャ

`nanochat` の `gpt.py` の forward をそのまま移植しています:

- parameterless RMSNorm (Embedding 直後 / 各ブロック前 / 最終)
- residual + x0 ブレンド (`resid_lambda`, `x0_lambda`)、smear (`smear_lambda`, `smear_gate` 24ch)
- 6 層、n_embd 768、12 head × head_dim 64、FFN 3072、語彙 32768、context 2048
- value embedding を layer 1/3/5 の V に加算 (`ve_gate` でゲート)
- RoPE (half-split, theta=100000) → QK norm (q/k 両方 ×1.2) → causal attention (scale 1/√64)
- relu² MLP、layer 3 後の backout (`backout_lambda`)、logit softcap 15

実装上のポイント:

- **bf16 は WGSL に無い** ので、読み込み時に f16 へ変換し `u32` に 2 個ずつパックして転送。
  カーネル側は `unpack2x16float` で復号するため `shader-f16` 機能は不要 (どの WebGPU でも動く)
- アクティベーションは f32、重みだけ f16。attention の K/V キャッシュのみ f16 パック
- 1 バッファ = 1 テンソル。最大でも 50 MB なので既定の `maxBufferSize` (256 MB) に収まる
- プロンプトは 1 回のディスパッチ列で prefill (最大 512 トークン)、以降は 1 トークンずつ decode
- サンプリング (temperature / top-k) と softcap は CPU 側。logits は 128 KB の readback

## 検証

同じ GGUF を NumPy で読む独立実装 (`tools/ref.py`) と、GPU の出力を突き合わせています。

```sh
python tools/ref.py model.gguf
# tokens  [32759, 32760, 31946, 2628, 1407, 266, 32761, 32762]
# top10   [(110, 13.6441), (2398, 11.3133), (31540, 10.4543), ...]
```

デモ側は `?test=<gguf の URL>` を付けると、同じプロンプトで prefill して top-10 を
`#result` に書き出します (`document.title` が `OK ...` になったら完了)。

```sh
chrome --remote-debugging-port=9222 --user-data-dir=%TEMP%\wgpu about:blank
node tools/verify.mjs "http://localhost:8000/?test=/model.gguf"
```

結果は top-10 の ID も logit も小数 4 桁まで一致します (誤差は f16 丸めのみ)。
参照実装の greedy 先頭トークン `n` (id 110) は、以前 llama.cpp / torch で確認した出力と一致しています。

## 性能

RTX 4060 / Chrome 155 での実測:

- 初回トークン ~30–50 ms (8 トークンの prefill)
- 生成 ~22 tok/s (1 トークンあたり ~45 ms)
- VRAM ~390 MB (重み 322 MB + KV キャッシュ 38 MB + アクティベーション)

## 制限

- **bf16 GGUF 専用**。Q4_K_M などの量子化 GGUF は読めません (dequant カーネルが別途必要)
- コンテキストは 2048、1 回の prefill は 512 トークンまで
- モデルが 49M なので応答品質はそれなり。数学的な再現性が目的のデモです
- `shader-f16` を使わない代わりに、重みは f16 精度 (元の bf16 と同等以上)
