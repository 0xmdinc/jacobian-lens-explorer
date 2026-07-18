# Jacobian Lens Explorer

An interactive web UI on top of Anthropic's [`jlens`](https://github.com/anthropics/jacobian-lens)
(companion code for *"Verbalizable Representations Form a Global Workspace in Language Models"*).

> Built by **[0xmd](https://github.com/0xmdinc)** to let anyone try `jlens` the easy way: no notebooks, no setup wrestling.
> Fit a lens once, then just type prompts in your browser and watch the model think.

Type a prompt in your browser and see two things side by side:

1. **The model's actual answer** for that prompt (its top next-token predictions plus a short greedy continuation).
2. **An interactive layer x position grid** showing what each internal activation is "disposed to make the model say", layer by layer, even for words that never appear in your text.

The page is a single dark-themed page: type, hit Analyze, and results appear inline.

## Requirements

- **Linux** with an **NVIDIA GPU** and a working CUDA driver (`nvidia-smi` should work).
- **Python 3.10+**.
- **VRAM**: about `2 x parameter_count` GB for a bf16 model, plus a few GB for the fit.
  - `Qwen/Qwen3-8B` (default) needs roughly **18-20 GB** (fits a 24 GB card such as an RTX 3090/4090).
  - For smaller cards, use a smaller model (see [Low VRAM](#low-vram)).
- **Disk**: the model is downloaded from Hugging Face on first use (~16 GB for an 8B model).

### Model compatibility (important)

The lens is fit by backpropagating a Jacobian **through every layer**, so the model has to be one
you can take gradients through end to end (a normal attention transformer).

- **Works**: dense standard-attention models like `Qwen/Qwen3-8B`, the smaller `Qwen3` dense
  models, and Llama / Mistral style models.
- **Mixture-of-experts (MoE)** models (e.g. `Qwen/Qwen3-30B-A3B`) work too. The router picks a
  few experts per token, but for any given input the forward pass is still differentiable through
  the experts it chose, which is all the lens needs. They are just big and slow. (The stock
  Hugging Face MoE code is differentiable; some fused/optimized MoE kernels are not, same caveat
  as quantized weights below.)
- **Does not work**: hybrid **mamba / linear-attention** models such as `Qwen3.5-*` and
  `Qwen3-Next-*`. Their gated-delta layers blow up the autograd pass with a CUDA error.
- **Does not work for fitting**: AWQ / GPTQ **quantized** builds, since you cannot get gradients
  through the quant kernels. Fit on the full-precision (bf16) weights.

## Install

```bash
git clone https://github.com/0xmdinc/jacobian-lens-explorer.git
cd jacobian-lens-explorer

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `import torch; torch.cuda.is_available()` prints `False`, pip installed a CPU-only PyTorch.
Reinstall a CUDA build from the [PyTorch selector](https://pytorch.org/get-started/locally/), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 1. Fit a lens

This downloads the model (first time only) and fits a lens, saving it to `out/lens.pt`.

```bash
python fit_lens.py --model Qwen/Qwen3-8B --out out/lens.pt
```

- The default is a small, fast **demo** fit (6 prompts, 8 layers). It is noisy but enough to explore.
- For a sharper lens, add more full sentences to `prompts.txt` (the paper suggests ~100-1000)
  and/or raise `--num-source-layers`. See `python fit_lens.py --help`.
- Gated by a private/faster download? Set a token: `export HF_TOKEN=hf_xxx`.

## 2. Run the explorer

```bash
python serve.py --model Qwen/Qwen3-8B --lens out/lens.pt
```

It loads the model and lens once, then serves the UI. Open:

```
http://localhost:5005
```

Type a sentence (a full sentence works best) and click **Analyze**.

## How to read it

- **Answer panel** (top): the model's real next-word prediction with probability bars, and a short continuation.
- **Grid**: columns are the words of your sentence, rows are the model's layers (bottom = early, top = late).
  Each cell is the top word the lens reads there.
  - **Hover** a cell for the full top-10 behind it.
  - **Click** a cell to pin a word; every occurrence of that word lights up in a color, and a chart tracks its rank across layers.
- **By Layer** panel: one word, all layers. **By Pos** panel: one layer, all words.

## Low VRAM

Use a smaller dense `Qwen3` model, for example:

```bash
python fit_lens.py --model Qwen/Qwen3-1.7B --out out/lens-1.7b.pt
python serve.py    --model Qwen/Qwen3-1.7B --lens out/lens-1.7b.pt
```

`Qwen/Qwen3-1.7B` runs in a few GB of VRAM and downloads in minutes, which is the fastest way to try the whole pipeline. `Qwen/Qwen3-0.6B` is smaller still if you are really tight on memory.

## Configuration

Both scripts take `--model`, and the same values can be set via environment variables:

| Variable      | Meaning                     | Default          |
|---------------|-----------------------------|------------------|
| `JLENS_MODEL` | HF model id or local path   | `Qwen/Qwen3-8B`  |
| `JLENS_LENS`  | path to the fitted lens     | `out/lens.pt`    |
| `JLENS_PORT`  | web server port             | `5005`           |
| `HF_TOKEN`    | Hugging Face access token   | (unset)          |

`serve.py` also accepts `--host`, `--port`, `--device` (`cuda` / `cuda:0` / `auto`), and `--dtype`.

## Project layout

```
serve.py          Flask web app (loads model + lens, serves the UI)
fit_lens.py       fits a lens and saves it to out/lens.pt
prompts.txt       sentences used to fit the lens (edit to improve quality)
assets/d3.min.js  bundled d3 (the viewer runs fully offline)
static/dark.css   dark theme for the embedded viewer
out/              fitted lenses (git-ignored; created by fit_lens.py)
```

## Notes

- The viewer is served locally and, thanks to the bundled `d3.min.js`, needs no internet at page-build time.
- The lens is a linear approximation; the grid is best treated as intuition, not ground truth. The answer panel is the model's real output.

## Credits and license

- Built by **[0xmd](https://github.com/0xmdinc)** to make `jlens` easy to try.
- Built on Anthropic's `jlens` (Apache-2.0): https://github.com/anthropics/jacobian-lens
- d3.js (ISC) is bundled under `assets/`.
- This wrapper is provided as-is for exploration and education.
