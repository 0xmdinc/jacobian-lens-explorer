#!/usr/bin/env python3
"""
Fit a Jacobian lens for a standard-attention causal LM and save it to disk.

The lens is the average input->output Jacobian of the residual stream, computed
by backpropagating through the model, so the model MUST be a standard-attention
decoder transformer. Hybrid mamba / linear-attention models (Qwen3.5, Qwen3-Next,
etc.) are NOT supported -- their gated-delta layers break the autograd pass.

Downloading the weights (first run) can be large (an 8B model in bf16 is ~16 GB).

Usage:
    python fit_lens.py                                   # Qwen/Qwen3-8B -> out/lens.pt
    python fit_lens.py --model Qwen/Qwen3-1.7B --out out/lens.pt   # smaller / less VRAM
    python fit_lens.py --prompts my_prompts.txt --dim-batch 4 --max-seq-len 96
"""
import argparse
import os

import torch
import transformers
import jlens

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PROMPTS = os.path.join(HERE, "prompts.txt")


def load_prompts(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        if lines:
            return lines
    # fallback built-in prompts (each must tokenize to > 17 tokens)
    return [
        "The history of the Roman Empire spans over a thousand years, beginning with the legendary founding of the city of Rome and ending with the fall of Constantinople in the fifteenth century.",
        "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food from carbon dioxide and water, releasing oxygen as a byproduct of the reaction.",
        "The theory of general relativity, developed by Albert Einstein in the early twentieth century, describes gravity not as a force but as the curvature of spacetime caused by mass and energy.",
        "In computer science, a sorting algorithm arranges the elements of a list into a particular order, and well known examples include quicksort, merge sort, and heap sort, each with different tradeoffs.",
        "The human circulatory system consists of the heart, blood vessels, and blood, working together to carry oxygen, nutrients, and hormones to cells throughout the body while removing metabolic waste.",
        "The currency used in the country shaped like a boot in southern Europe was the lira for many decades before it was replaced by the euro at the beginning of the twenty first century.",
    ]


def main():
    ap = argparse.ArgumentParser(description="Fit a Jacobian lens")
    ap.add_argument("--model", default=os.environ.get("JLENS_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "lens.pt"))
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS,
                    help="text file, one prompt per line (each should be a full sentence)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--num-source-layers", type=int, default=8,
                    help="how many evenly-spaced layers to fit a lens at (fewer = faster / less memory)")
    ap.add_argument("--dim-batch", type=int, default=2,
                    help="rows of the Jacobian computed per backward pass; higher = faster but more VRAM")
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--checkpoint", default=None, help="optional resumable checkpoint path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dtype = getattr(torch, args.dtype)

    print(f"Loading model '{args.model}' ({args.dtype}) ...", flush=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)

    n_layers = hf.config.num_hidden_layers
    step = max(1, n_layers // args.num_source_layers)
    source_layers = list(range(2, n_layers, step))
    print(f"model: {n_layers} layers, hidden={hf.config.hidden_size}", flush=True)
    print(f"source_layers: {source_layers}", flush=True)

    prompts = load_prompts(args.prompts)
    print(f"fitting on {len(prompts)} prompts (dim_batch={args.dim_batch}, "
          f"max_seq_len={args.max_seq_len}) ...", flush=True)

    lens = jlens.fit(
        model, prompts,
        source_layers=source_layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=args.checkpoint,
    )
    lens.save(args.out)
    print(f"DONE -> saved lens to {args.out}", flush=True)
    print(f"Now run:  python serve.py --model {args.model} --lens {args.out}", flush=True)


if __name__ == "__main__":
    main()
