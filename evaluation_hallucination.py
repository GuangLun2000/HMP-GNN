# evaluation_hallucination.py
# V2 M7 evaluation metric: Perplexity on an AG News (or other dataset) test subset.
#
# Design (no text generation, no probe file):
#   1. Load the FL-trained SeqCLS NewsClassifierModel from a checkpoint directory
#      (same layout produced by fed_checkpoint.save_global_model_checkpoint).
#   2. Build a fresh AutoModelForCausalLM from the same base model name and
#      transfer the fine-tuned backbone via decoder_adapters.resolve_adapter.
#      Only decoder-style backbones are supported (Qwen2 / Pythia / LLaMA-like).
#      Encoder-only models (DistilBERT / BERT) do not have a causal LM head
#      and are skipped gracefully with a warning.
#   3. Rebuild the dataset's test tokenizer + text through data_loader.DataManager.
#   4. Stratify-sample `n_samples` balanced across classes (deterministic seed).
#   5. For each sample compute the shifted-label mean NLL loss from the CausalLM
#      (HuggingFace autoregressive models return this directly as `outputs.loss`
#      when `labels=input_ids` is passed).
#   6. Aggregate PPL = exp(mean NLL) and per-class PPL, write a JSON report.
#
# CLI:
#   python evaluation_hallucination.py \
#       --checkpoint results/global_checkpoint \
#       --n-samples 200 \
#       --output results/eval_ppl.json

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Checkpoint loading (thin wrapper around run_downstream_generation helpers)  #
# --------------------------------------------------------------------------- #

def _resolve_checkpoint_paths(checkpoint: Path):
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        return checkpoint / "global_model.pt", checkpoint / "checkpoint_metadata.json"
    return checkpoint, checkpoint.parent / "checkpoint_metadata.json"


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_seqcls_and_meta(checkpoint: Path):
    from models import NewsClassifierModel

    pt_path, meta_path = _resolve_checkpoint_paths(checkpoint)
    if not pt_path.is_file():
        raise FileNotFoundError(f"Checkpoint tensor file not found: {pt_path}")
    pack = _torch_load(pt_path)
    meta = pack.get("metadata")
    if meta is None and meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    if not meta:
        raise ValueError(
            "Missing checkpoint metadata (expected 'metadata' in .pt or checkpoint_metadata.json)"
        )

    kw: Dict[str, Any] = {
        "model_name": meta["model_name"],
        "num_labels": int(meta["num_labels"]),
        "use_lora": bool(meta.get("use_lora", False)),
    }
    if kw["use_lora"]:
        kw["lora_r"] = meta.get("lora_r", 16)
        kw["lora_alpha"] = meta.get("lora_alpha", 32)
        kw["lora_dropout"] = meta.get("lora_dropout", 0.1)
        tm = meta.get("lora_target_modules")
        kw["lora_target_modules"] = None if tm is None else list(tm)

    model = NewsClassifierModel(**kw)
    incompatible = model.load_state_dict(pack["state_dict"], strict=False)
    if incompatible.missing_keys:
        print(f"  [PPL] missing_keys on load: {len(incompatible.missing_keys)} keys")
    if incompatible.unexpected_keys:
        print(f"  [PPL] unexpected_keys on load: {len(incompatible.unexpected_keys)} keys")
    model.eval()
    return model, meta


# --------------------------------------------------------------------------- #
# Test-set subset construction                                                #
# --------------------------------------------------------------------------- #

def _build_test_subset(
    dataset: str,
    num_labels: int,
    model_name: str,
    max_length: int,
    n_samples: int,
    seed: int,
    dataset_size_limit: Optional[int] = None,
):
    """
    Rebuild the test set via DataManager, then stratify-sample `n_samples`
    uniformly across classes. Returns a tokenizer and a list of
    (text, label_id) tuples of length ~n_samples.
    """
    from data_loader import DataManager

    dm = DataManager(
        num_clients=1,
        num_attackers=0,
        test_seed=seed,
        dataset_size_limit=dataset_size_limit,
        batch_size=1,
        test_batch_size=1,
        model_name=model_name,
        max_length=max_length,
        dataset=dataset,
    )

    texts = list(dm.test_texts)
    labels = list(dm.test_labels)

    # Group indices by class so we can take balanced stratified samples.
    per_class: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        per_class[int(y)].append(i)

    n_classes = num_labels if num_labels > 0 else len(per_class)
    per_class_target = max(1, n_samples // max(1, n_classes))

    rng = np.random.default_rng(seed)
    picked: List[int] = []
    for c in sorted(per_class.keys()):
        pool = per_class[c]
        if len(pool) == 0:
            continue
        k = min(per_class_target, len(pool))
        picked.extend(rng.choice(pool, size=k, replace=False).tolist())

    # In case requested n_samples was not evenly divisible, top-up randomly
    # from the remaining pool without replacement.
    if len(picked) < n_samples:
        remaining = [i for i in range(len(texts)) if i not in set(picked)]
        extra = min(n_samples - len(picked), len(remaining))
        if extra > 0:
            picked.extend(rng.choice(remaining, size=extra, replace=False).tolist())

    # Shuffle so downstream per-class accounting still has variety.
    rng.shuffle(picked)
    subset = [(texts[i], int(labels[i])) for i in picked]
    return dm.tokenizer, subset


# --------------------------------------------------------------------------- #
# PPL computation                                                             #
# --------------------------------------------------------------------------- #

def _is_decoder_backbone(model_name: str) -> bool:
    """Rough heuristic: decoder-only architectures the adapters support."""
    m = model_name.lower()
    decoder_tokens = [
        "gpt", "pythia", "opt-", "/opt", "llama", "mistral", "qwen", "bloom", "falcon",
    ]
    return any(t in m for t in decoder_tokens)


def compute_test_ppl(
    checkpoint_dir,
    n_samples: int = 200,
    seed: int = 42,
    device: Optional[torch.device] = None,
    max_length: Optional[int] = None,
    dataset_override: Optional[str] = None,
    num_labels_override: Optional[int] = None,
    dataset_size_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Transfer the FL-trained SeqCLS backbone into a CausalLM and compute mean
    perplexity on a stratified test subset.

    Returns a dict with keys:
        ppl_mean, ppl_per_class, n_samples, seed, model_name, dataset,
        skipped (bool), skip_reason (str | None)
    """
    from decoder_adapters import resolve_adapter

    checkpoint_dir = Path(checkpoint_dir)
    device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    print(f"[PPL] Loading SeqCLS checkpoint from: {checkpoint_dir}")
    seq_model, meta = _load_seqcls_and_meta(checkpoint_dir)
    model_name: str = meta["model_name"]
    num_labels: int = int(num_labels_override or meta["num_labels"])
    # Guess the dataset the model was fine-tuned on. The FL checkpoint metadata
    # does not store this directly, so fall back to AG News unless overridden.
    dataset: str = (dataset_override or meta.get("dataset") or "ag_news").lower()

    # Gate on decoder-style backbones -- encoder-only models have no LM head
    # to measure PPL against. Skip gracefully.
    if not _is_decoder_backbone(model_name):
        return {
            "skipped": True,
            "skip_reason": (
                f"model_name={model_name!r} is an encoder-only backbone; "
                "no AutoModelForCausalLM available for PPL evaluation."
            ),
            "model_name": model_name,
            "dataset": dataset,
            "n_samples": 0,
            "seed": seed,
            "ppl_mean": None,
            "ppl_per_class": {},
        }

    # Effective max_length matches the FL training setting by default.
    eff_max_length = int(
        max_length if max_length is not None else int(meta.get("max_length", 128))
    )

    # Build AutoModelForCausalLM and transfer fine-tuned backbone in.
    print(f"[PPL] Building AutoModelForCausalLM(base={model_name}) and transferring backbone...")
    causal_lm = AutoModelForCausalLM.from_pretrained(model_name)
    causal_lm.to(device)
    causal_lm.eval()

    seq_inner = seq_model.model
    # If LoRA is used the inner is a PEFT PeftModel; merge-and-unload gives
    # a standard HF model whose state_dict keys align with the CausalLM.
    try:
        adapter = resolve_adapter(model_name)
        adapter.transfer_backbone(seq_inner, causal_lm)
    except Exception as e:
        # If for some reason the transfer fails (e.g. new architecture without
        # an adapter), return a skip result rather than crashing the FL run.
        return {
            "skipped": True,
            "skip_reason": f"backbone transfer failed: {type(e).__name__}: {e}",
            "model_name": model_name,
            "dataset": dataset,
            "n_samples": 0,
            "seed": seed,
            "ppl_mean": None,
            "ppl_per_class": {},
        }

    # Build tokenizer + stratified test subset.
    tokenizer, subset = _build_test_subset(
        dataset=dataset,
        num_labels=num_labels,
        model_name=model_name,
        max_length=eff_max_length,
        n_samples=n_samples,
        seed=seed,
        dataset_size_limit=dataset_size_limit,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Compute shifted-label NLL for each sample, track per-class stats.
    total_nll = 0.0
    total_samples = 0
    per_class_nll: Dict[int, float] = defaultdict(float)
    per_class_count: Dict[int, int] = defaultdict(int)

    print(f"[PPL] Computing per-sample NLL on {len(subset)} test texts (device={device})")
    with torch.no_grad():
        for text, y in subset:
            enc = tokenizer(
                str(text),
                truncation=True,
                max_length=eff_max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            # Skip trivially short/empty tokenizations.
            if input_ids.shape[1] < 2:
                continue
            labels = input_ids.clone()
            out = causal_lm(input_ids=input_ids, labels=labels)
            nll = float(out.loss.detach().cpu().item())
            if not math.isfinite(nll):
                continue
            total_nll += nll
            total_samples += 1
            per_class_nll[int(y)] += nll
            per_class_count[int(y)] += 1

    if total_samples == 0:
        return {
            "skipped": True,
            "skip_reason": "no valid samples produced a finite NLL",
            "model_name": model_name,
            "dataset": dataset,
            "n_samples": 0,
            "seed": seed,
            "ppl_mean": None,
            "ppl_per_class": {},
        }

    mean_nll = total_nll / total_samples
    ppl_mean = math.exp(mean_nll)
    ppl_per_class = {
        str(c): math.exp(per_class_nll[c] / per_class_count[c])
        for c in per_class_nll
        if per_class_count[c] > 0
    }

    result = {
        "skipped": False,
        "skip_reason": None,
        "model_name": model_name,
        "dataset": dataset,
        "n_samples": total_samples,
        "seed": seed,
        "max_length": eff_max_length,
        "mean_nll": mean_nll,
        "ppl_mean": ppl_mean,
        "ppl_per_class": ppl_per_class,
    }
    print(f"[PPL] Done. mean_nll={mean_nll:.4f}, ppl_mean={ppl_mean:.4f} on {total_samples} samples")
    # Free GPU memory eagerly.
    del causal_lm, seq_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute test-set PPL for a FL checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to checkpoint directory (contains global_model.pt + checkpoint_metadata.json).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON path. Default: <checkpoint>/eval_ppl.json")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None,
                   help="Override dataset (ag_news | imdb | dbpedia | yahoo_answers).")
    p.add_argument("--num-labels", type=int, default=None)
    p.add_argument("--dataset-size-limit", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    device = None
    if args.device:
        device = torch.device(args.device)
    result = compute_test_ppl(
        checkpoint_dir=args.checkpoint,
        n_samples=args.n_samples,
        seed=args.seed,
        device=device,
        max_length=args.max_length,
        dataset_override=args.dataset,
        num_labels_override=args.num_labels,
        dataset_size_limit=args.dataset_size_limit,
    )
    out = args.output or (args.checkpoint / "eval_ppl.json")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[PPL] Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
