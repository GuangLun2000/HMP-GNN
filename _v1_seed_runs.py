"""
V1 M6 / V2 M7 validation: 3 seeds x {FedAvg, HMP-GAE} under Hallucination attack.

Collects three metrics per run:
    - Final clean accuracy    (higher is better)
    - Mean Classification Semantic Entropy over rounds (lower is better, V2 M7)
    - End-of-FL Perplexity on stratified test subset   (lower is better, V2 M7)

Reports mean +/- std across seeds, writes:
    - results/_v1_seed_runs/seed_summary.json  (structured numbers)
    - results/_v1_seed_runs/table_I.csv        (paper-ready Table I)
    - results/_v1_seed_runs/figA_seed_mean.pdf (accuracy bar, seed-averaged)
    - results/_v1_seed_runs/figC_trust_seed{s}.pdf (trust evolution per seed)
    - results/_v1_seed_runs/figE_metrics_grouped.pdf (acc / CSE / PPL grouped bar)
    - results/_v1_seed_runs/figF_cse_evolution.pdf (CSE curve, first seed for clarity)
"""

from __future__ import annotations

import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

import main as fl_main
from _v1_demo_run import base_config
from visualization import (
    plot_trust_weight_evolution,
    plot_defense_acc_bar,
    plot_cse_evolution,
    plot_hallucination_metrics_grouped_bar,
    summarize_run_multi_metric,
)


SEEDS = [42, 2024, 7777]


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
    return mean, math.sqrt(var)


def run_one(seed: int, defense: str) -> Tuple[Path, Path]:
    """Returns (results_json_path, ppl_json_path)."""
    exp_name = f"v1seed_{defense}_s{seed}"
    cfg = base_config(exp_name)
    cfg.update({
        'seed': seed,
        'num_attackers': 2,
        'attack_method': 'Hallucination',
        'defense_method': defense,
        'num_rounds': 3,
        # Per-run checkpoint dir to avoid clobbering.
        'global_checkpoint_subdir': f'global_checkpoint_{exp_name}',
    })
    print(f"\n===== RUN seed={seed} defense={defense} =====")
    fl_main.main(config_overrides=cfg)
    _cleanup()
    results_dir = Path('results')
    return (
        results_dir / f"{exp_name}_results.json",
        results_dir / f"{exp_name}_eval_ppl.json",
    )


def _write_table_i_csv(
    csv_path: Path,
    per_defense: Dict[str, Dict[str, List[float]]],
    include_ppl: bool,
) -> None:
    """
    per_defense[label] = {'acc': [..], 'cse': [..], 'ppl': [..]}
    Writes a paper-ready Table I with mean ± std for each metric.
    """
    headers = ['Defense', 'Accuracy (mean)', 'Accuracy (std)',
               'CSE (mean)', 'CSE (std)']
    if include_ppl:
        headers += ['PPL (mean)', 'PPL (std)']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for label, m in per_defense.items():
            acc_m, acc_s = _mean_std(m.get('acc', []))
            cse_m, cse_s = _mean_std(m.get('cse', []))
            row = [label, f'{acc_m:.4f}', f'{acc_s:.4f}',
                   f'{cse_m:.4f}', f'{cse_s:.4f}']
            if include_ppl:
                ppl_m, ppl_s = _mean_std(m.get('ppl', []))
                row += [f'{ppl_m:.4f}' if m.get('ppl') else 'N/A',
                        f'{ppl_s:.4f}' if m.get('ppl') else 'N/A']
            writer.writerow(row)
    print(f"  Table I CSV -> {csv_path}")


def main() -> int:
    out_dir = Path('results/_v1_seed_runs')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-defense metric accumulators.
    per_defense: Dict[str, Dict[str, List[float]]] = {
        'Hallu + FedAvg':  {'acc': [], 'cse': [], 'ppl': []},
        'Hallu + HMP-GAE': {'acc': [], 'cse': [], 'ppl': []},
    }
    per_seed_summaries: Dict[int, Dict[str, Dict]] = {}

    for seed in SEEDS:
        per_seed_summaries[seed] = {}
        for defense, label in [('fedavg', 'Hallu + FedAvg'),
                                ('hmp_gae', 'Hallu + HMP-GAE')]:
            res_json, ppl_json = run_one(seed, defense)
            if not res_json.is_file():
                print(f"[WARN] missing results json: {res_json}")
                continue
            s = summarize_run_multi_metric(
                res_json,
                ppl_json_path=ppl_json if ppl_json.is_file() else None,
                default_label=label,
            )
            per_defense[label]['acc'].append(float(s.get('final_clean_acc', 0.0)))
            if s.get('mean_cse') is not None:
                per_defense[label]['cse'].append(float(s['mean_cse']))
            if s.get('ppl') is not None:
                per_defense[label]['ppl'].append(float(s['ppl']))
            per_seed_summaries[seed][label] = s

    # Summaries and JSON dump.
    def _collect_stats(m: Dict[str, List[float]]) -> Dict[str, float]:
        acc_m, acc_s = _mean_std(m.get('acc', []))
        cse_m, cse_s = _mean_std(m.get('cse', []))
        ppl_m, ppl_s = _mean_std(m.get('ppl', []))
        return {
            'acc_mean': acc_m, 'acc_std': acc_s,
            'cse_mean': cse_m, 'cse_std': cse_s,
            'ppl_mean': ppl_m if m.get('ppl') else None,
            'ppl_std': ppl_s if m.get('ppl') else None,
        }

    fa_stats = _collect_stats(per_defense['Hallu + FedAvg'])
    hmp_stats = _collect_stats(per_defense['Hallu + HMP-GAE'])
    include_ppl = bool(per_defense['Hallu + FedAvg']['ppl']) and bool(per_defense['Hallu + HMP-GAE']['ppl'])

    print("\n\n========== V1 M6 / V2 M7 seed-sweep summary ==========")
    print(f"Seeds: {SEEDS}")
    for label, m in per_defense.items():
        st = _collect_stats(m)
        line = (
            f"{label:20s}: acc={st['acc_mean']:.4f} ± {st['acc_std']:.4f}, "
            f"cse={st['cse_mean']:.4f} ± {st['cse_std']:.4f}"
        )
        if st['ppl_mean'] is not None:
            line += f", ppl={st['ppl_mean']:.4f} ± {st['ppl_std']:.4f}"
        else:
            line += ", ppl=N/A (encoder-only or skipped)"
        print(line)

    delta_acc = hmp_stats['acc_mean'] - fa_stats['acc_mean']
    print(f"HMP-GAE accuracy delta: {delta_acc:+.4f}  (target >= +0.03)")

    summary_json = out_dir / 'seed_summary.json'
    summary_json.write_text(json.dumps({
        'seeds': SEEDS,
        'per_defense_raw': per_defense,
        'fedavg': fa_stats,
        'hmpgae': hmp_stats,
        'delta_accuracy': delta_acc,
    }, indent=2))
    print(f"\nWrote: {summary_json}")

    # Table I CSV (paper-ready).
    _write_table_i_csv(out_dir / 'table_I.csv', per_defense, include_ppl=include_ppl)

    # Fig A: accuracy bar with error bars (seed-averaged).
    plot_defense_acc_bar({
        'Hallu + FedAvg':  {'final_clean_acc': fa_stats['acc_mean'],  'acc_std': fa_stats['acc_std']},
        'Hallu + HMP-GAE': {'final_clean_acc': hmp_stats['acc_mean'], 'acc_std': hmp_stats['acc_std']},
    }, save_path=out_dir / 'figA_seed_mean.png',
       metric_key='final_clean_acc',
       attack_label='Hallucination (3-seed mean)')

    # Fig E: 3-metric grouped bar with error bars (Accuracy / CSE / PPL).
    fa_mean_summary = {
        'final_clean_acc': fa_stats['acc_mean'],
        'mean_cse':        fa_stats['cse_mean'],
        'ppl':             fa_stats['ppl_mean'],
        'acc_std':         fa_stats['acc_std'],
        'cse_std':         fa_stats['cse_std'],
        'ppl_std':         fa_stats['ppl_std'] if fa_stats['ppl_std'] else 0.0,
    }
    hmp_mean_summary = {
        'final_clean_acc': hmp_stats['acc_mean'],
        'mean_cse':        hmp_stats['cse_mean'],
        'ppl':             hmp_stats['ppl_mean'],
        'acc_std':         hmp_stats['acc_std'],
        'cse_std':         hmp_stats['cse_std'],
        'ppl_std':         hmp_stats['ppl_std'] if hmp_stats['ppl_std'] else 0.0,
    }
    plot_hallucination_metrics_grouped_bar(
        {'Hallu + FedAvg': fa_mean_summary, 'Hallu + HMP-GAE': hmp_mean_summary},
        save_path=out_dir / 'figE_metrics_grouped.png',
        attack_label='Hallucination (3-seed mean)',
    )

    # Fig C (per-seed) and Fig F (first seed's CSE curves).
    first_seed_cse_runs: Dict[str, Path] = {}
    for seed in SEEDS:
        for defense, label in [('fedavg', 'Hallu + FedAvg'),
                                ('hmp_gae', 'Hallu + HMP-GAE')]:
            res = Path('results') / f"v1seed_{defense}_s{seed}_results.json"
            if not res.is_file():
                continue
            if seed == SEEDS[0]:
                first_seed_cse_runs[label] = res
            if defense == 'hmp_gae':
                data = json.loads(res.read_text())
                server_log = data.get('results', [])
                num_clients = data.get('config', {}).get('num_clients', 10)
                num_attackers = data.get('config', {}).get('num_attackers', 2)
                attacker_ids = list(range(num_clients - num_attackers, num_clients))
                plot_trust_weight_evolution(
                    server_log, attacker_ids,
                    save_path=out_dir / f'figC_trust_seed{seed}.png',
                    num_clients=num_clients,
                    title_suffix=f'seed={seed}',
                )

    if first_seed_cse_runs:
        plot_cse_evolution(
            first_seed_cse_runs,
            save_path=out_dir / 'figF_cse_evolution.png',
            title_suffix=f'seed={SEEDS[0]}',
        )

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
