"""
V1 M6 validation: 3 seeds x {FedAvg, HMP-GAE} under Hallucination attack.

Reports mean +/- std of final clean accuracy and computes the HMP-GAE
improvement over FedAvg.
"""

from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

import torch

import main as fl_main
from _v1_demo_run import base_config  # reuse the V1 config skeleton
from visualization import (
    plot_trust_weight_evolution,
    plot_defense_acc_bar,
    summarize_run_for_fig_a,
)


SEEDS = [42, 2024, 7777]


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one(seed: int, defense: str) -> Path:
    cfg = base_config(f"v1seed_{defense}_s{seed}")
    cfg.update({
        'seed': seed,
        'num_attackers': 2,
        'attack_method': 'Hallucination',
        'defense_method': defense,
        'num_rounds': 3,
    })
    print(f"\n===== RUN seed={seed} defense={defense} =====")
    fl_main.main(config_overrides=cfg)
    _cleanup()
    return Path('results') / f"{cfg['experiment_name']}_results.json"


def summarize(accs):
    if not accs:
        return 0.0, 0.0
    mean = sum(accs) / len(accs)
    var = sum((x - mean) ** 2 for x in accs) / max(1, len(accs) - 1)
    return mean, math.sqrt(var)


def main():
    out_dir = Path('results/_v1_seed_runs')
    out_dir.mkdir(parents=True, exist_ok=True)

    fedavg_accs = []
    hmp_accs = []

    for seed in SEEDS:
        p_fa = run_one(seed, 'fedavg')
        p_hmp = run_one(seed, 'hmp_gae')
        if p_fa.is_file():
            fedavg_accs.append(summarize_run_for_fig_a(p_fa)['final_clean_acc'])
        if p_hmp.is_file():
            hmp_accs.append(summarize_run_for_fig_a(p_hmp)['final_clean_acc'])

    fa_mean, fa_std = summarize(fedavg_accs)
    hmp_mean, hmp_std = summarize(hmp_accs)
    delta = hmp_mean - fa_mean

    print("\n\n========== V1 M6 seed-sweep summary ==========")
    print(f"Seeds: {SEEDS}")
    print(f"Hallu + FedAvg   : mean={fa_mean:.4f}, std={fa_std:.4f}, vals={['%.4f' % a for a in fedavg_accs]}")
    print(f"Hallu + HMP-GAE  : mean={hmp_mean:.4f}, std={hmp_std:.4f}, vals={['%.4f' % a for a in hmp_accs]}")
    print(f"HMP-GAE delta    : {delta:+.4f}  (target >= +0.03)")
    print(f"HMP-GAE std      : {hmp_std:.4f}  (target < 0.02)")

    summary_json = out_dir / 'seed_summary.json'
    summary_json.write_text(json.dumps({
        'seeds': SEEDS,
        'fedavg_accs': fedavg_accs, 'fedavg_mean': fa_mean, 'fedavg_std': fa_std,
        'hmpgae_accs': hmp_accs, 'hmpgae_mean': hmp_mean, 'hmpgae_std': hmp_std,
        'delta': delta,
    }, indent=2))
    print(f"\nWrote: {summary_json}")

    # Fig A with error bars.
    plot_defense_acc_bar({
        'Hallu + FedAvg':  {'final_clean_acc': fa_mean,  'acc_std': fa_std},
        'Hallu + HMP-GAE': {'final_clean_acc': hmp_mean, 'acc_std': hmp_std},
    }, save_path=out_dir / 'figA_seed_mean.png',
       metric_key='final_clean_acc',
       attack_label='Hallucination (3-seed mean)')

    # Fig C from the first HMP-GAE run that has data.
    for seed in SEEDS:
        p_hmp = Path('results') / f"v1seed_hmp_gae_s{seed}_results.json"
        if p_hmp.is_file():
            data = json.loads(p_hmp.read_text())
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


if __name__ == '__main__':
    sys.exit(main() or 0)
