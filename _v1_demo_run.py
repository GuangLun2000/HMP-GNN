"""
V1 demo runner: produce Fig A (attack-vs-defense bar) and Fig C (trust weight
evolution) from 3 tiny FL experiments.

This script is intentionally standalone (not wired into main's CLI). It is
meant for the M5 acceptance check; longer-form experiments for the paper
should use a dedicated batch runner in V2.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch

import main as fl_main
from visualization import (
    plot_trust_weight_evolution,
    plot_defense_acc_bar,
    summarize_run_for_fig_a,
)


def base_config(exp_name: str) -> dict:
    return {
        'experiment_name': exp_name,
        'seed': 42,
        # V1 target regime: N=10, 2 attackers (20%), 3 rounds for demo speed.
        'num_clients': 10,
        'num_attackers': 0,
        'num_rounds': 3,
        'client_lr': 5e-5,
        'server_lr': 1.0,
        'batch_size': 32,
        'test_batch_size': 128,
        'local_epochs': 1,
        'grad_clip_norm': 1.0,
        'alpha': 0.0,
        'dataset': 'ag_news',
        'num_labels': 4,
        'max_length': 64,
        'data_distribution': 'iid',
        'dirichlet_alpha': 0.3,
        'dataset_size_limit': 800,
        'use_lora': True,
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1,
        'lora_target_modules': None,
        'model_name': 'distilbert-base-uncased',
        'attack_method': 'NoAttack',
        'attack_start_round': 0,
        'hallu_flip_ratio': 1.0,
        'hallu_flip_mode': 'pairwise',
        'hallu_flip_map': {0: 1, 1: 0, 2: 3, 3: 2},
        'hallu_target_class': None,
        'hallu_attack_start_round': 0,
        'defense_method': 'fedavg',
        'defense_config': {
            'proj_dim': 32, 'eta_dim': 32, 'hidden_dim': 32, 'latent_dim': 16,
            'num_hmp_layers': 2, 'knn_k': 3,
            'train_steps_per_round': 5, 'train_lr': 1e-3,
            'lambda_H': 1.0, 'lambda_A': 1.0, 'lambda_hist': 0.5,
            'graph_weight': 1.0,
            'residual_weight_alpha': 0.3, 'hist_weight_beta': 0.0,
            'softmax_tau': 0.1, 'hist_ema_beta': 0.9,
            'device': 'cpu', 'random_proj_seed': 42,
            'cold_start_fallback': True,
            'trust_mode': 'reject_then_fedavg',
            'reject_z_threshold': 1.0,
            'keep_min': 1,
        },
        'server_similarity_mode': 'pairwise',
        'use_lagrangian_dual': False,
        'save_global_checkpoint': False,
        'run_downstream_after_fl': False,
    }


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_exp(label: str, overrides: dict) -> Path:
    cfg = base_config(overrides['experiment_name'])
    cfg.update(overrides)
    print(f"\n\n========== DEMO RUN: {label} ==========\n")
    fl_main.main(config_overrides=cfg)
    _cleanup()
    return Path('results') / f"{cfg['experiment_name']}_results.json"


def main():
    Path('results/_v1_demo').mkdir(parents=True, exist_ok=True)

    runs = [
        ('No Attack', {
            'experiment_name': 'v1demo_noattack',
            'num_attackers': 0,
            'attack_method': 'NoAttack',
            'defense_method': 'fedavg',
        }),
        ('Hallu + FedAvg', {
            'experiment_name': 'v1demo_hallu_fedavg',
            'num_attackers': 2,
            'attack_method': 'Hallucination',
            'defense_method': 'fedavg',
        }),
        ('Hallu + HMP-GAE', {
            'experiment_name': 'v1demo_hallu_hmpgae',
            'num_attackers': 2,
            'attack_method': 'Hallucination',
            'defense_method': 'hmp_gae',
        }),
    ]

    summaries = {}
    for label, overrides in runs:
        res_json = run_exp(label, overrides)
        if not res_json.is_file():
            print(f"[WARN] Missing result file: {res_json}")
            continue
        summaries[label] = summarize_run_for_fig_a(res_json, default_label=label)

    fig_a_path = Path('results/_v1_demo/figA_defense_bar.png')
    plot_defense_acc_bar(summaries, save_path=fig_a_path,
                          metric_key='final_clean_acc',
                          attack_label='Hallucination (label-flip)')

    # Fig C from HMP-GAE run
    hmp_json = Path('results/v1demo_hallu_hmpgae_results.json')
    if hmp_json.is_file():
        with open(hmp_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        server_log = data.get('results', [])
        # With num_clients=10 and num_attackers=2, attackers are the last 2.
        num_clients = data.get('config', {}).get('num_clients', 10)
        num_attackers = data.get('config', {}).get('num_attackers', 2)
        attacker_ids = list(range(num_clients - num_attackers, num_clients))
        plot_trust_weight_evolution(
            server_log, attacker_ids,
            save_path=Path('results/_v1_demo/figC_trust_evolution.png'),
            num_clients=num_clients,
            title_suffix=f'Hallu N={num_clients}',
        )

    print("\n==== Summaries ====")
    for label, s in summaries.items():
        print(f"  {label}: final_acc={s['final_clean_acc']:.4f}, best={s['best_clean_acc']:.4f}")


if __name__ == '__main__':
    sys.exit(main() or 0)
