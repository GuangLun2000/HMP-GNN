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
    plot_cse_evolution,
    plot_hallucination_metrics_grouped_bar,
    summarize_run_for_fig_a,
    summarize_run_multi_metric,
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
        # V2 M7: CSE is free and always on. PPL requires a decoder-only
        # backbone; DistilBERT (this demo's default) skips PPL gracefully.
        'eval_classification_semantic_entropy': True,
        'eval_perplexity': True,
        'ppl_num_samples': 200,
        'ppl_seed': 42,
        'ppl_max_length': None,
        # Save checkpoint so PPL eval can run.
        'save_global_checkpoint': True,
        'global_checkpoint_subdir': 'global_checkpoint',
        'run_downstream_after_fl': False,
    }


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_exp(label: str, overrides: dict) -> Path:
    cfg = base_config(overrides['experiment_name'])
    cfg.update(overrides)
    # Each experiment writes its checkpoint to its own subdirectory to avoid
    # clobbering earlier runs' checkpoints (and to give PPL eval the right
    # checkpoint to load).
    cfg['global_checkpoint_subdir'] = f"global_checkpoint_{cfg['experiment_name']}"
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

    # Run the 3 experiments and collect per-run result + PPL paths.
    summaries_fig_a = {}     # legacy Fig A (accuracy bar)
    summaries_multi = {}     # V2 M7 multi-metric (acc + CSE + PPL)
    cse_runs = {}            # for Fig F
    for label, overrides in runs:
        exp_name = overrides['experiment_name']
        res_json = run_exp(label, overrides)
        ppl_json = Path('results') / f"{exp_name}_eval_ppl.json"
        if not res_json.is_file():
            print(f"[WARN] Missing result file: {res_json}")
            continue
        summaries_fig_a[label] = summarize_run_for_fig_a(res_json, default_label=label)
        summaries_multi[label] = summarize_run_multi_metric(
            res_json,
            ppl_json_path=ppl_json if ppl_json.is_file() else None,
            default_label=label,
        )
        cse_runs[label] = res_json

    # Fig A: legacy accuracy bar (kept for backward compatibility).
    fig_a_path = Path('results/_v1_demo/figA_defense_bar.png')
    plot_defense_acc_bar(summaries_fig_a, save_path=fig_a_path,
                         metric_key='final_clean_acc',
                         attack_label='Hallucination (label-flip)')

    # Fig C: trust-weight evolution from the HMP-GAE run.
    hmp_json = Path('results/v1demo_hallu_hmpgae_results.json')
    if hmp_json.is_file():
        with open(hmp_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        server_log = data.get('results', [])
        num_clients = data.get('config', {}).get('num_clients', 10)
        num_attackers = data.get('config', {}).get('num_attackers', 2)
        attacker_ids = list(range(num_clients - num_attackers, num_clients))
        plot_trust_weight_evolution(
            server_log, attacker_ids,
            save_path=Path('results/_v1_demo/figC_trust_evolution.png'),
            num_clients=num_clients,
            title_suffix=f'Hallu N={num_clients}',
        )

    # Fig E (V2 M7): three-panel Accuracy / CSE / PPL grouped bar.
    plot_hallucination_metrics_grouped_bar(
        summaries_multi,
        save_path=Path('results/_v1_demo/figE_metrics_grouped.png'),
        attack_label='Hallucination (label-flip)',
    )

    # Fig F (V2 M7): per-round CSE evolution.
    plot_cse_evolution(
        cse_runs,
        save_path=Path('results/_v1_demo/figF_cse_evolution.png'),
        title_suffix=f'N={base_config("")["num_clients"]}',
    )

    print("\n==== Summaries ====")
    for label, s in summaries_multi.items():
        acc = s.get('final_clean_acc', 0.0)
        cse = s.get('mean_cse')
        ppl = s.get('ppl')
        cse_str = f'{cse:.4f}' if cse is not None else 'N/A'
        ppl_str = f'{ppl:.2f}' if ppl is not None else 'N/A (encoder-only)'
        print(f"  {label}:  acc={acc:.4f}  mean_cse={cse_str}  ppl={ppl_str}")


if __name__ == '__main__':
    sys.exit(main() or 0)
