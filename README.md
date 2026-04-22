# HMP-GNN

- Hallucination Immunization for Multimodal Federated LLMs via Hypergraph Message Passing.
- [Hanlin Cai](https://caihanlin.com/)

## File Structure

```
.
├── .gitignore
├── LICENSE
├── README.md                          # This documentation
├── requirements.txt                   # Python dependencies
├── main.py                            # Entry: configure and run federated learning
├── client.py                          # BenignClient, AttackerClient (GRMP), baselines hook
├── server.py                          # Aggregation, evaluation, round orchestration
├── models.py                          # NewsClassifierModel, VGAE, etc.
├── data_loader.py                     # DataManager / datasets (AG News, Yahoo Answers, IMDB, DBpedia)
├── fed_checkpoint.py                  # Save global model + metadata after FL
├── decoder_adapters.py                # SeqCLS backbone → CausalLM transfer adapters
├── run_downstream_generation.py       # CLI: checkpoint + probes → JSONL (Task 2)
├── visualization.py                   # Experiment figures / plots
├── attack_baseline_alie.py            # ALIE baseline (NeurIPS ’19)
├── attack_baseline_gaussian.py        # Gaussian baseline (USENIX Security ’20)
├── attack_baseline_sign_flipping.py   # Sign-flipping baseline (ICML ’18)
├── attack_baseline_hallucination.py   # Hallucination attack via label-flipping (V1)
├── defense.py                         # Pluggable defense strategies (V1: FedAvg / HMP-GAE)
├── hmp_gae/                           # HMP-GAE defense sub-package (this paper)
│   ├── node_features.py               #   eta_i = f_enc(Delta_i, stats, history)
│   ├── hypergraph.py                  #   k-NN hypergraph H, D_V, D_E
│   ├── encoder.py                     #   L-layer HMP encoder (node↔hyperedge)
│   ├── decoder.py                     #   GAE decoder: A_hat, H_hat
│   ├── losses.py                      #   BCE(H,H_hat) + smoothness + hist
│   ├── trust_scorer.py                #   closed-form trust -> alpha_i
│   └── runtime.py                     #   end-to-end HMPGAERuntime
├── GRMP_Attack_Colab.ipynb            # Colab-oriented notebook
└── data/                              # Datasets for Task 1 and Task 2
```

## Supported Models

- Encoder-only (BERT-style): `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`
- Decoder-only (GPT-style): `gpt2`, `EleutherAI/pythia-160m`, `EleutherAI/pythia-1b`, `facebook/opt-125m`, `Qwen/Qwen2.5-0.5B`
- Configure in `main.py` via `model_name`.

## Supported Datasets

- **AG News**: `dataset='ag_news'`, `num_labels=4`, `max_length=128` (default)
- **Yahoo Answers** (yassiracharki/Yahoo_Answers_10_categories_for_NLP): `dataset='yahoo_answers'`, `num_labels=10`, `max_length=256` (10 topic classes, 1.4M train / 60K test)
- **IMDB** (stanfordnlp/imdb): `dataset='imdb'`, `num_labels=2`, `max_length=512` (or 256 for lower memory)
- **DBpedia 14** (fancyzhx/dbpedia_14): `dataset='dbpedia'`, `num_labels=14`, `max_length=512` (14 topic classes, 560K train / 70K test)
- Configure in `main.py` via `dataset`, `num_labels`, and `max_length`.

<br>

## Install Dependencies

```python
!pip install -r requirements.txt
```

## Run the Code

### Local Execution

```bash
python main.py
```

### Google Colab Execution (or other Cloud AI platforms)

**Option 1: Simple Version (Recommended for quick runs)**

```python
# Cell 1: Install dependencies
!git clone https://github.com/GuangLun2000/HMP-GNN.git
!pip install -r ./HMP-GNN/requirements.txt

# Cell 2: Run experiment

!cd ./HMP-GNN && python main.py
```

**Option 2: Interactive Notebook (Recommended for configuration changes)**

1. Open `GRMP_Attack_Colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells: Runtime → Run all

<br>

---

### Checkpoints and Task 2 (downstream generation)

In [`main.py`](main.py) → `config`, turn on **`save_global_checkpoint`** and optionally **`global_checkpoint_subdir`** (under `results/`). You get `global_model.pt`, `checkpoint_metadata.json`, and with LoRA a **`peft_adapter/`** folder. Train with a causal **`model_name`** that matches **`num_labels`** / **`dataset`** (e.g. AG News + Pythia or Qwen2.5 as in **Supported Models**).

**Task 2** classifies each probe with the saved SeqCLS head, copies the backbone into **`AutoModelForCausalLM`** (no LM fine-tuning), and decodes a short explanation. AG News labels: 0–3 → World, Sports, Business, Sci/Tech. Backbone wiring lives in [`decoder_adapters.py`](decoder_adapters.py). Default probes: [`data/ag_news_business_30.json`](data/ag_news_business_30.json).

To chain after FL, set **`run_downstream_after_fl`**: `True` (plus `downstream_probes`, `downstream_output`, `downstream_cli_args`, …). Or run the CLI:

```bash
python run_downstream_generation.py \
  --checkpoint results/global_checkpoint \
  --probes data/ag_news_business_30.json \
  --output results/downstream_gen.jsonl \
  --stable
```

`--stable` is a conservative greedy preset; use **`--help`** for decoding flags. Each output line is JSONL (labels + text); compare predictions to ground-truth categories and read the rationale fields to study poisoning.

**Other decoder families:** implement `DecoderAdapter` (`matches`, `transfer_backbone`), append to **`ADAPTER_REGISTRY`** in [`decoder_adapters.py`](decoder_adapters.py), then point Task 2 at checkpoints with the same **`model_name`**.

<br>

---

## HMP-GAE Immunization (V1)

V1 ships the paper's core immunization pipeline end-to-end:

- **Attack**: `HallucinationAttackerClient` — the client trains on (partially) label-flipped data. No nested optimization loop, same wall-clock as benign clients.
- **Defense**: `HMPGAEDefense` — server-side hypergraph message-passing graph autoencoder that self-supervises on each round's updates, outputs per-client trust weights, and aggregates accordingly.

### Configure via `main.py::main()`

```python
# Attack
'attack_method': 'Hallucination',
'hallu_flip_ratio': 1.0,               # 0..1, fraction of samples flipped
'hallu_flip_mode': 'pairwise',         # 'pairwise' | 'targeted' | 'random'
'hallu_flip_map': {0: 1, 1: 0, 2: 3, 3: 2},   # AG News: World<->Sports, Business<->Sci/Tech

# Defense
'defense_method': 'hmp_gae',           # or 'fedavg' for the baseline
'defense_config': {
    'knn_k': 3, 'hidden_dim': 64, 'latent_dim': 32, 'num_hmp_layers': 2,
    'train_steps_per_round': 5, 'train_lr': 1e-3,
    'lambda_H': 1.0, 'lambda_A': 1.0, 'lambda_hist': 0.5,
    'graph_weight': 1.0, 'residual_weight_alpha': 0.3, 'hist_weight_beta': 0.0,
    'trust_mode': 'reject_then_fedavg', 'reject_z_threshold': 0.75,
    'softmax_tau': 0.1, 'hist_ema_beta': 0.9,
    'cold_start_fallback': True,
    'device': 'cpu', 'random_proj_seed': 42,
},
```

### V1 demo: end-to-end verification

```bash
# Single demo run: NoAttack / Hallu+FedAvg / Hallu+HMP-GAE, produces Fig A and Fig C
python _v1_demo_run.py

# 3-seed sweep (M6 validation): mean +/- std of final clean accuracy
python _v1_seed_runs.py
```

Representative V1 demo numbers (N=10 clients, 2 attackers, 3 rounds, AG News subset, DistilBERT + LoRA):

| Setting | Final Clean Acc (3-seed mean ± std) |
|---|---|
| Hallu + FedAvg   | 0.5667 ± 0.0661 |
| Hallu + HMP-GAE  | 0.6361 ± 0.0474 |
| **Delta (HMP-GAE improvement)** | **+0.0694** |

The trust-weight evolution PDF (`figC_*.pdf`) shows the two attackers (red) collapsing to `alpha_i ≈ 0` from round 2 onward, while the 8 benign clients share the remaining mass close to uniform `1/8 = 0.125`.

### V1 limitations / V2 roadmap

- V1 omits comparison baselines (Krum / Median / FLTrust / FLDetector / Safe-FedLLM). They are planned for V2.
- Evaluation currently reports classification accuracy only. Semantic entropy, perplexity, and ASR on Task 2 generation are planned for V2.
- Single modality (text) -- the paper's multimodal formulation is simulated via LoRA-only updates; true multimodal encoders are V2 work.
- Tuning presets above are calibrated for the N=10 / 2-attackers / AG News regime. For N<=4 the defense auto-falls back to FedAvg; for very heterogeneous (Dirichlet alpha << 0.3) data, `reject_z_threshold` may need to be raised.

<br>
