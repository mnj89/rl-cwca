# RL-GACA: Reinforcement Learning with Gravity-Aware Cache Allocation for D2D Network Offloading

[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Nabeel Ali, Asif Kabir, Tamer Mekkawy, Ashraf Mahran**  
> University of Kotli, AJK, Pakistan · Military Technical College, Cairo, Egypt  
> Submitted to *ICT Express* (Elsevier), Round 3

---

## Overview

RL-GACA is a proactive D2D caching method that assigns each (user, file) pair a **gravity score** — combining file popularity, user interest match, and meeting probability — and uses an **A2C controller** to adjust those scores and diversify cached content across devices. A greedy fill procedure guarantees full cache utilization by construction.

The key insight: existing heuristics push a single global file ranking to every device; existing D2D-DRL methods treat cache decisions as binary (cache or skip). RL-GACA combines domain-structured scoring with a learned multiplier, producing a distinct ranked candidate list per device without binary placement.

### Key Results (London Marylebone, 172 devices)

| Metric | Value |
|--------|-------|
| Offloading at 2 GB | **0.6462** |
| vs CFCA (best heuristic) | +69.6% |
| vs DQN-Interest | +24.3% |
| vs DQN with same gravity scoring | +10.4% |
| Statistical significance (10 seeds) | p = 0.0002, r = 1.000 |
| Aligned filter variant (matched threshold) | **0.6620** (+27.4% vs DQN-Interest) |
| Training time | ~75 s / seed (single CPU core) |
| Deployment time (172 devices) | ~8 s |

### Transfer to T-Drive (Beijing Taxis)

| Cache | Zero-shot | Fine-tune (20 eps) | CFCA |
|-------|-----------|-------------------|------|
| 500 MB | 0.1776 | **0.2118** | 0.1944 |
| 1 GB | 0.2871 | **0.3240** | 0.2806 |
| 2 GB | 0.4329 | **0.4496** | 0.4102 |

Fine-tuning costs ~30 s/seed on a single CPU core.

---

## Repository Structure

```
rlgaca/
├── rlgaca_main.py          # Complete experimental pipeline (Phases 1–12)
├── fetch_tdrive.py         # T-Drive dataset downloader and contact extractor
├── results/
│   ├── sumo_sim/
│   │   └── contact_data.npz        # SUMO Marylebone contact matrices
│   ├── tdrive/
│   │   └── tdrive_contact_data.npz # T-Drive Beijing taxi contact matrices
│   └── runs/
│       └── <YYYYMMDD_HHMMSS>/
│           ├── figures/    # 300 DPI PDF + PNG publication figures
│           ├── tables/     # CSV tables (all results)
│           ├── logs/
│           └── meta.json   # Run configuration and summary
├── tables/                 # Pre-computed results CSVs (see below)
│   ├── table1_main.csv
│   ├── table2_cache_sweep.csv
│   ├── table3_ablation.csv
│   ├── table4_significance.csv
│   ├── table5_controller.csv
│   ├── table6_coverage_fairness.csv
│   ├── table7_contention.csv
│   ├── table8_aligned_comparison.csv
│   └── table9_transfer_tdrive.csv
└── README.md
```

---

## Installation

```bash
git clone https://github.com/[repository]/rlgaca.git
cd rlgaca
pip install numpy scipy torch tqdm pandas matplotlib seaborn
```

**Requirements:** Python 3.8+, PyTorch (CPU is sufficient — no GPU needed).

No CUDA required. The full pipeline runs in under 10 minutes on a standard laptop CPU.

---

## Reproducing Results

### Step 1 — Marylebone SUMO trace (required)

The SUMO contact data is provided as a pre-processed NPZ file. Place it at:

```
results/sumo_sim/contact_data.npz
```

The file contains:
- `avg_ct` — (172, 172) mean contact duration per pair (seconds)
- `n_ct` — (172, 172) number of contact events per pair
- `vtypes_arr` — (172,) device type labels (`'car'`, `'bike'`, `'pedestrian'`)

### Step 2 — Run all phases (Marylebone)

```bash
python rlgaca_main.py
```

This runs Phases 1–12 sequentially and writes all results to a timestamped directory under `results/runs/`. Typical total runtime: **~8 minutes** on a single CPU core.

**What each phase produces:**

| Phase | Output |
|-------|--------|
| 1 | Trained A2C, PPO, DQN agents + heuristic baselines |
| 2 | `table1_main.csv` — main comparison (1 GB, 3 seeds) |
| 3 | `table2_cache_sweep.csv` — offloading at 4 cache sizes + `table8_aligned_comparison.csv` |
| 4 | `table3_ablation.csv` — component ablation |
| 5 | Zipf sensitivity, file size sensitivity figures |
| 6 | `table4_significance.csv` — Mann-Whitney U (10 seeds) |
| 7 | `table7_contention.csv` — collision factor sweep |
| 8 | `table6_coverage_fairness.csv` — Jain index |
| 9 | Scalability figure (library size 100–400) |
| 10 | `table5_controller.csv` — A2C vs PPO vs DQN |
| 11 | All 13 publication figures (300 DPI PDF + PNG) |
| 12 | All LaTeX-ready tables as CSV |

### Step 3 — T-Drive transfer validation (optional, ~3 min)

```bash
# Download and process the T-Drive dataset (~35 MB)
python fetch_tdrive.py --n-taxis 200 --out results/tdrive/tdrive_contact_data.npz

# Run Phase 13 (transfer validation) using checkpoint from Step 2
python rlgaca_main.py --phase13
```

`fetch_tdrive.py` downloads the Microsoft Research T-Drive GPS dataset (no registration required), extracts contact events using the same 120 m radio range and 30% fragment threshold as the Marylebone model, and saves the NPZ.

Phase 13 evaluates in two modes:
- **Zero-shot**: Marylebone-trained weights applied directly to T-Drive topology
- **Fine-tune**: 20 additional A2C episodes on T-Drive (~30 s/seed)

Results are saved to `table9_transfer_tdrive.csv`.

---

## Pre-computed Results

All tables from the paper are provided in `tables/` so results can be verified without re-running the pipeline.

| File | Description |
|------|-------------|
| `table1_main.csv` | Main comparison: offloading and CHR at 1 GB, γ=0.6, 3 seeds |
| `table2_cache_sweep.csv` | Offloading at 500 MB / 1 GB / 1.5 GB / 2 GB |
| `table3_ablation.csv` | Component ablation (6 variants + aligned filter) |
| `table4_significance.csv` | Mann-Whitney U tests, 10 seeds |
| `table5_controller.csv` | A2C vs PPO vs DQN with same gravity base |
| `table6_coverage_fairness.csv` | Request coverage and Jain's fairness index |
| `table7_contention.csv` | Collision factor ρ sweep (MAC-layer contention) |
| `table8_aligned_comparison.csv` | Aligned threshold (I_uc ≥ 0.05) across 4 cache sizes |
| `table9_transfer_tdrive.csv` | T-Drive zero-shot and fine-tune results |

---

## Method Summary

### Gravity Score

Each (user *u*, file *f*) pair receives a base priority:

```
G(u, f) = P_f · I_{u,cat(f)} · (1 + κ · π̄_u)
         └─── file mass ────┘   └── user pull ──┘
```

where `P_f` is Zipf popularity, `I_{u,cat(f)}` is the user's interest weight for the file's category, and `π̄_u` is the interest-similarity-weighted average meeting probability. The constant `κ = 5` was set by grid search.

### A2C Controller

An actor-critic network maps a 30-dimensional state vector (26 user features + 4 file features) to one of five priority multipliers: `{0.2×, 0.5×, 1.0×, 2.0×, 5.0×}`. The adjusted score `Ĝ = G · m_a` governs placement order. A greedy fill procedure places files in descending `Ĝ` order, guaranteeing full cache utilization.

### Training

- **Episodes:** 45 (each re-randomizes interest vectors and file requests)
- **Reward:** `r = 200φ + 100·CHR` (2:1 weighting of offloading over CHR)
- **Architecture:** 128→64 ReLU, actor head (5 logits) + critic head (scalar)
- **Optimizer:** Adam, lr = 3×10⁻³, entropy coefficient = 0.01

### Baselines

**Heuristics:** Popular Cache, Greedy, SAA, CFCA  
**DRL variants (same architecture, different base score):**
- `DQN-Interest`: interest-weighted base with 0.05 filter
- `DQN-Weak`: shallow interest base with 0.05 filter  
- `DQN-Pop`: popularity only, no filtering

---

## Ablation Results (1 GB, γ=0.6)

| Configuration | Offloading | Δ |
|--------------|-----------|---|
| Full RL-GACA (A2C) | **0.4914 ±.0058** | — |
| w/o A2C controller (gravity + greedy) | 0.2835 ±.0110 | −42.3% |
| w/o Gravity scoring (DQN on P_f only) | 0.3150 ±.0206 | −35.9% |
| w/o Interest filter | 0.4802 ±.0116 | −2.3% |
| w/o Priority order | 0.4910 ±.0057 | −0.1% |
| w/o Speed constraint (bug confirmed fixed) | 0.4914 ±.0058 | 0.0% |
| Aligned filter (I_uc ≥ 0.05) | **0.5177 ±.0068** | +5.4% |

The A2C controller is the larger contributor (−42.3%) over gravity scoring (−35.9%). The aligned variant exceeds the full model — the strict 0.005 filter is conservative.

---

## Citation

If you use this code or results, please cite:

```bibtex
Add when published
```

The T-Drive dataset:

```bibtex
@inproceedings{yuan2010tdrive,
  author    = {Jing Yuan and Yu Zheng and Chengyang Zhang and Wenlei Xie
               and Xing Xie and Guangzhong Sun and Yan Huang},
  title     = {T-Drive: Driving Directions Based on Taxi Trajectories},
  booktitle = {Proceedings of the 18th ACM SIGSPATIAL International Conference
               on Advances in Geographic Information Systems},
  pages     = {99--108},
  year      = {2010},
  doi       = {10.1145/1869790.1869807}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

The T-Drive dataset is provided by Microsoft Research under their own terms. See the [Microsoft Research publication page](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/) for dataset licensing.
