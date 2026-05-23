# RL-CWCA: Contact-Weighted Reinforcement Learning for Proactive Cache Allocation in Smart Urban Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Nabeel Ali, Asif Kabir, Muazzam A. Khan Khattak, Adeel Ali**  
> University of Kotli, AJK, Pakistan · Quaid-i-Azam University, Islamabad, Pakistan · University of Northumbria at Newcastle, UK  
> Submitted to *Sustainable Cities and Society* (Elsevier)

---

## Overview

RL-CWCA is a proactive D2D caching method that assigns each (user, file) pair a **contact-weighted (CW) score** — combining file popularity, user interest match, and interest-similarity-weighted meeting probability — and uses an **A2C controller** to adjust those scores and diversify cached content across devices. A greedy fill procedure guarantees full cache utilization by construction.

The key insight: existing heuristics push a single global file ranking to every device; existing D2D-DRL methods treat cache decisions as binary (cache or skip). RL-CWCA combines domain-structured scoring with a learned multiplier controller, producing a distinct ranked candidate list per device without binary placement decisions.

### Key Results (London Marylebone, 172 devices, 2,370 contact events)

| Metric | Value |
|--------|-------|
| Offloading at 1 GB | **0.5693** |
| Offloading at 2 GB | **0.7633** |
| vs CFCA at 2 GB (best heuristic) | +54.3% |
| vs DQN-Interest at 2 GB | +17.6% |
| vs DRL-Binary at 2 GB | +10.9% |
| Statistical significance (10 seeds) | p = 0.0002, r = 1.000 |
| Aligned filter variant (I_uc ≥ 0.05) at 2 GB | **0.8163** (+25.7% vs DQN-Interest) |
| Cache Hit Ratio at 1 GB | **0.4534** |
| Training time | ~75 s / seed (single CPU core) |
| Deployment time (172 devices) | ~8.2 s |

### Offloading Across Cache Sizes

| Method | 500 MB | 1 GB | 1.5 GB | 2 GB |
|--------|--------|------|--------|------|
| **RL-CWCA (A2C)** | **0.4375** | **0.5693** | **0.6693** | **0.7633** |
| DRL-Binary | 0.4080 | 0.5264 | 0.6207 | 0.6883 |
| DQN-Interest | 0.3710 | 0.5022 | 0.5790 | 0.6493 |
| SAA | 0.2920 | 0.3969 | 0.4760 | 0.5327 |
| CFCA | 0.2691 | 0.3702 | 0.4414 | 0.4948 |

### Transfer to T-Drive (Beijing Taxis, 200 taxis)

| Cache | Zero-shot | Cold-start (30 eps) | CFCA |
|-------|-----------|---------------------|------|
| 500 MB | 0.2056 | — | 0.2166 |
| 1 GB | 0.3163 | **0.3264** | 0.2978 |

Zero-shot beats CFCA at all three tested cache sizes. Cold-start retraining costs ~45 s/seed on a single CPU core and extends the lead at 1 GB to +9.6%.

---

## Repository Structure

```
rl-cwca/
├── rlcwca_main.py    # Complete experimental pipeline (all phases)
├── fetch_tdrive.py           # T-Drive dataset downloader and contact extractor
├── results/
│   ├── sumo_sim/
│   │   └── contact_data.npz         # SUMO Marylebone contact matrices
│   ├── tdrive/
│   │   └── tdrive_contact_data.npz  # T-Drive Beijing taxi contact matrices
│   └── runs/
│       └── <YYYYMMDD_HHMMSS>/
│           ├── figures/    # 300 DPI PDF + PNG publication figures
│           ├── tables/     # CSV tables (all results)
│           ├── logs/
│           └── meta.json   # Run configuration and summary
├── tables/                  # Pre-computed results CSVs (see below)
│   ├── table1_main.csv
│   ├── table2_cache_sweep.csv
│   ├── table3_ablation.csv
│   ├── table4_significance.csv
│   ├── table5_controller.csv
│   ├── table6_coverage_fairness.csv
│   ├── table7_sensitivity.csv
│   ├── table8_aligned_comparison.csv
│   └── table9_transfer_tdrive.csv
└── README.md
```

---

## Installation

```bash
git clone https://github.com/mnj89/rl-cwca.git
cd rl-cwca
pip install numpy scipy torch tqdm pandas matplotlib seaborn
```

**Requirements:** Python 3.8+, PyTorch (CPU is sufficient — no GPU needed).

No CUDA required. The full pipeline runs in under 15 minutes on a standard laptop CPU.

---

## Reproducing Results

### Step 1 — Marylebone SUMO trace (required)

The SUMO contact data is provided as a pre-processed NPZ file. Place it at:

```
results/sumo_sim/contact_data.npz
```

The file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `avg_ct` | (172, 172) | Mean contact duration per pair (seconds) |
| `n_ct` | (172, 172) | Number of contact events per pair |
| `vtypes_arr` | (172,) | Device type labels: `'car'`, `'bike'`, `'pedestrian'` |

### Step 2 — Run all phases (Marylebone)

```bash
python rlcwca_main.py
```

This runs all phases sequentially and writes results to a timestamped directory under `results/runs/`. Typical total runtime: **~12 minutes** on a single CPU core.

**What each phase produces:**

| Phase | Output |
|-------|--------|
| 1 | Trained A2C, PPO, DQN agents + heuristic and DRL baselines |
| 2 | `table1_main.csv` — main comparison (1 GB, 3 seeds) |
| 3 | `table2_cache_sweep.csv` — offloading at 4 cache sizes + `table8_aligned_comparison.csv` |
| 4 | `table3_ablation.csv` — five-component ablation |
| 5 | `table7_sensitivity.csv` — Zipf, file size, and hyperparameter sweeps |
| 6 | `table4_significance.csv` — Mann-Whitney U (10 seeds) |
| 7 | Coverage and fairness results |
| 8 | `table6_coverage_fairness.csv` — Jain index |
| 9 | Scalability figure (library size 100–400) |
| 10 | `table5_controller.csv` — A2C vs PPO vs DQN under shared CW scoring |
| 11 | All publication figures (300 DPI PDF + PNG) |
| 12 | All LaTeX-ready tables as CSV |

### Step 3 — T-Drive transfer validation (optional, ~3 min)

```bash
# Download and process T-Drive dataset (~35 MB)
python fetch_tdrive.py --n-taxis 200 --out results/tdrive/tdrive_contact_data.npz

# Run transfer phase using checkpoint from Step 2
python rlcwca_main.py --transfer
```

`fetch_tdrive.py` downloads the Microsoft Research T-Drive GPS dataset (no registration required), extracts contact events using the same 120 m radio range and 30% fragment threshold as the Marylebone model, and saves the NPZ.

The transfer phase evaluates in two modes:

- **Zero-shot**: Marylebone-trained weights applied directly to the T-Drive contact topology with no further training.
- **Cold-start retraining**: 30 A2C episodes on T-Drive from random initial weights (~45 s/seed).

Results are saved to `table9_transfer_tdrive.csv`.

---

## Pre-computed Results

All tables from the paper are provided in `tables/` so results can be verified without re-running the pipeline.

| File | Description |
|------|-------------|
| `table1_main.csv` | Main comparison: offloading and CHR at 1 GB, γ=0.6, 3 seeds |
| `table2_cache_sweep.csv` | Offloading at 500 MB / 1 GB / 1.5 GB / 2 GB |
| `table3_ablation.csv` | Five-component ablation + aligned filter variant |
| `table4_significance.csv` | Mann-Whitney U tests, 10 seeds |
| `table5_controller.csv` | A2C vs PPO vs DQN under shared CW base score |
| `table6_coverage_fairness.csv` | Request coverage and Jain's fairness index |
| `table7_sensitivity.csv` | Zipf skew, file size, κ, and multiplier set sweeps |
| `table8_aligned_comparison.csv` | Aligned threshold (I_uc ≥ 0.05) across 4 cache sizes |
| `table9_transfer_tdrive.csv` | T-Drive zero-shot and cold-start results |

---

## Method Summary

### Contact-Weighted (CW) Score

Each (user *u*, file *f*) pair receives a base priority:

```
CW(u, f) = P_f · I_{u,cat(f)} · (1 + κ · π̄_u)
           └── file mass ──┘   └── user contact pull ──┘
```

where:
- `P_f` — Zipf popularity of file *f*
- `I_{u,cat(f)}` — user *u*'s interest weight for the file's content category
- `π̄_u` — interest-similarity-weighted average meeting probability (contact reach)
- `κ = 5` — contact-weight constant, set by grid search over {1, 3, 5, 8}

The contact-reach term `π̄_u` is computed as an O(N) approximation of the full O(N²) pairwise matrix, keeping the state vector compact while preserving the dominant structure of the contact graph.

### A2C Controller

An actor-critic network maps a **30-dimensional state vector** (26 user features + 4 file features) to one of five priority multipliers: `{0.2×, 0.5×, 1.0×, 2.0×, 5.0×}`. The adjusted score `Ĝ = CW(u,f) · m_a` governs placement order. A greedy fill procedure places files in descending `Ĝ` order, guaranteeing full cache utilization.

**State vector composition:**

| Features | Dimensions | Description |
|----------|-----------|-------------|
| Contact reach | 1 | π̄_u (interest-similarity-weighted meeting probability) |
| Interest weights | 15 | I_{u,c} for each of the 15 content categories |
| Device type flags | 3 | One-hot: pedestrian / bicycle / vehicle |
| Mean contact duration | 1 | Seconds per encounter |
| Cache utilization | 1 | Current fill fraction |
| Neighbor density | 1 | Devices within radio range |
| Vehicle flag (repeated) | 1 | Explicit speed-constraint signal |
| File popularity | 1 | Zipf P_f |
| Normalized file size | 1 | s_f / 250 MB |
| Cache penetration | 1 | Fraction of devices caching this file |
| Content category | 1 | Integer category index |

### Training

- **Episodes:** 100 (each re-randomizes interest vectors and file requests; contact graph stays fixed)
- **Reward:** `r = w_off · φ + w_chr · CHR`, weight configuration selected per seed from {1:1, 2:1, 3:1} grid
- **Architecture:** Two independent fully connected networks (PolicyNet and ValueNet), 128 → 64 ReLU units; actor head outputs 5-way softmax over multipliers; critic head outputs scalar state value
- **Optimizer:** Adam, lr = 3×10⁻³, entropy coefficient = 0.01, gradient clip = 1.0
- **Cold-start retraining:** 30 episodes from random weights at lr = 3×10⁻³

### Baselines

**Heuristics:** Popular Cache, Greedy, SAA, CFCA

**DRL variants** (same A2C architecture, different base scoring function):
- `DRL-Binary`: same CW score as RL-CWCA, binary cache-or-skip action space
- `DQN-Interest`: interest-weighted base `P_f · (1 + 2·I_{u,c})` with 0.05 filter, DQN controller
- `DQN-Weak`: shallow interest base with 0.05 filter, DQN controller
- `DQN-Pop`: popularity only `P_f`, no filter, DQN controller

All DRL variants share the same state vector, multiplier set, and training budget. Only the base scoring function and/or learning algorithm changes.

---

## Ablation Results (1 GB, γ=0.6, 3 seeds)

| Configuration | Offloading | Δ |
|--------------|-----------|---|
| Full RL-CWCA (A2C) | **0.5693 ±.0091** | — |
| w/o CW score (DQN on P_f only) | 0.3575 ±.0036 | −37.2% |
| w/o A2C controller (CW + greedy fill) | 0.3702 ±.0086 | −35.0% |
| w/o Interest filter | 0.5562 ±.0147 | −2.3% |
| w/o Priority ordering | 0.5572 ±.0021 | −2.1% |
| Per-user reward variant | 0.5505 ±.0047 | −3.3% |
| Aligned filter (I_uc ≥ 0.05) | **0.6411 ±.0078** | +12.6% |

The CW score and the A2C controller contribute roughly equally and are both necessary. Without the CW score the system falls below DQN-Interest. Without the A2C controller the system reduces exactly to CFCA (0.3702). The 0.05 aligned filter is the recommended threshold for deployment.

---

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Area / City | 3.74 km² / London Marylebone |
| Devices | 172 (30 cars at 30 km/h, 30 bikes at 15 km/h, 112 pedestrians at 5 km/h) |
| Contact events | 2,370 pairwise events over 1,000 s |
| Mean / median contact duration | 166.3 s / 89.0 s |
| Radio range | 120 m |
| Cache sizes | 500 MB – 2 GB |
| File library | 250 files (main); 100–400 (scalability sweep) |
| Zipf γ | 0.6 (main); 0.6–1.2 (sensitivity sweep) |
| Seeds (performance / significance) | 3 / 10 |

---

## Citation

If you use this code or results, please cite:

```bibtex
% Add when published
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

The SUMO traffic simulator:

```bibtex
@inproceedings{sumo2018,
  author    = {Pablo Alvarez Lopez and Michael Behrisch and Laura Bieker-Walz
               and Jakob Erdmann and Yun-Pang Fl{\"o}tter{\"o}d and
               Robert Hilbrich and Leonhard L{\"u}cken and Johannes Rummel
               and Peter Wagner and Evamarie Wie{\ss}ner},
  title     = {Microscopic Traffic Simulation using SUMO},
  booktitle = {2018 21st International Conference on Intelligent Transportation Systems (ITSC)},
  year      = {2018},
  doi       = {10.1109/ITSC.2018.8569938}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

The T-Drive dataset is provided by Microsoft Research under their own terms. See the [Microsoft Research publication page](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/) for dataset licensing.
