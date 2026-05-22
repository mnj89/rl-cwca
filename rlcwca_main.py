#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  RL-CWCA: Complete Experimental Pipeline  
  ───────────────────────────────────────────────────────────
  Run management:
    Each execution creates a timestamped run directory:
      results/runs/<YYYYMMDD_HHMMSS>/figures/
      results/runs/<YYYYMMDD_HHMMSS>/tables/
      results/runs/<YYYYMMDD_HHMMSS>/logs/
      results/runs/<YYYYMMDD_HHMMSS>/meta.json
    A registry at results/runs/registry.json tracks all runs.
    The latest run ID is written to results/runs/latest.txt.

  Phases:
    1  Train all DRL agents (A2C primary + A2C/DQN comparison controllers)
    2  Main evaluation      (3 seeds — ALL DRL same seeds)
    3  Cache size sweep     (includes aligned variant)
    4  Ablation study       (same 3 seeds as main eval)
    5  Sensitivity sweeps   (Zipf γ, file size, cache size)
    6  10-seed significance (Mann-Whitney U)
    7  Contention sweep     (collision factor)
    8  Coverage & fairness  (Jain index)
    9  Scalability          (file library size)
   10  Controller comparison
   11  Generate all 13 publication figures (300 DPI PDF + PNG)
   12  Generate all LaTeX tables + CSV (including Table 8)

  Speed constraint bug fix (Rev. 3):
    'nospeed' filter now correctly removes vehicle penalty only
    (th = s.gt for all users) rather than inflating all thresholds
    to s.gt * 1.5.

  Aligned threshold (Rev. 3):
    RL-CWCA-aligned (I_uc >= 0.05) is now evaluated across all
    cache sizes and reported in Table 8.
═══════════════════════════════════════════════════════════════
"""

# ──────────────────────────────────────────────────────────────
# 0. IMPORTS & GLOBAL SETUP
# ──────────────────────────────────────────────────────────────
import os, sys, random, time, warnings, json
# Must be set BEFORE `import torch` for CUBLAS determinism to take effect.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '42')
# Force UTF-8 stdout/stderr on Windows so Greek/math/arrow glyphs in
# console output never trigger cp1252 encode errors mid-run.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, deque
from scipy import stats
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm, trange

warnings.filterwarnings('ignore')

# ── CLI args ─────────────────────────────────────────────────────
import argparse as _ap
_parser = _ap.ArgumentParser()
_parser.add_argument('--phase13', action='store_true',
                     help='Load checkpoint and run Phase 13 only (skip phases 2-12)')
_args, _ = _parser.parse_known_args()
_P13_ONLY = _args.phase13



# ──────────────────────────────────────────────────────────────
# RUN MANAGEMENT
# ──────────────────────────────────────────────────────────────
class RunManager:
    """
    Each call to new_run() creates a timestamped directory tree:
      results/runs/<run_id>/figures/
      results/runs/<run_id>/tables/
      results/runs/<run_id>/logs/
      results/runs/<run_id>/meta.json
    A global registry at results/runs/registry.json lists all runs.
    results/runs/latest.txt always points to the most recent run.
    """
    def __init__(self, base='results/runs'):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base / 'registry.json'

    def new_run(self, config=None):
        run_id  = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = self.base / run_id
        for sub in ('figures', 'tables', 'logs'):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        meta = {
            'run_id':   run_id,
            'created':  datetime.now().isoformat(),
            'config':   config or {},
            'status':   'running',
            'phases':   {},
        }
        self._write_meta(run_dir, meta)
        self._update_registry(run_id, meta)
        (self.base / 'latest.txt').write_text(run_id)
        print(f"  Run ID : {run_id}")
        print(f"  Run dir: {run_dir}")
        return run_id, run_dir

    def mark_phase(self, run_dir, phase_name, result_summary=None):
        meta = self._read_meta(run_dir)
        meta['phases'][phase_name] = {
            'completed': datetime.now().isoformat(),
            'summary':   result_summary or {},
        }
        self._write_meta(run_dir, meta)

    def complete_run(self, run_dir, results_summary=None):
        meta = self._read_meta(run_dir)
        meta['status']    = 'complete'
        meta['completed'] = datetime.now().isoformat()
        if results_summary:
            meta['results_summary'] = results_summary
        self._write_meta(run_dir, meta)
        self._update_registry(meta['run_id'], meta)
        print(f"  Run {meta['run_id']} marked complete.")

    def fail_run(self, run_dir, error):
        meta = self._read_meta(run_dir)
        meta['status'] = 'failed'
        meta['error']  = str(error)
        self._write_meta(run_dir, meta)

    def list_runs(self):
        reg = self._load_registry()
        rows = []
        for rid, m in sorted(reg.items()):
            rows.append({'run_id': rid, 'status': m.get('status','?'),
                         'created': m.get('created','')[:19]})
        return pd.DataFrame(rows)

    # ── internals ──
    def _write_meta(self, run_dir, meta):
        with open(Path(run_dir) / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

    def _read_meta(self, run_dir):
        with open(Path(run_dir) / 'meta.json') as f:
            return json.load(f)

    def _load_registry(self):
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}

    def _update_registry(self, run_id, meta):
        reg = self._load_registry()
        reg[run_id] = meta
        tmp = self.registry_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(reg, f, indent=2)
        os.replace(tmp, self.registry_path)


# ──────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS  (available before main() runs)
# ──────────────────────────────────────────────────────────────
GLOBAL_SEED = 42

# Task 12: per-episode offloading log. Each trainer call appends a list
# of phi values (res['off'] per episode) keyed by its `desc` argument.
# Read in Phase 11 to draw the per-agent offloading convergence panel.
_TRAINING_PHIS = {}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass
    # Pin BLAS to 1 thread: multi-threaded reductions in MKL/OpenMP are not
    # bit-reproducible across runs (accumulation order varies with scheduling),
    # which is the dominant source of run-to-run drift on CPU PyTorch.
    torch.set_num_threads(1)

# ── Publication style ──────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         15,
    'font.weight':       'bold',
    'axes.titlesize':    16,
    'axes.titleweight':  'bold',
    'axes.labelsize':    15,
    'axes.labelweight':  'bold',
    'xtick.labelsize':   13,
    'ytick.labelsize':   13,
    'legend.fontsize':   13,
    'legend.framealpha': 0.93,
    'legend.edgecolor':  '#bbbbbb',
    'legend.borderpad':  0.6,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.28,
    'grid.linestyle':    '--',
    'grid.linewidth':    0.7,
    'figure.dpi':        100,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'text.usetex':       False,
})

# Colorblind-safe, consistent per-algorithm styles
S = {
    'RL-CWCA':       dict(color='#E63946', marker='*',  ls='-',   lw=2.8, ms=11, z=5),
    'DQN-Interest':  dict(color='#1D6FA4', marker='o',  ls='--',  lw=2.1, ms=6,  z=4),
    'DQN-Weak':      dict(color='#2CA02C', marker='^',  ls='-.',  lw=2.1, ms=6,  z=4),
    'DQN-Pop':       dict(color='#FF7F0E', marker='s',  ls=':',   lw=1.9, ms=5,  z=3),
    'DRL-Binary':    dict(color='#F58518', marker='X',  ls='--',  lw=2.0, ms=6,  z=3),
    'CFCA':          dict(color='#1D6FA4', marker='o',  ls='--',  lw=2.1, ms=6,  z=3),
    'SAA':           dict(color='#2CA02C', marker='^',  ls='-.',  lw=2.1, ms=6,  z=3),
    'Greedy':        dict(color='#9467BD', marker='D',  ls=':',   lw=1.9, ms=5,  z=2),
    'Popular Cache': dict(color='#8C8C8C', marker='v',  ls=':',   lw=1.7, ms=5,  z=2),
    'A2C':           dict(color='#17BECF', marker='p',  ls='--',  lw=2.1, ms=7,  z=3),

    'PPO':           dict(color='#BCBD22', marker='h',  ls='-.',  lw=2.1, ms=7,  z=3),
    'DQN': dict(color='#8B4513', marker='D',  ls='--',  lw=2.0, ms=6,  z=3),
}

def savefig(fig, name):
    """300 DPI PDF (submit) + 150 DPI PNG (preview)."""
    fig.savefig(f'{FD}/{name}.pdf', dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(f'{FD}/{name}.png', dpi=150, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f'  [OK] {name}')

def ep(nm, **kw):
    """Shorthand: errorbar kwargs for algorithm nm."""
    return dict(color=S[nm]['color'], marker=S[nm]['marker'],
                ls=S[nm]['ls'], lw=S[nm]['lw'], ms=S[nm]['ms'],
                zorder=S[nm]['z'], capsize=4, **kw)

def lp(nm, **kw):
    """Shorthand: line plot kwargs for algorithm nm."""
    return dict(color=S[nm]['color'], marker=S[nm]['marker'],
                ls=S[nm]['ls'], lw=S[nm]['lw'], ms=S[nm]['ms'],
                zorder=S[nm]['z'], **kw)

# ──────────────────────────────────────────────────────────────
# 1. SIMULATION ENGINE
# ──────────────────────────────────────────────────────────────
_SUMO_NPZ = Path('results/sumo_sim/contact_data_172.npz')
if not _SUMO_NPZ.exists():
    raise FileNotFoundError(
        f"SUMO contact data not found: {_SUMO_NPZ}\n"
        "Run the SUMO simulation first (results/generate_contact_data_172.py)."
    )
data = np.load(_SUMO_NPZ, allow_pickle=True)
ACT = data['avg_ct']
NCT = data['n_ct']
VT  = list(data['vtypes_arr'])
ND  = len(VT)
# Task 3: state-dim ablation flag.
#   'full_30'    — paper-stated 30-dim layout; indices [23],[24] are placeholder
#                  zeros (speed, zone density).
#   'reduced_28' — drop [23],[24] entirely; state vector is 28-dim and the
#                  network input layer is correspondingly 28.
# All trainers, networks, and st_vec consult this flag at call time, so a
# single global assignment is enough to switch the whole pipeline.
STATE_DIM_MODE = os.environ.get('STATE_DIM_MODE', 'full_30')
assert STATE_DIM_MODE in ('full_30', 'reduced_28'), \
    f"STATE_DIM_MODE must be 'full_30' or 'reduced_28', got {STATE_DIM_MODE!r}"
SD  = 30 if STATE_DIM_MODE == 'full_30' else 28
MULTS = np.array([.2, .5, 1., 2., 5.])

# Human-readable labels for SUMO traces discovered by Phase 14 (cross-city
# eval). Keys are NPZ stems (filename without .npz); values are how the
# trace should appear in tables/figures. Unknown stems fall back to a
# title-cased version of the stem with underscores replaced by spaces.
_TRACE_LABEL_MAP = {
    'contact_data_172':           'Marylebone (London)',
    'contact_data_lust':           'Luxembourg (LuST, full day)',
    'contact_data_lust_morning':   'Luxembourg (LuST, AM peak)',
    'contact_data_lust_evening':   'Luxembourg (LuST, PM peak)',
    'contact_data_cologne':        'Cologne (TAPASCologne)',
    'contact_data_monaco':         'Monaco (MoST)',
    'contact_data_berlin':         'Berlin (BeST)',
    'contact_data_ingolstadt':     'Ingolstadt (InTAS)',
    'contact_data_bologna':        'Bologna',
}
print(f"Loaded SUMO: {ND} devices  (state-dim mode: {STATE_DIM_MODE}, SD={SD})\n")


class Sim:
    def __init__(s, F, gm, cmb, fmb, br, seed=42, coll_factor=1.0, _contacts=None):
        rng = np.random.RandomState(seed)
        s.seed = seed
        s._rng = rng                  # persistent RNG reused by exchange()
        if _contacts is None:
            act, nct, vt, nd = ACT, NCT, VT, ND
        else:
            act, nct, vt, nd = (_contacts['act'], _contacts['nct'],
                                 _contacts['vt'],  _contacts['nd'])
        s.N = nd; s.F = F; s.br = br; s.avg_ct = act
        s.coll = coll_factor
        mx = nct.max() if nct.max() > 0 else 1
        s.mp = nct / mx; np.fill_diagonal(s.mp, 0)
        s.ints = rng.dirichlet(np.ones(15), s.N)
        mn  = np.minimum(s.ints[:, None, :], s.ints[None, :, :]).sum(2)
        mx2 = np.maximum(s.ints[:, None, :], s.ints[None, :, :]).sum(2)
        s.isim = np.divide(mn, mx2, out=np.zeros_like(mn), where=mx2 > 0)
        np.fill_diagonal(s.isim, 0)
        # Vehicle-type flag per paper: 0 = pedestrian, 0.5 = bicycle, 1 = car/other vehicle.
        # The size-eligibility filter `s_f <= r_v * T_bar * (1 - 0.3 * vr)` then scales
        # smoothly across the three classes.
        _PED_TYPES  = {'pedestrian', 'ped', 'walking', 'type_pedestrian'}
        _BIKE_TYPES = {'bike', 'bicycle', 'type_bike'}
        def _vr_of(t):
            tl = str(t).lower()
            if tl in _PED_TYPES:  return 0.0
            if tl in _BIKE_TYPES: return 0.5
            return 1.0
        s.vr = np.array([_vr_of(t) for t in vt])
        rk = np.arange(1, F + 1, dtype=float)
        rp = rk ** (-gm); s.fp = rp / rp.sum()
        s.fc = np.arange(F) % 15
        s.fs = rng.uniform(max(10, fmb * .3), fmb * 2., F)
        v = act[act > 0]
        s.gt = br * np.mean(v) if len(v) > 0 else br * 60
        s.cache = np.zeros((s.N, F), bool)
        s.cu = np.zeros(s.N)
        s.tc = np.full(s.N, float(cmb))
        s.comb = s.mp * s.isim
        s.pri  = s.comb.mean(1)

    def clear(s):
        s.cache[:] = False; s.cu[:] = 0

    def put(s, u, f):
        z = s.fs[f]
        if s.cache[u, f] or s.cu[u] + z > s.tc[u]:
            return False
        s.cache[u, f] = True; s.cu[u] += z
        return True

    def exchange(s, nreq=15, delivery_threshold=0.30, max_neighbours=8):
        """
        D2D exchange with:
        - per-neighbor `delivery_threshold` fragment threshold (default 0.30)
        - up to `max_neighbours` providers tried per request, ranked by
          meeting probability (default 8)
        - coll_factor scales effective bit rate (contention model)
        - returns per-user offloading for Jain fairness

        Promoting the threshold and neighbour cap to parameters lets Phase 15
        sweep them without touching the model. Defaults match the paper
        (tau=0.30 per the CFCA-derived delivery condition, K=8 ranked by
        meeting probability).
        """
        rng = s._rng                  # use instance RNG so exchange() is seeded consistently
        tr = lh = dh = dd = 0.
        eff_br = s.br * s.coll
        per_u   = np.zeros(s.N)
        req_u   = np.zeros(s.N)
        chrwt_u = np.zeros(s.N)   # per-user CHR-weighted hits (Task 9)

        # Task 10: per-contact-type tally. Bikes and pedestrians are both
        # treated as 'P' (non-motor-vehicle) to match the paper's 4-type
        # scheme (P2P / V2V / V2P / P2V).
        _type_dh = {'P2P': 0,    'V2V': 0,    'V2P': 0,    'P2V': 0}
        _type_dd = {'P2P': 0.0,  'V2V': 0.0,  'V2P': 0.0,  'P2V': 0.0}

        for i in range(s.N):
            nr = rng.poisson(nreq)
            w  = s.fp * (.2 + .8 * s.ints[i, s.fc]); w /= w.sum()
            for fid in rng.choice(s.F, nr, p=w):
                tr += 1; req_u[i] += 1
                if s.cache[i, fid]:
                    lh += 1; per_u[i] += 1.0; chrwt_u[i] += 1.0; continue
                pv = np.where(s.cache[:, fid])[0]
                if len(pv) == 0: continue
                pr = s.mp[i, pv]
                k  = min(max_neighbours, len(pv))
                ti = np.argpartition(-pr, k)[:k] if k < len(pv) else np.arange(len(pv))
                delta = 0.0
                _contribs = []   # (provider_idx, fraction-of-file delivered)
                for idx in ti:
                    j = pv[idx]
                    if rng.random() < pr[idx]:
                        ct   = s.avg_ct[i, j]
                        frac = min(s.fs[fid], eff_br * ct) / s.fs[fid]
                        if frac >= delivery_threshold:
                            actual = min(frac, 1.0 - delta)
                            _contribs.append((j, actual))
                            delta = min(1.0, delta + frac)
                            if delta >= 1.0: break
                if delta > 0:
                    dh += 1; dd += delta * s.fs[fid]
                    chrwt_u[i] += 0.7 * delta            # D2D weighted at 0.7
                    # Task 10: attribute bytes per provider type.
                    _req = 'V' if s.vr[i] >= 1.0 else 'P'
                    for _j, _portion in _contribs:
                        _prv = 'V' if s.vr[_j] >= 1.0 else 'P'
                        _key = f'{_req}2{_prv}'
                        _type_dh[_key] += 1
                        _type_dd[_key] += _portion * s.fs[fid]
                per_u[i] += delta

        off = (lh + dh) / max(tr, 1)
        ch  = (lh + .7 * dh) / max(tr, 1)
        cu  = s.cu.mean() / s.tc.mean()
        d2d = dh / max(dh + max(tr - lh - dh, 0), 1)
        lr  = lh / max(tr, 1)

        # per-user offloading ratio (for Jain + Task 9 per-user reward)
        phi      = np.where(req_u > 0, per_u   / req_u, 0.0)
        chr_per_u = np.where(req_u > 0, chrwt_u / req_u, 0.0)
        active = phi[req_u > 0]
        jain = (active.sum()**2 / (len(active) * (active**2).sum())
                if len(active) > 1 else 1.0)
        cov  = (per_u > 0).sum() / max(s.N, 1)   # fraction of users served

        return dict(off=off, chr=ch, lh=lh, dh=dh, tr=tr, dd=dd,
                    cu=cu, d2d=d2d, lr=lr, jain=jain, cov=cov,
                    phi_per_u=phi, chr_per_u=chr_per_u,
                    type_dh=_type_dh, type_dd=_type_dd)


def st_vec(s, u, f):
    """
    30-dim state vector per the paper:
      [0:15]   interest weights (15 categories)
      [15]     mean meeting probability
      [16]     max  meeting probability
      [17]     mean Jaccard similarity with nearby providers
      [18]     max  Jaccard similarity with nearby providers
      [19]     cache usage fraction
      [20]     cached file count (normalised by F)
      [21]     vehicle-type flag (0 ped, 0.5 bike, 1 vehicle)
      [22]     mean contact duration (normalised by 300 s)
      [23]     speed         -- ALWAYS 0.0 (paper-reserved placeholder)
      [24]     zone density  -- ALWAYS 0.0 (paper-reserved placeholder)
      [25]     vehicle-type flag repeated as joint scoring context
      [26]     Zipf popularity P_f  (scaled by 100)
      [27]     normalised file size  s_f / 250
      [28]     fraction of devices currently caching f
      [29]     content category index (normalised by 15)
    """
    if STATE_DIM_MODE == 'reduced_28':
        # 28-dim variant: drop the two zero placeholders [23] speed and
        # [24] zone density. All other indices keep the same semantic
        # interpretation (vr repeat at [25] stays as [23] in this layout).
        return np.concatenate([
            s.ints[u],
            [s.mp[u].mean(), s.mp[u].max(),
             s.isim[u].mean(), s.isim[u].max(),
             s.cu[u] / max(s.tc[u], 1),
             s.cache[u].sum() / max(s.F, 1),
             s.vr[u],
             s.avg_ct[u].mean() / 300.,
             s.vr[u]],                                                # vr repeat
            [s.fp[f] * 100,
             s.fs[f] / 250.,
             s.cache[:, f].sum() / max(s.N, 1),
             s.fc[f] / 15.]
        ])
    return np.concatenate([
        s.ints[u],                                                   # [0:15]
        [s.mp[u].mean(),                                              # [15]
         s.mp[u].max(),                                               # [16]
         s.isim[u].mean(),                                            # [17]
         s.isim[u].max(),                                             # [18]
         s.cu[u] / max(s.tc[u], 1),                                   # [19]
         s.cache[u].sum() / max(s.F, 1),                              # [20]
         s.vr[u],                                                     # [21]
         s.avg_ct[u].mean() / 300.,                                   # [22]
         0.0,                                                          # [23] speed placeholder
         0.0,                                                          # [24] zone density placeholder
         s.vr[u]],                                                    # [25] vr repeat
        [s.fp[f] * 100,                                               # [26]
         s.fs[f] / 250.,                                              # [27]
         s.cache[:, f].sum() / max(s.N, 1),                           # [28]
         s.fc[f] / 15.]                                               # [29]
    ])


# ──────────────────────────────────────────────────────────────
# 2. HEURISTIC ALGORITHMS
# ──────────────────────────────────────────────────────────────
def a_pop(s):
    s.clear()
    for u in range(s.N):
        for f in np.argsort(-s.fp): s.put(u, int(f))

def a_greedy(s):
    s.clear()
    for u in range(s.N):
        th = s.gt * (1 - .5 * s.vr[u])
        for f in np.argsort(-s.fp):
            f = int(f)
            if s.fs[f] <= th: s.put(u, f)

def a_saa(s):
    s.clear()
    for u in np.argsort(-s.pri):
        th = s.gt * (1 - .4 * s.vr[u])
        for f in np.argsort(-s.fp):
            f = int(f)
            if s.ints[u, s.fc[f]] < .02 or s.fs[f] > th: continue
            s.put(u, f)

def a_cfca(s):
    s.clear()
    for u in np.argsort(-s.pri):
        th = s.gt * (1 - .4 * s.vr[u])
        for f in np.argsort(-s.fp):
            f = int(f)
            if s.ints[u, s.fc[f]] < .01 or s.fs[f] > th: continue
            s.put(u, f)


# ──────────────────────────────────────────────────────────────
# 3. DRL NETWORKS
# ──────────────────────────────────────────────────────────────
class DQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SD, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 5))
    def forward(self, x): return self.net(x)


class BinaryDQNet(nn.Module):
    """Cache-or-skip binary baseline (Task 2). Identical FC stack to DQNet
    but the output head has 2 logits: action 0 = do not cache, 1 = cache.
    Trained with the same DQN epsilon-greedy schedule and reward as the
    5-multiplier DQNs; differs only in the action space."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SD, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2))
    def forward(self, x): return self.net(x)


class PolicyNet(nn.Module):
    """Actor head for A2C/PPO."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SD, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 5))
    def forward(self, x): return self.net(x)


class ValueNet(nn.Module):
    """Critic head for A2C/PPO."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SD, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────
# 3b. RL CONTROLLERS
# ──────────────────────────────────────────────────────────────
CONTROLLER = 'A2C'   # primary RL controller for RL-CWCA (v6 — back to paper-stated A2C)


class PPOController:
    """PPO (Proximal Policy Optimization) controller for RL-CWCA.

    API
    ---
    select_action(state) → (action_idx, log_prob, value)
    update(transitions)  → loss_dict
    save(path) / load(path)
    """
    def __init__(self, lr=3e-3, clip=0.2, ppo_epochs=4, gae_lambda=0.95,
                 ent_coef=0.01, val_coef=0.5):
        self.actor      = PolicyNet()
        self.critic     = ValueNet()
        self.opt        = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.clip       = clip
        self.ppo_epochs = ppo_epochs
        self.gae_lambda = gae_lambda
        self.ent_coef   = ent_coef
        self.val_coef   = val_coef

    def select_action(self, state):
        """state: np.ndarray (SD,) → (action_idx, log_prob, value)"""
        sv_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(sv_t)
            probs  = torch.softmax(logits, -1)
            act    = torch.multinomial(probs, 1).item()
            lp     = torch.log(probs[0, act] + 1e-8).item()
            val    = self.critic(sv_t).item()
        return act, lp, val

    def update(self, transitions):
        """transitions: list of (state, action, old_log_prob, advantage, return_)
        Returns dict with policy_loss, value_loss, entropy."""
        states     = torch.FloatTensor(np.array([t[0] for t in transitions]))
        actions    = torch.LongTensor([t[1] for t in transitions])
        old_lps    = torch.FloatTensor([t[2] for t in transitions])
        advantages = torch.FloatTensor([t[3] for t in transitions])
        returns    = torch.FloatTensor([t[4] for t in transitions])

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = len(transitions)
        total_pl = total_vl = total_el = 0.0

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(n).numpy()
            for start in range(0, n, 64):
                mb  = idx[start:start + 64]
                sb  = states[mb]; ab = actions[mb]; olb = old_lps[mb]
                adv_b = advantages[mb]; ret_b = returns[mb]

                logits = self.actor(sb); vals = self.critic(sb)
                lp_all = torch.log_softmax(logits, -1)
                nlp    = lp_all.gather(1, ab.unsqueeze(1)).squeeze()
                ratio  = torch.exp(nlp - olb)
                s1     = ratio * adv_b
                s2     = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_b
                pl     = -torch.min(s1, s2).mean()
                vl     = nn.MSELoss()(vals, ret_b)
                el     = -(torch.softmax(logits, -1) * lp_all).sum(1).mean()
                loss   = pl + self.val_coef * vl - self.ent_coef * el
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 1.)
                self.opt.step()
                total_pl += pl.item(); total_vl += vl.item(); total_el += el.item()

        return {'policy_loss': total_pl, 'value_loss': total_vl, 'entropy': total_el}

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict()}, path)

    def load(self, path):
        d = torch.load(path, map_location='cpu', weights_only=True)
        self.actor.load_state_dict(d['actor'])
        self.critic.load_state_dict(d['critic'])




# ──────────────────────────────────────────────────────────────
# 4. SCORING FUNCTIONS & ORDERING
# ──────────────────────────────────────────────────────────────
def make_b_cw_score(kappa=5):
    """Factory: returns a CW-score function with the given κ baked in.

    The CW score is  G(u, f) = P_f · I_{u, c(f)} · (1 + κ · π̄_u). κ controls
    how strongly the user's mean meeting probability π̄_u amplifies the gravity
    term. The default κ = 5 is the paper's calibrated value; the κ-sweep
    script (`task_kappa_sweep.py`) compares κ ∈ {1, 3, 5, 8} by training reward.
    """
    def _scorer(s, u, f):
        return s.fp[f] * s.ints[u, s.fc[f]] * (1 + s.pri[u] * kappa)
    _scorer.__name__ = f'b_cw_score' if kappa == 5 else f'b_cw_score_k{kappa}'
    _scorer.kappa = kappa
    return _scorer

# Default RL-CWCA scorer: κ = 5. All call sites that import `b_cw_score`
# receive the κ=5 version; the κ-sweep produces alternates via the factory.
b_cw_score  = make_b_cw_score(5)
def b_interest(s, u, f): return s.fp[f] * (1 + s.ints[u, s.fc[f]] * 2)
def b_weak(s, u, f):     return s.fp[f] * (1 + s.ints[u, s.fc[f]] * .5)
def b_pop(s, u, f):      return s.fp[f]
def o_pri(s):  return np.argsort(-s.pri)
def o_rand(s): idx = np.arange(s.N); s._rng.shuffle(idx); return idx


def norm_pri(raw_pri):
    """
    Normalise meeting-probability scores to [0, 1] within a dataset.

    Why this is needed
    ──────────────────
    b_cw_score computes:  G(u,f) = Pf * I_{u,c} * (1 + κ * π̄_u)   κ=5

    κ=5 was calibrated so the pull term spans 1–6 on Marylebone, where
    π̄_u ∈ [0.05, 0.80].  On T-Drive (sparse taxi contacts) π̄_u ∈ [0.01, 0.08],
    so the pull term only spans 1.05–1.40 — a 1.33× range instead of 4×.
    The contact-weighted (CW) score loses almost all contact-based differentiation.

    Normalising π̄_u to [0, 1] per dataset restores the pull term to 1–6
    regardless of the dataset's absolute contact frequencies, so the gravity
    function is topology-agnostic.
    """
    lo, hi = raw_pri.min(), raw_pri.max()
    if hi - lo < 1e-9:          # all equal — no contact heterogeneity
        return np.zeros_like(raw_pri)
    return (raw_pri - lo) / (hi - lo)

# DRL agent configs — corrected names
# filt: 'full'=0.005 threshold, 'aligned'=0.05 threshold, 'none'=no filter
# Controller key: 'a2c' | 'ppo' | 'dqn'
AGENT_CONFIGS = {
    'RL-CWCA':      (b_cw_score,  'full',    'pri', 'a2c'),  # A2C is primary controller
    'DQN-Interest': (b_interest, 'aligned', 'pri', 'dqn'),
    'DQN-Weak':     (b_weak,     'aligned', 'rand','dqn'),
    'DQN-Pop':      (b_pop,      'none',    'rand','dqn'),
    # Task 2: binary cache-or-skip baseline using the SAME CW score base
    # and SAME state vector as RL-CWCA. The point of comparison is the
    # 5-multiplier head vs the 2-action head, all else held equal.
    'DRL-Binary':   (b_cw_score, 'full',    'pri', 'binary_dqn'),
}


# ──────────────────────────────────────────────────────────────
# ── Checkpoint support ──────────────────────────────────────────
_CKPT = Path('results/checkpoints/agents_v6_a2c.pt')
_BFN_MAP  = {'b_cw_score': b_cw_score, 'b_interest': b_interest,
             'b_weak': b_weak, 'b_pop': b_pop}
_UORD_MAP = {'o_pri': o_pri, 'o_rand': o_rand}
_NET_MAP  = {'PolicyNet': PolicyNet, 'DQNet': DQNet, 'BinaryDQNet': BinaryDQNet}

def _save_checkpoint(agents, w_off=None, w_chr=None):
    _CKPT.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {'__meta__': {'w_off': w_off, 'w_chr': w_chr}}
    for name, (net, bfn, filt, uord) in agents.items():
        entry = {
            'state_dict': net.state_dict(),
            'net_type': net.__class__.__name__,
            'bfn':  bfn.__name__,
            'filt': filt,
            'uord': uord.__name__,
        }
        if hasattr(net, '_ctrl_critic'):
            entry['critic_state_dict'] = net._ctrl_critic.state_dict()
        ckpt[name] = entry
    torch.save(ckpt, _CKPT)
    print(f'  [CKPT] Saved {len(ckpt) - 1} agents -> {_CKPT}  '
          f'(w_off={w_off}, w_chr={w_chr})')

def _load_checkpoint():
    """Returns (agents, meta_dict). meta_dict has w_off / w_chr if saved,
    None otherwise (older checkpoints from before the meta entry was added)."""
    # weights_only=True trips on the __meta__ dict (Python int values are
    # fine, but torch.save serialises it via pickle); allow full unpickle
    # since this file is a local artifact we just wrote.
    ckpt = torch.load(_CKPT, map_location='cpu', weights_only=False)
    meta = ckpt.pop('__meta__', {}) if isinstance(ckpt.get('__meta__'), dict) else {}
    agents = {}
    for name, d in ckpt.items():
        net = _NET_MAP[d['net_type']]()
        # Strip any _ctrl_critic.* keys that may have leaked into older state_dicts
        # before object.__setattr__ was used to bypass nn.Module auto-registration.
        _clean_sd = {k: v for k, v in d['state_dict'].items()
                     if not k.startswith('_ctrl_critic')}
        net.load_state_dict(_clean_sd)
        if 'critic_state_dict' in d:
            critic = ValueNet()
            critic.load_state_dict(d['critic_state_dict'])
            object.__setattr__(net, '_ctrl_critic', critic)
        agents[name] = (net, _BFN_MAP[d['bfn']], d['filt'], _UORD_MAP[d['uord']])
    if meta.get('w_off') is not None:
        print(f'  [CKPT] Loaded {len(agents)} agents from {_CKPT}  '
              f'(w_off={meta["w_off"]}, w_chr={meta["w_chr"]})')
    else:
        print(f'  [CKPT] Loaded {len(agents)} agents from {_CKPT}  '
              '(no reward-weight meta — checkpoint pre-dates this feature)')
    return agents, meta

# 5. TRAINING FUNCTIONS
# ──────────────────────────────────────────────────────────────
def _candidates(s, filt, uord, topF=28):
    """Yield (user, file, state_vec, base_cw_score) candidate pairs."""
    cands = []
    for u in uord(s)[:min(s.N, 55)]:
        if filt == 'none':
            th = s.gt * 1.5          # no size filter at all
        elif filt == 'nospeed':
            # FIX: remove vehicle penalty only — treat vehicles same as pedestrians.
            # Previously used s.gt * 1.5 for ALL users, incorrectly relaxing
            # pedestrian/bike thresholds too.  Correct: th = s.gt for every user.
            th = s.gt
        else:
            th = s.gt * (1 - .3 * s.vr[u])   # 'full' / 'aligned'
        for fi in np.argsort(-s.fp)[:topF]:
            fi = int(fi); c = s.fc[fi]
            if s.fs[fi] > th: continue
            if (filt == 'full' or filt == 'nospeed') and s.ints[u, c] < .005: continue
            elif filt == 'aligned' and s.ints[u, c] < .050: continue
            cands.append((u, fi))
    return cands


def train_dqn(s, eps, base_fn, filt, uord_fn, seed=GLOBAL_SEED, desc='DQN',
              w_off=200, w_chr=100):
    set_seed(seed)
    net = DQNet(); tgt = DQNet(); tgt.load_state_dict(net.state_dict())
    opt = optim.Adam(net.parameters(), lr=1e-3)
    buf = deque(maxlen=12000); best_r = -1e9; best_w = None; rews = []; ev = 1.
    _TRAINING_PHIS[desc] = []   # Task 12: per-episode phi log

    for ep in trange(eps, desc=f'  {desc}', leave=False, ncols=80,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ep [{elapsed}<{remaining}]'):
        s.clear()
        cands = _candidates(s, filt, uord_fn)
        sc = []
        for u, fi in cands:
            base = base_fn(s, u, fi); sv = st_vec(s, u, fi)
            act = (random.randint(0, 4) if random.random() < ev
                   else net(torch.FloatTensor(sv).unsqueeze(0)).argmax(1).item())
            sc.append((u, fi, base * MULTS[act], sv, act, base))

        sc.sort(key=lambda x: -x[2])
        for u, fi, _, _, _, _ in sc: s.put(u, fi)
        res = s.exchange(); gr = res['off'] * w_off + res['chr'] * w_chr
        _TRAINING_PHIS[desc].append(res['off'])

        for _, _, _, sv, act, bs in sc:
            buf.append((sv, act, bs * MULTS[act] * 10 + gr / max(len(sc), 1), sv.copy()))

        if len(buf) >= 64:
            for _ in range(4):
                batch = [buf[i] for i in random.sample(range(len(buf)), 64)]
                ss, aa, rr, nn_ = zip(*batch)
                ss  = torch.FloatTensor(np.array(ss)); aa = torch.LongTensor(aa)
                rr  = torch.FloatTensor(rr);           nn_= torch.FloatTensor(np.array(nn_))
                qv  = net(ss).gather(1, aa.unsqueeze(1)).squeeze()
                with torch.no_grad(): nq = tgt(nn_).max(1)[0]
                loss = nn.MSELoss()(qv, rr + .95 * nq)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()

        if ep % 5 == 0: tgt.load_state_dict(net.state_dict())
        ev = max(.05, ev - .95 / eps); rews.append(gr)
        if gr > best_r:
            best_r = gr
            best_w = {k: v.clone() for k, v in net.state_dict().items()}

    if best_w: net.load_state_dict(best_w)
    return net, rews


def train_binary_dqn(s, eps, base_fn, filt, uord_fn, seed=GLOBAL_SEED,
                     desc='DRL-Binary', w_off=200, w_chr=100):
    """Cache-or-skip binary baseline (Task 2).

    Each (user, file) candidate gets a binary action (0=skip, 1=cache).
    Placement: pairs are processed in descending CW-score order; pairs
    with action=1 fill the device cache until full. Reward, replay buffer,
    target sync and epsilon schedule match train_dqn -- only the action
    space changes."""
    set_seed(seed)
    net = BinaryDQNet(); tgt = BinaryDQNet(); tgt.load_state_dict(net.state_dict())
    opt = optim.Adam(net.parameters(), lr=1e-3)
    buf = deque(maxlen=12000); best_r = -1e9; best_w = None; rews = []; ev = 1.
    _TRAINING_PHIS[desc] = []

    for ep in trange(eps, desc=f'  {desc}', leave=False, ncols=80,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ep [{elapsed}<{remaining}]'):
        s.clear()
        cands = _candidates(s, filt, uord_fn)
        sc = []
        for u, fi in cands:
            base = base_fn(s, u, fi); sv = st_vec(s, u, fi)
            act = (random.randint(0, 1) if random.random() < ev
                   else net(torch.FloatTensor(sv).unsqueeze(0)).argmax(1).item())
            sc.append((u, fi, base, sv, act))

        # Place in descending CW-score order, accepting only action=1 pairs.
        # s.put() returns False when the cache is full so subsequent puts
        # on the same user become no-ops automatically.
        sc.sort(key=lambda x: -x[2])
        for u, fi, _, _, action in sc:
            if action == 1:
                s.put(u, fi)

        res = s.exchange(); gr = res['off'] * w_off + res['chr'] * w_chr
        _TRAINING_PHIS[desc].append(res['off'])

        # Reward shaping: each transition gets the base CW score if it was
        # accepted (action=1) plus the shared episode bonus. Skipped pairs
        # earn nothing from the base component, so the agent learns to say
        # action=1 on pairs with high base score and action=0 on the rest.
        for _, _, base, sv, act in sc:
            r_tx = (base * 10 if act == 1 else 0.0) + gr / max(len(sc), 1)
            buf.append((sv, act, r_tx, sv.copy()))

        if len(buf) >= 64:
            for _ in range(4):
                batch = [buf[i] for i in random.sample(range(len(buf)), 64)]
                ss, aa, rr, nn_ = zip(*batch)
                ss  = torch.FloatTensor(np.array(ss)); aa = torch.LongTensor(aa)
                rr  = torch.FloatTensor(rr);           nn_= torch.FloatTensor(np.array(nn_))
                qv  = net(ss).gather(1, aa.unsqueeze(1)).squeeze()
                with torch.no_grad(): nq = tgt(nn_).max(1)[0]
                loss = nn.MSELoss()(qv, rr + .95 * nq)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()

        if ep % 5 == 0: tgt.load_state_dict(net.state_dict())
        ev = max(.05, ev - .95 / eps); rews.append(gr)
        if gr > best_r:
            best_r = gr
            best_w = {k: v.clone() for k, v in net.state_dict().items()}

    if best_w: net.load_state_dict(best_w)
    return net, rews


def train_a2c(s, eps, base_fn, filt, uord_fn, seed=GLOBAL_SEED,
              lr=3e-3, gamma=0.95, ent=0.01, vc=0.5, desc='A2C',
              w_off=200, w_chr=100,
              pretrained_actor=None, pretrained_critic=None,
              reward_mode='shared'):
    """A2C trainer. lr=3e-3 per paper for main training; pass lr=3e-4 for
    Phase 13 T-Drive fine-tune (one-tenth the main rate).

    reward_mode:
      'shared'   (default) — every transition in an episode receives the
                  same return  R = (w_off * phi + w_chr * CHR) / n_tx.
                  Matches the v5 / original-paper recipe.
      'per_user' — Task 9 ablation. Each transition for user u receives
                  R_u = w_off * phi_u + w_chr * CHR_u, so the credit
                  assignment is per-user rather than episode-aggregate.
    """
    assert reward_mode in ('shared', 'per_user'), \
        f"reward_mode must be 'shared' or 'per_user', got {reward_mode!r}"
    set_seed(seed)
    actor = PolicyNet(); critic = ValueNet()
    if pretrained_actor is not None:
        _src_sd = {k: v for k, v in pretrained_actor.state_dict().items()
                   if not k.startswith('_ctrl_critic')}
        actor.load_state_dict(_src_sd)
    if pretrained_critic is not None:
        critic.load_state_dict(pretrained_critic.state_dict())
    opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    best_r = -1e9; best_w = None; rews = []
    _TRAINING_PHIS[desc] = []   # Task 12: per-episode phi log

    for ep in trange(eps, desc=f'  {desc}', leave=False, ncols=80,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ep [{elapsed}<{remaining}]'):
        s.clear()
        cands = _candidates(s, filt, uord_fn)
        svs, acts_t, users_t, cand_pairs = [], [], [], []
        for u, fi in cands:
            sv_t = torch.FloatTensor(st_vec(s, u, fi)).unsqueeze(0)
            with torch.no_grad():
                logits = actor(sv_t)
                probs  = torch.softmax(logits, -1)
                act    = torch.multinomial(probs, 1).item()
            svs.append(st_vec(s, u, fi)); acts_t.append(act); users_t.append(u)
            cand_pairs.append((u, fi, base_fn(s, u, fi) * MULTS[act]))

        cand_pairs.sort(key=lambda x: -x[2])
        for u, fi, _ in cand_pairs: s.put(u, fi)
        res = s.exchange(); gr = res['off'] * w_off + res['chr'] * w_chr; rews.append(gr)
        _TRAINING_PHIS[desc].append(res['off'])

        if svs:
            sb = torch.FloatTensor(np.array(svs))
            ab = torch.LongTensor(acts_t)
            if reward_mode == 'per_user':
                phi_u = res['phi_per_u']; chr_u = res['chr_per_u']
                ret_list = [w_off * float(phi_u[u]) + w_chr * float(chr_u[u])
                            for u in users_t]
                ret = torch.FloatTensor(ret_list)
            else:
                ret = torch.FloatTensor([gr / max(len(svs), 1)] * len(svs))
            logits = actor(sb); vals = critic(sb)
            lp_all = torch.log_softmax(logits, -1)
            adv    = (ret - vals.detach())
            pl = -(lp_all.gather(1, ab.unsqueeze(1)).squeeze() * adv).mean()
            vl = nn.MSELoss()(vals, ret)
            ent_l = -(torch.softmax(logits, -1) * lp_all).sum(1).mean()
            loss  = pl + vc * vl - ent * ent_l
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.)
            opt.step()

        if gr > best_r:
            best_r = gr; best_w = {k: v.clone() for k, v in actor.state_dict().items()}

    if best_w: actor.load_state_dict(best_w)
    # Attach critic on the actor so checkpoint save/load preserves both
    # halves of the A2C policy (same pattern as train_ppo).
    object.__setattr__(actor, '_ctrl_critic', critic)
    return actor, rews


def train_ppo(s, eps, base_fn, filt, uord_fn, seed=GLOBAL_SEED,
              lr=3e-3, clip=0.2, ppo_ep=4, ent=0.01, vc=0.5, desc='PPO',
              w_off=200, w_chr=100, pretrained_actor=None, pretrained_critic=None):
    set_seed(seed)
    ctrl = PPOController(lr=lr, clip=clip, ppo_epochs=ppo_ep,
                         gae_lambda=0.95, ent_coef=ent, val_coef=vc)
    if pretrained_actor is not None:
        # Strip any attached non-actor keys (e.g. _ctrl_critic.*) so loading
        # into a fresh PolicyNet does not raise on unexpected keys.
        _src_sd = {k: v for k, v in pretrained_actor.state_dict().items()
                   if not k.startswith('_ctrl_critic')}
        ctrl.actor.load_state_dict(_src_sd)
    if pretrained_critic is not None:
        ctrl.critic.load_state_dict(pretrained_critic.state_dict())
    elif pretrained_actor is not None:
        # Critic-only warm-up: freeze actor for the first N episodes so the critic
        # learns the value function before joint PPO begins.  Without this, random
        # critic values produce adversarial advantages (normalised to -val/std) that
        # immediately corrupt the well-trained actor weights.
        _wu_n = min(8, max(1, eps // 4))
        for _p in ctrl.actor.parameters(): _p.requires_grad_(False)
        for _ in range(_wu_n):
            s.clear()
            _wu_cands = _candidates(s, filt, uord_fn)
            _wu_pairs = []; _wu_roll = []
            for u, fi in _wu_cands:
                sv = st_vec(s, u, fi)
                a, lp, v = ctrl.select_action(sv)
                _wu_pairs.append((u, fi, base_fn(s, u, fi) * MULTS[a]))
                _wu_roll.append((sv, a, lp, v))
            _wu_pairs.sort(key=lambda x: -x[2])
            for u, fi, _ in _wu_pairs: s.put(u, fi)
            _wu_res = s.exchange(); _wu_gr = _wu_res['off'] * w_off + _wu_res['chr'] * w_chr
            if _wu_roll:
                _wu_n2 = len(_wu_roll); _wu_ret = _wu_gr / max(_wu_n2, 1)
                ctrl.update([(sv, a, lp, _wu_ret - v, _wu_ret)
                             for sv, a, lp, v in _wu_roll])
        for _p in ctrl.actor.parameters(): _p.requires_grad_(True)
        s.clear()
    best_r = -1e9; best_w = None; rews = []
    _TRAINING_PHIS[desc] = []   # Task 12: per-episode phi log

    for ep in trange(eps, desc=f'  {desc}', leave=False, ncols=80,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ep [{elapsed}<{remaining}]'):
        s.clear()
        cands = _candidates(s, filt, uord_fn)
        rollout    = []   # (state, action, log_prob, value)
        cand_pairs = []
        for u, fi in cands:
            sv = st_vec(s, u, fi)
            act, lp, val = ctrl.select_action(sv)
            cand_pairs.append((u, fi, base_fn(s, u, fi) * MULTS[act]))
            rollout.append((sv, act, lp, val))

        cand_pairs.sort(key=lambda x: -x[2])
        for u, fi, _ in cand_pairs: s.put(u, fi)
        res = s.exchange(); gr = res['off'] * w_off + res['chr'] * w_chr; rews.append(gr)
        _TRAINING_PHIS[desc].append(res['off'])

        if rollout:
            n    = len(rollout)
            ret_ = gr / max(n, 1)
            # GAE advantage (λ=0.95): single shared episode reward → adv_t = ret - V_t
            transitions = [
                (sv, act, lp, ret_ - val, ret_)
                for sv, act, lp, val in rollout
            ]
            ctrl.update(transitions)

        if gr > best_r:
            best_r = gr
            best_w = {k: v.clone() for k, v in ctrl.actor.state_dict().items()}

    if best_w: ctrl.actor.load_state_dict(best_w)
    # Attach critic via object.__setattr__ to bypass nn.Module's auto-registration
    # (otherwise the critic shows up in actor.state_dict() and breaks load_state_dict
    # into a fresh PolicyNet).
    object.__setattr__(ctrl.actor, '_ctrl_critic', ctrl.critic)
    return ctrl.actor, rews


def deploy(s, net, base_fn, filt, uord_fn):
    """Greedy deterministic deployment.

    Three placement paths depending on the trained policy type:
      DQNet / PolicyNet  (5-output, multiplier head)
        -> rank candidates by base * MULTS[argmax], place in that order.
      BinaryDQNet (2-output, cache-or-skip head)
        -> rank candidates by base CW score only; place where argmax==1
           until the cache fills (Task 2)."""
    s.clear(); cands = []; bases = []; net.eval()
    for u in uord_fn(s):
        if filt == 'none':
            th = s.gt * 1.5
        elif filt == 'nospeed':
            th = s.gt
        else:
            th = s.gt * (1 - .3 * s.vr[u])
        for fi in np.argsort(-s.fp):
            fi = int(fi); c = s.fc[fi]
            if s.fs[fi] > th: continue
            if (filt == 'full' or filt == 'nospeed') and s.ints[u, c] < .005: continue
            elif filt == 'aligned' and s.ints[u, c] < .050: continue
            cands.append((u, fi, st_vec(s, u, fi)))
            bases.append(base_fn(s, u, fi))
    if not cands: return
    with torch.no_grad():
        acts = net(torch.FloatTensor(np.array([c[2] for c in cands]))).argmax(1).numpy()
    if isinstance(net, BinaryDQNet):
        # Binary cache-or-skip: sort by CW base only; place action==1 first.
        for idx in np.argsort(-np.array(bases)):
            if acts[idx] == 1:
                u, fi, _ = cands[idx]
                s.put(u, fi)
    else:
        # 5-multiplier: rank by base * MULTS[argmax] and place in that order.
        for idx in np.argsort(-(np.array(bases) * MULTS[acts])):
            u, fi, _ = cands[idx]; s.put(u, fi)


# ──────────────────────────────────────────────────────────────
# 5b. TIMING: track deploy time per algorithm
# ──────────────────────────────────────────────────────────────
DEPLOY_TIMES = {}   # filled during Phase 2 evaluation

# ──────────────────────────────────────────────────────────────
# 6. EVALUATION HELPERS
# ──────────────────────────────────────────────────────────────
MAIN_SEEDS = [42, 43, 44]      # 3-seed performance tables
SIG_SEEDS  = list(range(42, 52))  # 10-seed significance




def main():
    """Run the full RL-CWCA experimental pipeline (phases 1–13)."""
    global FD, TD, LD, RUN_ID, RUN_DIR

    _RUN_CONFIG = {
        'seeds_main': [42, 43, 44],
        'seeds_sig':  list(range(42, 52)),
        'episodes':   100,
        'cache_1gb':  1000,
        'gamma':      0.6,
        'speed_fix':  'nospeed=th_gt_for_all (Rev3)',
        'aligned_threshold': 0.05,
    }
    _rm = RunManager()
    RUN_ID, RUN_DIR = _rm.new_run(config=_RUN_CONFIG)
    FD = str(RUN_DIR / 'figures')
    TD = str(RUN_DIR / 'tables')
    LD = str(RUN_DIR / 'logs')

    set_seed(GLOBAL_SEED)

    # ──────────────────────────────────────────────────────────────
    # PRE-PHASE: SELECT BEST REWARD WEIGHTS
    # ──────────────────────────────────────────────────────────────
    # In --phase13 mode the agents are already trained, so the pre-phase
    # (3 weight configs x 3 seeds x 100 A2C eps) is wasted compute. Skip
    # it and use the 2:1 config that was selected on prior runs; the only
    # downstream use of W_OFF/W_CHR in Phase 13 is the cold-start fine-tune
    # reward weighting, which is robust to this default.
    if _P13_ONLY:
        W_OFF, W_CHR = 200, 100
        print(f"[--phase13] Skipping pre-phase. Using W_OFF={W_OFF}, W_CHR={W_CHR}.")
    else:
        print("=" * 60)
        print("PRE-PHASE: Reward weight comparison (selects w_off, w_chr)")
        print("=" * 60)
        W_OFF, W_CHR = run_reward_weight_comparison()

    # ──────────────────────────────────────────────────────────────
    # PHASE 1: TRAIN ALL DRL AGENTS
    # ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"PHASE 1: Training DRL agents (100 eps, seed=42, "
          f"reward={W_OFF}×off+{W_CHR}×chr)")
    print("=" * 60)
    
    if _P13_ONLY:
        if not _CKPT.exists():
            print("  WARNING: --phase13 set but no checkpoint found.")
            print("  Run without --phase13 first to train and save agents.")
            import sys; sys.exit(1)
        agents, _meta = _load_checkpoint()
        # If the checkpoint stored the W_OFF / W_CHR that were dynamically
        # selected during the original training, use those instead of the
        # 200/100 fallback set up above.
        if _meta.get('w_off') is not None:
            W_OFF, W_CHR = _meta['w_off'], _meta['w_chr']
            print(f"  [--phase13] Restored W_OFF={W_OFF}, W_CHR={W_CHR} from checkpoint.")
        curves = {}
        print("  [SKIP] Phase 1 — loaded from checkpoint")
    else:
        # NOTE: 'w/o Speed Constraint' ablation removed.  Under SUMO contact times
        # (mean ≈ 516s), the size threshold s.gt ≫ max file size, so the filter
        # never binds and 'nospeed' is indistinguishable from 'full'.  The
        # constraint is retained in code for traces with shorter contacts but is
        # not a meaningful ablation in this regime.
        ABL_CONFIGS = {
            'w/o Interest Filter':  (b_cw_score, 'none',    'pri'),
            'w/o Priority Order':   (b_cw_score, 'full',    'rand'),
            'w/o CW Score (DQN+Pop)':(b_pop,    'full',    'pri'),
        }
        
        uord_map = {'pri': o_pri, 'rand': o_rand}
        agents  = {}
        curves  = {}
        
        TRAINER_MAP = {'a2c': train_a2c, 'ppo': train_ppo, 'dqn': train_dqn,
                       'binary_dqn': train_binary_dqn}
        
        for name, (bfn, filt, uo, ctrl) in tqdm(AGENT_CONFIGS.items(),
                                                  desc='Phase 1 agents', ncols=80):
            t0 = time.time()
            s  = Sim(250, .6, 1000, 80, 1, seed=GLOBAL_SEED)
            net, rews = TRAINER_MAP[ctrl](s, 100, bfn, filt, uord_map[uo],
                                           seed=GLOBAL_SEED, desc=f'{name} [{ctrl.upper()}]',
                                           w_off=W_OFF, w_chr=W_CHR)
            agents[name] = (net, bfn, filt, uord_map[uo])
            curves[name] = rews
            tqdm.write(f"  {name} [{ctrl.upper()}]: {time.time()-t0:.0f}s  off={max(rews)/300:.4f}")
        
        # Controller comparison: standalone PPO, A2C and DQN trained on the
        # same CW-score base, for a 3-way head-to-head against RL-CWCA's full
        # A2C pipeline. RL-CWCA's primary bar shares the A2C algorithm but
        # has the full feature stack (filter, priority order, dynamic reward
        # weights) -- the standalone A2C bar shows what plain A2C+CW gives.
        print("\nTraining standalone PPO, A2C and DQN (CW score, for controller comparison)...")
        for ctrl_name, trainer_fn in tqdm([('PPO', train_ppo),
                                            ('A2C', train_a2c),
                                            ('DQN', train_dqn)],
                                            desc='Phase 1 ctrl', ncols=80):
            t0 = time.time()
            s  = Sim(250, .6, 1000, 80, 1, seed=GLOBAL_SEED)
            net, rews = trainer_fn(s, 100, b_cw_score, 'full', o_pri,
                                    seed=GLOBAL_SEED, desc=ctrl_name,
                                    w_off=W_OFF, w_chr=W_CHR)
            agents[ctrl_name] = (net, b_cw_score, 'full', o_pri)
            curves[ctrl_name] = rews
            tqdm.write(f"  {ctrl_name}: {time.time()-t0:.0f}s")

        # RL-CWCA with aligned threshold (EXP 2) — uses A2C to match primary controller
        print("\nTraining RL-CWCA (aligned threshold 0.05)...")
        s = Sim(250, .6, 1000, 80, 1, seed=GLOBAL_SEED)
        net_aln, _ = train_a2c(s, 100, b_cw_score, 'aligned', o_pri,
                                 seed=GLOBAL_SEED, desc='RL-CWCA-aligned',
                                 w_off=W_OFF, w_chr=W_CHR)
        agents['RL-CWCA-aligned'] = (net_aln, b_cw_score, 'aligned', o_pri)

        # Ablation variants — use A2C to match RL-CWCA primary controller
        print("\nTraining ablation variants...")
        for name, (bfn, filt, uo) in tqdm(ABL_CONFIGS.items(),
                                            desc='Phase 1 ablation', ncols=80):
            s = Sim(250, .6, 1000, 80, 1, seed=GLOBAL_SEED)
            net, _ = train_a2c(s, 100, bfn, filt, uord_map[uo],
                                seed=GLOBAL_SEED, desc=name[:25],
                                w_off=W_OFF, w_chr=W_CHR)
            agents[name] = (net, bfn, filt, uord_map[uo])
            tqdm.write(f"  {name}: done")

        # Task 9: train an extra RL-CWCA variant with per-user reward
        # credit assignment. Same config as primary RL-CWCA except for
        # the reward_mode flag.
        print("\nTraining RL-CWCA (per-user reward) ablation variant...")
        s = Sim(250, .6, 1000, 80, 1, seed=GLOBAL_SEED)
        net_pu, _ = train_a2c(s, 100, b_cw_score, 'full', o_pri,
                               seed=GLOBAL_SEED, desc='RL-CWCA-PerUser',
                               w_off=W_OFF, w_chr=W_CHR,
                               reward_mode='per_user')
        agents['RL-CWCA-PerUser'] = (net_pu, b_cw_score, 'full', o_pri)

        _rm.mark_phase(RUN_DIR, 'phase1_training',
                       {'agents_trained': list(agents.keys())})

        _save_checkpoint(agents, w_off=W_OFF, w_chr=W_CHR)
    
    
    
    # ──────────────────────────────────────────────────────────────
    if not _P13_ONLY:
        # PHASE 2: MAIN EVALUATION  (3 same seeds)
        # ──────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("PHASE 2: Main evaluation (3 seeds)")
        print("=" * 60)
        
        MAIN_ALGOS = ['RL-CWCA', 'DRL-Binary',
                      'DQN-Interest', 'DQN-Weak', 'DQN-Pop',
                      'CFCA', 'SAA', 'Greedy', 'Popular Cache']
        HEURISTICS = {'CFCA': a_cfca, 'SAA': a_saa, 'Greedy': a_greedy, 'Popular Cache': a_pop}
        
        main_res = {}
        _p2 = tqdm(MAIN_ALGOS, desc='Phase 2 eval', ncols=80)
        for name in _p2:
            _p2.set_postfix(algo=name[:14])
            r = defaultdict(list)
            for sd in tqdm(MAIN_SEEDS, desc=f'  seeds', leave=False, ncols=60):
                s = Sim(250, .6, 1000, 80, 1, seed=sd)
                t0 = time.time()
                if name in HEURISTICS:
                    HEURISTICS[name](s)
                else:
                    net, bfn, filt, uo = agents[name]
                    deploy(s, net, bfn, filt, uo)
                DEPLOY_TIMES.setdefault(name, []).append(time.time() - t0)
                for k, v in s.exchange().items(): r[k].append(v)
            main_res[name] = r
            tqdm.write(f"  {name}: off={np.mean(r['off']):.4f}+/-{np.std(r['off']):.4f}")
        
        _rm.mark_phase(RUN_DIR, 'phase2_main_eval',
                       {nm: f"{np.mean(main_res[nm]['off']):.4f}" for nm in MAIN_ALGOS})
        
        
        # ──────────────────────────────────────────────────────────────
        # PHASE 3: CACHE SIZE SWEEP  (3 seeds)
        # ──────────────────────────────────────────────────────────────
        print("\n>>> Cache size sweep")
        CACHE_SIZES = [500, 750, 1000, 1500, 2000]
        # Include aligned variant in sweep for Table 8 (reviewer request: aligned at 2 GB)
        SWEEP_ALGOS = MAIN_ALGOS + ['RL-CWCA-aligned']
        cache_res = {a: defaultdict(lambda: defaultdict(list)) for a in SWEEP_ALGOS}
        _p3 = tqdm(CACHE_SIZES, desc='Phase 3 cache', ncols=80)
        for cm in _p3:
            _p3.set_postfix(cache=f'{cm}MB')
            for sd in tqdm(MAIN_SEEDS, desc=f'  seeds@{cm}MB', leave=False, ncols=60):
                for name in SWEEP_ALGOS:
                    s = Sim(250, .6, cm, 80, 1, seed=sd)
                    if name in HEURISTICS: HEURISTICS[name](s)
                    else:
                        net, bfn, filt, uo = agents[name]
                        deploy(s, net, bfn, filt, uo)
                    r = s.exchange()
                    for k, v in r.items(): cache_res[name][cm][k].append(v)
        
        _rm.mark_phase(RUN_DIR, 'phase3_cache_sweep',
                       {str(cm): {nm: f"{np.mean(cache_res[nm][cm]['off']):.4f}"
                                  for nm in ['RL-CWCA','CFCA','RL-CWCA-aligned']}
                        for cm in [1000, 2000]})
        
        
        # ──────────────────────────────────────────────────────────────
        # PHASE 4: ABLATION STUDY  (same 3 seeds as main eval)
        # ──────────────────────────────────────────────────────────────
        print("\n>>> Ablation study")
        abl_res = {}
        # Full RL-CWCA
        r = defaultdict(list)
        for sd in tqdm(MAIN_SEEDS, desc='  Full RL-CWCA', leave=False, ncols=60):
            s = Sim(250, .6, 1000, 80, 1, seed=sd)
            net, bfn, filt, uo = agents['RL-CWCA']
            deploy(s, net, bfn, filt, uo)
            for k, v in s.exchange().items(): r[k].append(v)
        abl_res['Full\nRL-CWCA'] = (np.mean(r['off']), np.std(r['off']))
        
        # w/o DQN = CFCA proxy
        r = defaultdict(list)
        for sd in tqdm(MAIN_SEEDS, desc='  w/o Controller', leave=False, ncols=60):
            s = Sim(250, .6, 1000, 80, 1, seed=sd); a_cfca(s)
            for k, v in s.exchange().items(): r[k].append(v)
        abl_res['w/o A2C\n(Gravity only)'] = (np.mean(r['off']), np.std(r['off']))
        
        # Ablation variants
        for name in tqdm(ABL_CONFIGS, desc='  Phase 4 ablation', ncols=80):
            r = defaultdict(list)
            for sd in tqdm(MAIN_SEEDS, desc=f'  {name[:20]}', leave=False, ncols=60):
                s = Sim(250, .6, 1000, 80, 1, seed=sd)
                net, bfn, filt, uo = agents[name]
                deploy(s, net, bfn, filt, uo)
                for k, v in s.exchange().items(): r[k].append(v)
            label = name.replace('w/o ', 'w/o\n')
            abl_res[label] = (np.mean(r['off']), np.std(r['off']))
            tqdm.write(f"  {name}: {np.mean(r['off']):.4f}+/-{np.std(r['off']):.4f}")
        
        # Aligned threshold (EXP 2 — same seeds)
        r = defaultdict(list)
        for sd in tqdm(MAIN_SEEDS, desc='  Aligned thresh', leave=False, ncols=60):
            s = Sim(250, .6, 1000, 80, 1, seed=sd)
            net, bfn, filt, uo = agents['RL-CWCA-aligned']
            deploy(s, net, bfn, filt, uo)
            for k, v in s.exchange().items(): r[k].append(v)
        abl_res['Aligned\nThreshold'] = (np.mean(r['off']), np.std(r['off']))
        tqdm.write(f"  Aligned threshold: {np.mean(r['off']):.4f}+/-{np.std(r['off']):.4f}")

        # Task 9 ablation: per-user reward variant (only if the agent was
        # trained in this run; --phase13 with an older checkpoint will skip).
        if 'RL-CWCA-PerUser' in agents:
            r = defaultdict(list)
            for sd in tqdm(MAIN_SEEDS, desc='  Per-user reward',
                           leave=False, ncols=60):
                s = Sim(250, .6, 1000, 80, 1, seed=sd)
                net, bfn, filt, uo = agents['RL-CWCA-PerUser']
                deploy(s, net, bfn, filt, uo)
                for k, v in s.exchange().items(): r[k].append(v)
            abl_res['Per-user\nreward'] = (np.mean(r['off']), np.std(r['off']))
            tqdm.write(f"  Per-user reward: {np.mean(r['off']):.4f}+/-{np.std(r['off']):.4f}")

        for k, (m, sd) in abl_res.items():
            tqdm.write(f"  {k.replace(chr(10),' ')}: {m:.4f}+/-{sd:.4f}")
        
        _rm.mark_phase(RUN_DIR, 'phase4_ablation',
                       {k.replace('\n',' '): f'{v[0]:.4f}' for k,v in abl_res.items()})
        
        
        # ──────────────────────────────────────────────────────────────
        # PHASE 5: SENSITIVITY SWEEPS
        # ──────────────────────────────────────────────────────────────
        print("\n>>> Sensitivity sweeps")
        SHOW = ['RL-CWCA', 'DRL-Binary', 'DQN-Interest', 'CFCA', 'SAA', 'Greedy', 'Popular Cache']

        # Zipf γ
        ZIPF_VALS = [0.6, 0.8, 1.0, 1.2]
        zipf_res = {a: defaultdict(list) for a in SHOW}
        for gm in tqdm(ZIPF_VALS, desc='  Zipf gamma sweep', ncols=80):
            for sd in MAIN_SEEDS:
                for name in SHOW:
                    s = Sim(250, gm, 1000, 80, 1, seed=sd)
                    if name in HEURISTICS: HEURISTICS[name](s)
                    else:
                        net, bfn, filt, uo = agents[name]; deploy(s, net, bfn, filt, uo)
                    zipf_res[name][gm].append(s.exchange()['off'])
        # ── Zipf CSV + MD ──
        _zipf_rows = []
        for gm in ZIPF_VALS:
            r = {'gamma': gm}
            for a in SHOW:
                vals = zipf_res[a][gm]
                r[f'{a}_off'] = round(float(np.mean(vals)), 4)
                r[f'{a}_off_std'] = round(float(np.std(vals)), 4)
            _zipf_rows.append(r)
        pd.DataFrame(_zipf_rows).to_csv(f'{TD}/table_sensitivity_zipf.csv', index=False)
        _z_md = ['| Zipf gamma | ' + ' | '.join(SHOW) + ' |',
                 '|---:|' + '---:|' * len(SHOW)]
        for r in _zipf_rows:
            _zc = [str(r['gamma'])]
            for a in SHOW:
                _zc.append(f'**{r[f"{a}_off"]:.4f}**' if a == 'RL-CWCA'
                           else f'{r[f"{a}_off"]:.4f}')
            _z_md.append('| ' + ' | '.join(_zc) + ' |')
        Path(f'{TD}/table_sensitivity_zipf.md').write_text('\n'.join(_z_md) + '\n', encoding='utf-8')
        print(f'  [OK] table_sensitivity_zipf.csv + .md')

        # File size
        FSIZES = [30, 70, 100, 150, 250]
        fsize_res = {a: [] for a in SHOW}
        _fsize_std = {a: [] for a in SHOW}
        for fm in tqdm(FSIZES, desc='  File size sweep', ncols=80):
            for name in SHOW:
                offs_fm = []
                for sd_fm in MAIN_SEEDS:
                    s = Sim(250, .6, 1000, fm, 1, seed=sd_fm)
                    if name in HEURISTICS: HEURISTICS[name](s)
                    else:
                        net, bfn, filt, uo = agents[name]; deploy(s, net, bfn, filt, uo)
                    offs_fm.append(s.exchange()['off'])
                fsize_res[name].append(float(np.mean(offs_fm)))
                _fsize_std[name].append(float(np.std(offs_fm)))
        # ── File size CSV + MD ──
        _fs_rows = []
        for i, fm in enumerate(FSIZES):
            r = {'fmb_MB': fm}
            for a in SHOW:
                r[f'{a}_off'] = round(fsize_res[a][i], 4)
                r[f'{a}_off_std'] = round(_fsize_std[a][i], 4)
            _fs_rows.append(r)
        pd.DataFrame(_fs_rows).to_csv(f'{TD}/table_sensitivity_fsize.csv', index=False)
        _fs_md = ['| Max File Size (MB) | ' + ' | '.join(SHOW) + ' |',
                  '|---:|' + '---:|' * len(SHOW)]
        for r in _fs_rows:
            _fsc = [str(r['fmb_MB'])]
            for a in SHOW:
                _fsc.append(f'**{r[f"{a}_off"]:.4f}**' if a == 'RL-CWCA'
                            else f'{r[f"{a}_off"]:.4f}')
            _fs_md.append('| ' + ' | '.join(_fsc) + ' |')
        Path(f'{TD}/table_sensitivity_fsize.md').write_text('\n'.join(_fs_md) + '\n', encoding='utf-8')
        print(f'  [OK] table_sensitivity_fsize.csv + .md')
        
        # Bit rate sweep (single seed, per original)
        BITRATES = [1, 2, 3, 4]
        brate_res = {a: [] for a in SHOW}
        for br_ in tqdm(BITRATES, desc='  Bit rate sweep', ncols=80):
            for name in SHOW:
                s = Sim(250, .6, 1000, 80, br_, seed=MAIN_SEEDS[0])
                if name in HEURISTICS: HEURISTICS[name](s)
                else:
                    net, bfn, filt, uo = agents[name]; deploy(s, net, bfn, filt, uo)
                brate_res[name].append(s.exchange()['off'])
        
        # HP sensitivity: training episodes
        print(">>> Training episodes HP sensitivity")
        EP_VALS = [20, 30, 40, 50, 60]
        hp_offs = []
        for n_ep in tqdm(EP_VALS, desc='  HP episodes', ncols=80):
            s = Sim(250, .6, 1000, 80, 1, seed=GLOBAL_SEED)
            net_hp, _ = train_dqn(s, n_ep, b_cw_score, 'full', o_pri,
                                    seed=GLOBAL_SEED, desc=f'HP ep={n_ep}',
                                    w_off=W_OFF, w_chr=W_CHR)
            s2 = Sim(250, .6, 1000, 80, 1, seed=GLOBAL_SEED)
            deploy(s2, net_hp, b_cw_score, 'full', o_pri)
            hp_offs.append(s2.exchange()['off'])
            tqdm.write(f"  {n_ep} eps: {hp_offs[-1]:.4f}")
        
        # NNPM: net profit margin  (Hassan et al., IEEE Access 2025, §III-D)
        # Revenue and cost are both expressed in MB so the subtraction is meaningful.
        #
        # revenue_MB  = total data offloaded via D2D (dd), already in MB.
        # cost_MB     = total data pre-loaded into caches during off-peak hours
        #               = cu_m (utilisation fraction 0-1)
        #                 × cm  (per-user cache capacity in MB, same for all users
        #                        because s.tc = np.full(N, float(cmb)))
        #                 × ND  (number of users, global constant)
        # This recovers the aggregate fill cost in MB without instantiating a new Sim.
        print(">>> NNPM computation")
        # DQN-Pop, Greedy, Popular Cache excluded: dd~=0 (local-hit-only algorithms),
        # so NNPM = (dd - cu*cm*N)/dd is numerically unstable (denominator -> 0).
        # DRL-Binary, DQN-Interest, DQN-Weak all drive D2D delivery via Sim.exchange()
        # and produce non-zero dd, so their NNPM is well-defined.
        _NNPM_ALGOS = ['RL-CWCA', 'DRL-Binary', 'DQN-Interest', 'DQN-Weak', 'SAA', 'CFCA']
        nnpm_res = {a: [] for a in _NNPM_ALGOS}
        for cm in tqdm(CACHE_SIZES, desc='  NNPM', ncols=80):
            for name in _NNPM_ALGOS:
                if name not in cache_res:
                    nnpm_res[name].append(0.0)
                    continue
                dd_m = np.mean(cache_res[name][cm]['dd'])   # MB delivered via D2D
                cu_m = np.mean(cache_res[name][cm]['cu'])   # cache utilisation (0-1)
                revenue_MB = dd_m
                cost_MB    = cu_m * cm * 250                # fill cost in MB (N=250 sim users)
                nnpm_res[name].append(
                    (revenue_MB - cost_MB) / max(revenue_MB, 1e-9) * 100
                )

        # ── Auto-emit combined Off + NNPM table (CSV / MD / TeX) ──
        # Same schema as gen_off_nnpm_table.py so the artifact is identical
        # whether produced in-pipeline or post-hoc. Lands in Phase 12's table
        # dir so all paper tables sit together.
        _ofn_rows = []
        for ci, cm in enumerate(CACHE_SIZES):
            r = {'cache_MB': cm}
            for name in _NNPM_ALGOS:
                if name in cache_res:
                    r[f'{name}_off']     = float(np.mean(cache_res[name][cm]['off']))
                    r[f'{name}_off_std'] = float(np.std (cache_res[name][cm]['off']))
                else:
                    r[f'{name}_off'] = 0.0; r[f'{name}_off_std'] = 0.0
                r[f'{name}_nnpm'] = float(nnpm_res[name][ci])
            _ofn_rows.append(r)
        pd.DataFrame(_ofn_rows).to_csv(
            f'{TD}/table_off_nnpm_combined.csv', index=False)
        # Markdown preview
        _md_hdr = '| Cache (MB) ' + ''.join(f'| {a} Off | {a} NNPM ' for a in _NNPM_ALGOS) + '|'
        _md_div = '|-----------:' + ''.join('|----:|-----:' for _ in _NNPM_ALGOS) + '|'
        _md = [_md_hdr, _md_div]
        for r in _ofn_rows:
            _parts = [f' {r["cache_MB"]} ']
            for a in _NNPM_ALGOS:
                _bold = (a == 'RL-CWCA')
                _o = f'{r[f"{a}_off"]:.3f}';  _n = f'{r[f"{a}_nnpm"]:+.1f}%'
                if _bold:
                    _parts.append(f'| **{_o}** | **{_n}** ')
                else:
                    _parts.append(f'| {_o} | {_n} ')
            _md.append('|' + ''.join(_parts) + '|')
        Path(f'{TD}/table_off_nnpm_combined.md').write_text('\n'.join(_md) + '\n', encoding='utf-8')
        print(f"  [OK] table_off_nnpm_combined.csv + .md")

        # ── Full 9-algorithm NNPM-only table ──
        # DQN-Pop and Popular Cache have dd~=0; their NNPM is marked undefined.
        _NNPM_ALL   = ['RL-CWCA', 'DRL-Binary', 'DQN-Interest', 'DQN-Weak', 'DQN-Pop',
                       'CFCA', 'SAA', 'Greedy', 'Popular Cache']
        _DD_THRESH  = 1.0   # MB; below this treat NNPM as undefined
        _full_nnpm  = {}    # algo -> list of (nnpm_val, is_unstable) per cache size
        for name in _NNPM_ALL:
            _full_nnpm[name] = []
            for cm in CACHE_SIZES:
                if name not in cache_res:
                    _full_nnpm[name].append((float('nan'), True)); continue
                dd_m = float(np.mean(cache_res[name][cm]['dd']))
                cu_m = float(np.mean(cache_res[name][cm]['cu']))
                unstable = dd_m < _DD_THRESH
                val = (dd_m - cu_m * cm * 250) / max(dd_m, 1e-9) * 100
                _full_nnpm[name].append((val, unstable))
        # CSV (includes raw dd for audit)
        _fn_rows = []
        for ci, cm in enumerate(CACHE_SIZES):
            r = {'cache_MB': cm}
            for a in _NNPM_ALL:
                v, unstable = _full_nnpm[a][ci]
                r[f'{a}_nnpm'] = round(v, 2) if not unstable else 'UNDEF'
                r[f'{a}_unstable'] = int(unstable)
            _fn_rows.append(r)
        pd.DataFrame(_fn_rows).to_csv(f'{TD}/table_nnpm_full.csv', index=False)
        # Markdown
        _fn_md = ['| Cache (MB) | ' + ' | '.join(_NNPM_ALL) + ' |',
                  '|---:|' + '---:|' * len(_NNPM_ALL)]
        for r in _fn_rows:
            _fnc = [str(r['cache_MB'])]
            for a in _NNPM_ALL:
                v, unstable = _full_nnpm[a][_fn_rows.index(r)]
                if unstable:
                    _fnc.append('— †')
                elif a == 'RL-CWCA':
                    _fnc.append(f'**{v:+.1f}%**')
                else:
                    _fnc.append(f'{v:+.1f}%')
            _fn_md.append('| ' + ' | '.join(_fnc) + ' |')
        _fn_md.append('')
        _fn_md.append('† dd ~= 0; NNPM numerically unstable (denominator -> 0).')
        Path(f'{TD}/table_nnpm_full.md').write_text('\n'.join(_fn_md) + '\n', encoding='utf-8')
        print(f"  [OK] table_nnpm_full.csv + .md  (9 algorithms)")

        _rm.mark_phase(RUN_DIR, 'phase5_sensitivity', {'status': 'done'})
        
        
        # ──────────────────────────────────────────────────────────────
        # PHASE 6: 10-SEED SIGNIFICANCE  (EXP 3)
        # ──────────────────────────────────────────────────────────────
        print("\n>>> 10-seed significance test")
        sig_res = {a: [] for a in ['RL-CWCA', 'CFCA', 'SAA', 'Greedy', 'Popular Cache']}
        for sd in tqdm(SIG_SEEDS, desc='Phase 6 sig seeds', ncols=80):
            for name in sig_res:
                s = Sim(250, .6, 1000, 80, 1, seed=sd)
                if name in HEURISTICS: HEURISTICS[name](s)
                else:
                    net, bfn, filt, uo = agents[name]; deploy(s, net, bfn, filt, uo)
                sig_res[name].append(s.exchange()['off'])
        
        SIG_NAMES = list(sig_res.keys())
        n_sig = len(SIG_NAMES)
        SIG_P = np.ones((n_sig, n_sig))
        for i in range(n_sig):
            for j in range(n_sig):
                if i != j:
                    _, p = stats.mannwhitneyu(sig_res[SIG_NAMES[i]], sig_res[SIG_NAMES[j]],
                                              alternative='two-sided')
                    SIG_P[i, j] = p
        print("  Significance matrix done")
        _rm.mark_phase(RUN_DIR, 'phase6_significance', {'status': 'done'})
        
        
        # ──────────────────────────────────────────────────────────────
        # PHASE 7: CONTENTION SWEEP  (EXP 4)
        # ──────────────────────────────────────────────────────────────
        print("\n>>> Contention sensitivity")
        COLL_FACTORS = [1.00, 0.85, 0.70, 0.55, 0.40]
        coll_res = {a: [] for a in ['RL-CWCA', 'CFCA', 'SAA']}
        for cf in tqdm(COLL_FACTORS, desc='Phase 7 contention', ncols=80):
            for name in coll_res:
                offs_cf = []
                for sd_cf in tqdm(MAIN_SEEDS, desc=f'  ρ={cf}', leave=False, ncols=60):
                    s = Sim(250, .6, 1000, 80, 1, seed=sd_cf, coll_factor=cf)
                    if name in HEURISTICS: HEURISTICS[name](s)
                    else:
                        net, bfn, filt, uo = agents[name]; deploy(s, net, bfn, filt, uo)
                    offs_cf.append(s.exchange()['off'])
                coll_res[name].append(float(np.mean(offs_cf)))
        _rm.mark_phase(RUN_DIR, 'phase7_contention', {'status': 'done'})
        
        
        # ──────────────────────────────────────────────────────────────
        # PHASE 8: COVERAGE & FAIRNESS  (3 seeds)
        # ──────────────────────────────────────────────────────────────
        print("\n>>> Coverage & fairness")
        COV_ALGOS = ['RL-CWCA', 'SAA', 'CFCA', 'Greedy', 'Popular Cache']
        cov_res = {a: defaultdict(list) for a in COV_ALGOS}
        for sd in tqdm(MAIN_SEEDS, desc='Phase 8 coverage', ncols=80):
            for name in COV_ALGOS:
                s = Sim(250, .6, 1000, 80, 1, seed=sd)
                if name in HEURISTICS: HEURISTICS[name](s)
                else:
                    net, bfn, filt, uo = agents[name]; deploy(s, net, bfn, filt, uo)
                r = s.exchange()
                cov_res[name]['jain'].append(r['jain'])
                cov_res[name]['cov'].append(r['cov'])
                cov_res[name]['off'].append(r['off'])
        _rm.mark_phase(RUN_DIR, 'phase8_coverage', {'status': 'done'})
        
        
        # ──────────────────────────────────────────────────────────────
        # PHASE 9: SCALABILITY  (file library size)
        # ──────────────────────────────────────────────────────────────
        print("\n>>> Scalability")
        FLIB = [100, 150, 200, 300, 400]
        _FLIB_ALGOS = ['RL-CWCA', 'DRL-Binary', 'SAA', 'CFCA']
        _FLIB_HEUR  = {'CFCA': a_cfca, 'SAA': a_saa}
        flib_res  = {a: [] for a in _FLIB_ALGOS}
        _flib_std = {a: [] for a in _FLIB_ALGOS}
        for fl in tqdm(FLIB, desc='Phase 9 scalability', ncols=80):
            for name in _FLIB_ALGOS:
                offs_fl = []
                for sd_fl in MAIN_SEEDS:
                    s = Sim(fl, .6, 1000, 80, 1, seed=sd_fl)
                    if name in _FLIB_HEUR: _FLIB_HEUR[name](s)
                    else:
                        net, bfn, filt, uo = agents[name]; deploy(s, net, bfn, filt, uo)
                    offs_fl.append(s.exchange()['off'])
                flib_res[name].append(float(np.mean(offs_fl)))
                _flib_std[name].append(float(np.std(offs_fl)))
        # ── Scalability CSV + MD ──
        _fl_rows = []
        for i, fl in enumerate(FLIB):
            r = {'F': fl}
            for a in _FLIB_ALGOS:
                r[f'{a}_off'] = round(flib_res[a][i], 4)
                r[f'{a}_off_std'] = round(_flib_std[a][i], 4)
            r['gap_pct_vs_cfca'] = round(
                (flib_res['RL-CWCA'][i] - flib_res['CFCA'][i]) / flib_res['CFCA'][i] * 100, 2)
            _fl_rows.append(r)
        pd.DataFrame(_fl_rows).to_csv(f'{TD}/table_scalability_flib.csv', index=False)
        _fl_md = ['| F | ' + ' | '.join(_FLIB_ALGOS) + ' | RL-CWCA vs CFCA |',
                  '|---:|' + '---:|' * len(_FLIB_ALGOS) + '---:|']
        for r in _fl_rows:
            _flc = [str(r['F'])]
            for a in _FLIB_ALGOS:
                _flc.append(f'**{r[f"{a}_off"]:.4f}**' if a == 'RL-CWCA'
                            else f'{r[f"{a}_off"]:.4f}')
            _flc.append(f'+{r["gap_pct_vs_cfca"]:.1f}%')
            _fl_md.append('| ' + ' | '.join(_flc) + ' |')
        Path(f'{TD}/table_scalability_flib.md').write_text('\n'.join(_fl_md) + '\n', encoding='utf-8')
        print(f'  [OK] table_scalability_flib.csv + .md')
        _rm.mark_phase(RUN_DIR, 'phase9_scalability', {'status': 'done'})
        
        
        # ──────────────────────────────────────────────────────────────
        # PHASE 10: CONTROLLER COMPARISON  (same 3 seeds as main eval)
        # ──────────────────────────────────────────────────────────────
        print("\n>>> Controller comparison (same seeds)")
        ctrl_res = {}
        _p10 = tqdm([1000, 2000], desc='Phase 10 controller', ncols=80)
        for cm in _p10:
            _p10.set_postfix(cache=f'{cm}MB')
            ctrl_res[cm] = {}
            for name in tqdm(['RL-CWCA', 'PPO', 'A2C', 'DQN', 'CFCA'],
                             desc='  controllers', leave=False, ncols=60):
                offs = []
                for sd in MAIN_SEEDS:
                    s = Sim(250, .6, cm, 80, 1, seed=sd)
                    if name == 'CFCA': a_cfca(s)
                    else:
                        net, bfn, filt, uo = agents[name]; deploy(s, net, bfn, filt, uo)
                    offs.append(s.exchange()['off'])
                ctrl_res[cm][name] = (np.mean(offs), np.std(offs))
                tqdm.write(f"  {name} @ {cm}MB: {np.mean(offs):.4f}+/-{np.std(offs):.4f}")
        _rm.mark_phase(RUN_DIR, 'phase10_controller', {'status': 'done'})
        
        
        # ──────────────────────────────────────────────────────────────
        # PHASE 11: GENERATE ALL FIGURES
        # ──────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("PHASE 11: Generating all figures")
        print("=" * 60)
        _fig_pbar = tqdm(total=21, desc='Phase 11 figures', ncols=80, unit='fig')
        
        # ── fig01: Training curves (3 panels: reward / stability / offloading) ──
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.8))
        w = 5
        drl_agents = ['RL-CWCA', 'DQN', 'DQN-Interest', 'DQN-Weak', 'DQN-Pop']
        for nm in drl_agents:
            rw = curves[nm]
            sm = np.convolve(rw, np.ones(w)/w, 'valid')
            ax1.plot(range(w-1, len(rw)), sm, label=nm,
                     color=S[nm]['color'], lw=S[nm]['lw'], ls=S[nm]['ls'], zorder=S[nm]['z'])
        ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward (w=5)')
        ax1.set_title('DRL Training Convergence')
        # No per-axis legend on ax1 -- a single shared legend lives at the
        # bottom of the figure (see fig.legend(...) below).
        rw = curves['RL-CWCA']
        std_c = [np.std(rw[max(0, i-9):i+1]) for i in range(len(rw))]
        ax2.plot(std_c, color=S['RL-CWCA']['color'], lw=2.2)
        ax2.fill_between(range(len(std_c)), std_c, alpha=0.18, color=S['RL-CWCA']['color'])
        ax2.set_xlabel('Episode'); ax2.set_ylabel('Std Dev (w=10)')
        ax2.set_title('RL-CWCA Training Stability')
        # Panel 3 (Task 12): per-episode offloading ratio. Pulls from _TRAINING_PHIS
        # keyed by training `desc`; falls back silently if a curve is missing.
        _PHI_KEY = {'RL-CWCA': 'RL-CWCA [A2C]', 'DQN': 'DQN',
                    'DQN-Interest': 'DQN-Interest [DQN]',
                    'DQN-Weak': 'DQN-Weak [DQN]', 'DQN-Pop': 'DQN-Pop [DQN]'}
        _cfca_handle = None
        for nm in drl_agents:
            phis = _TRAINING_PHIS.get(_PHI_KEY.get(nm, nm), [])
            if not phis: continue
            sm_phi = np.convolve(phis, np.ones(w)/w, 'valid')
            # Note: NOT passing label= here -- ax1 already declared each curve's
            # label, and we use ax1's handles for the shared legend below.
            ax3.plot(range(w-1, len(phis)), sm_phi,
                     color=S[nm]['color'], lw=S[nm]['lw'], ls=S[nm]['ls'],
                     zorder=S[nm]['z'])
        # CFCA deployment phi at 1 GB from main_res (Phase 2)
        try:
            _cfca_phi = float(np.mean(main_res['CFCA']['off']))
            _cfca_line = ax3.axhline(_cfca_phi, color='gray', ls=':', lw=1.5)
            _cfca_handle = (_cfca_line, f'CFCA deploy ($\\phi$ = {_cfca_phi:.3f})')
        except (KeyError, NameError):
            pass
        ax3.set_xlabel('Episode'); ax3.set_ylabel('Offloading $\\phi$ (w=5)')
        ax3.set_title('Per-Episode Offloading Convergence')
        # Pad y so curves never crowd the top edge; legend lives at the bottom
        # of the whole figure, so no panel-level legend is needed.
        _y_lo, _y_hi = ax3.get_ylim()
        ax3.set_ylim(_y_lo, _y_hi + 0.05 * (_y_hi - _y_lo))

        # ── Single shared legend at the bottom of the figure ─────────────
        _h, _l = ax1.get_legend_handles_labels()
        if _cfca_handle is not None:
            _h = _h + [_cfca_handle[0]]; _l = _l + [_cfca_handle[1]]
        fig.legend(_h, _l, loc='lower center', ncol=len(_l),
                   bbox_to_anchor=(0.5, -0.02), frameon=True,
                   fontsize=12, columnspacing=1.4, handlelength=2.6)
        # Reserve ~13% of the figure height at the bottom for the legend
        # so it doesn't overlap any of the three panel x-axes.
        plt.tight_layout(rect=[0, 0.13, 1, 1])
        savefig(fig, 'fig01_training_curves')
        _fig_pbar.update(1)
        
        # ── fig02: Offloading vs Cache Size ──
        # NOTE: DRL_ORDER is the row order for table2_cache_sweep.csv (Phase 12)
        # and the legend order for fig02/fig03. Keep DRL agents first so the
        # paper's main row order stays consistent.
        DRL_ORDER = ['RL-CWCA', 'DRL-Binary', 'DQN-Interest', 'DQN-Weak',
                     'DQN-Pop', 'CFCA', 'SAA']
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for nm in DRL_ORDER:
            m = [np.mean(cache_res[nm][c]['off']) for c in CACHE_SIZES]
            sd= [np.std(cache_res[nm][c]['off'])  for c in CACHE_SIZES]
            ax.errorbar(CACHE_SIZES, m, yerr=sd, label=nm, **ep(nm))
        ax.set_xlabel('Cache Size (MB)'); ax.set_ylabel('Offloading Ratio')
        ax.set_title('Offloading Ratio vs Cache Size')
        ax.set_xticks(CACHE_SIZES[::1])
        ax.legend(ncol=2, loc='lower right')
        plt.tight_layout(); savefig(fig, 'fig02_off_vs_cache')
        _fig_pbar.update(1)
        
        # ── fig03: CHR vs Cache Size ──
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for nm in DRL_ORDER:
            m = [np.mean(cache_res[nm][c]['chr']) for c in CACHE_SIZES]
            sd= [np.std(cache_res[nm][c]['chr'])  for c in CACHE_SIZES]
            ax.errorbar(CACHE_SIZES, m, yerr=sd, label=nm, **ep(nm))
        ax.set_xlabel('Cache Size (MB)'); ax.set_ylabel('Cache Hit Ratio (CHR)')
        ax.set_title('Cache Hit Ratio vs Cache Size')
        ax.legend(ncol=2, loc='lower right')
        plt.tight_layout(); savefig(fig, 'fig03_chr_vs_cache')
        _fig_pbar.update(1)
        
        # ── fig04: Improvement bars ──
        RL_OFF = np.mean(main_res['RL-CWCA']['off'])
        RL_CHR = np.mean(main_res['RL-CWCA']['chr'])
        HEUR_NAMES = ['CFCA', 'SAA', 'Greedy', 'Popular Cache']
        imp_o = [(RL_OFF - np.mean(main_res[b]['off'])) / np.mean(main_res[b]['off']) * 100
                 for b in HEUR_NAMES]
        imp_c = [(RL_CHR - np.mean(main_res[b]['chr'])) / np.mean(main_res[b]['chr']) * 100
                 for b in HEUR_NAMES]
        x = np.arange(len(HEUR_NAMES)); bw = 0.36
        fig, ax = plt.subplots(figsize=(8, 5.2))
        b1 = ax.bar(x-bw/2, imp_o, bw, label='Offloading ↑',
                    color='#E63946', edgecolor='black', lw=0.7, alpha=0.88)
        b2 = ax.bar(x+bw/2, imp_c, bw, label='CHR ↑',
                    color='#1D6FA4', edgecolor='black', lw=0.7, alpha=0.88)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.35,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(HEUR_NAMES, rotation=12, ha='right')
        ax.set_ylabel('Improvement (%)'); ax.legend()
        ax.set_title('RL-CWCA Improvement over Baselines at 1 GB Cache')
        ax.set_ylim(0, max(imp_o+imp_c)*1.22)
        plt.tight_layout(); savefig(fig, 'fig04_improvement_bars')
        _fig_pbar.update(1)
        
        # ── fig05: Zipf γ sensitivity ──
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for nm in SHOW:
            m = [np.mean(zipf_res[nm][g]) for g in ZIPF_VALS]
            sd= [np.std(zipf_res[nm][g])  for g in ZIPF_VALS]
            ax.errorbar(ZIPF_VALS, m, yerr=sd, label=nm, **ep(nm))
        ax.set_xlabel('Zipf Parameter (γ)'); ax.set_ylabel('Offloading Ratio')
        ax.set_title('Offloading Ratio vs File Popularity (Zipf γ)')
        ax.legend(ncol=2, loc='upper left')
        plt.tight_layout(); savefig(fig, 'fig05_zipf_sensitivity')
        _fig_pbar.update(1)
        
        # ── fig06: File size sensitivity ──
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for nm in SHOW:
            ax.plot(FSIZES, fsize_res[nm], label=nm, **lp(nm))
        ax.set_xlabel('Average File Size (MB)'); ax.set_ylabel('Offloading Ratio')
        ax.set_title('Offloading Ratio vs File Size')
        ax.legend(ncol=2)
        plt.tight_layout(); savefig(fig, 'fig06_fsize_sensitivity')
        _fig_pbar.update(1)
        
        # ── fig07: Ablation ──
        albl = list(abl_res.keys())
        amn  = [abl_res[k][0] for k in albl]
        asd  = [abl_res[k][1] for k in albl]
        full_val = amn[0]
        acols = ['#E63946'] + ['#9467BD']*(len(albl)-2) + ['#FF7F0E']
        fig, ax = plt.subplots(figsize=(12, 5.2))
        bars = ax.bar(albl, amn, yerr=asd, color=acols, edgecolor='black',
                      lw=0.8, alpha=0.88, capsize=4)
        bars[0].set_linewidth(2.5)
        ax.axhline(full_val, color='#E63946', ls='--', lw=1.6, alpha=0.65, label='Full RL-CWCA')
        for bar, m, s_ in zip(bars, amn, asd):
            ax.text(bar.get_x()+bar.get_width()/2, m+max(asd)*0.5+0.002,
                    f'{m:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_ylabel('Offloading Ratio')
        ax.set_title('Ablation Study: Component Contribution')
        ax.legend(); ax.set_ylim(min(amn)*0.87, max(amn)*1.10)
        plt.xticks(rotation=10, ha='right')
        plt.tight_layout(); savefig(fig, 'fig07_ablation')
        _fig_pbar.update(1)
        
        # ── fig08: Significance matrix ──
        from matplotlib.colors import LinearSegmentedColormap
        _navy_cmap = LinearSegmentedColormap.from_list('white_navy', ['#ffffff', '#0a2756'], N=256)
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        im = ax.imshow(SIG_P, cmap=_navy_cmap, vmin=0, vmax=0.12,
                       interpolation='nearest', aspect='auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('p-value', fontsize=15, fontweight='bold')
        cbar.ax.tick_params(labelsize=13)
        ax.set_xticks(range(n_sig)); ax.set_yticks(range(n_sig))
        ax.set_xticklabels(SIG_NAMES, rotation=40, ha='right', fontsize=13, fontweight='bold')
        ax.set_yticklabels(SIG_NAMES, fontsize=13, fontweight='bold')
        for i in range(n_sig):
            for j in range(n_sig):
                p = SIG_P[i, j]
                sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
                tc = 'white' if p > 0.06 else 'black'
                ax.text(j, i, f'{p:.3f}\n{sig}', ha='center', va='center',
                        fontsize=13, color=tc, fontweight='bold')
        ax.set_title('Pairwise Statistical Significance (Mann-Whitney U Test)', pad=10)
        plt.tight_layout(); savefig(fig, 'fig08_significance')
        _fig_pbar.update(1)
        
        # ── fig09: Scalability ──
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for nm in ['RL-CWCA', 'CFCA']:
            ax.plot(FLIB, flib_res[nm], label=nm, **lp(nm))
        ax.fill_between(FLIB, flib_res['CFCA'], flib_res['RL-CWCA'],
                        alpha=0.13, color='#E63946', label='Advantage gap')
        ax.set_xlabel('File Library Size (F)'); ax.set_ylabel('Offloading Ratio')
        ax.set_title('Scalability: Offloading Ratio vs File Library Size')
        ax.legend()
        plt.tight_layout(); savefig(fig, 'fig09_scalability')
        _fig_pbar.update(1)
        
        # ── fig10: Controller comparison ──
        # RL-CWCA is A2C-based in v6. The three standalone bars (PPO / A2C /
        # DQN) all use the same CW-score base and train for the same number
        # of episodes -- they isolate the controller choice from the rest of
        # the RL-CWCA pipeline (full-feature filter, priority order, dynamic
        # reward weights).
        cnames = ['A2C\n(RL-CWCA)', 'PPO\n(CW only)', 'A2C\n(CW only)',
                  'DQN\n(CW score)', 'CFCA']
        cmap_  = {'A2C\n(RL-CWCA)': 'RL-CWCA', 'PPO\n(CW only)': 'PPO',
                  'A2C\n(CW only)': 'A2C', 'DQN\n(CW score)': 'DQN',
                  'CFCA': 'CFCA'}
        x = np.arange(len(cnames)); bw = 0.36
        m1 = [ctrl_res[1000][cmap_[k]][0] for k in cnames]
        s1 = [ctrl_res[1000][cmap_[k]][1] for k in cnames]
        m2 = [ctrl_res[2000][cmap_[k]][0] for k in cnames]
        s2 = [ctrl_res[2000][cmap_[k]][1] for k in cnames]
        # Colours: RL-CWCA / PPO / A2C / DQN / CFCA  (S[...] palette aliases)
        ccols = ['#E63946', '#BCBD22', '#17BECF', '#8B4513', '#1D6FA4']
        fig, ax = plt.subplots(figsize=(9, 5.2))
        b1 = ax.bar(x-bw/2, m1, bw, yerr=s1, label='1 GB',
                    color=ccols, edgecolor='black', lw=0.7, alpha=0.90, capsize=4)
        b2 = ax.bar(x+bw/2, m2, bw, yerr=s2, label='2 GB',
                    color=ccols, edgecolor='black', lw=0.7, alpha=0.50, capsize=4, hatch='//')
        for bar, m in zip(list(b1)+list(b2), m1+m2):
            ax.text(bar.get_x()+bar.get_width()/2, m+0.005,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=11.5)
        ax.set_xticks(x); ax.set_xticklabels(cnames)
        ax.set_ylabel('Offloading Ratio')
        ax.set_title('Controller Comparison: RL-CWCA (A2C, full) vs PPO vs A2C vs DQN')
        ax.legend()
        plt.tight_layout(); savefig(fig, 'fig10_controller_compare')
        _fig_pbar.update(1)
        
        # ── fig11: Contention ──
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for nm in ['RL-CWCA', 'CFCA', 'SAA']:
            ax.plot(COLL_FACTORS, coll_res[nm], label=nm, **lp(nm))
        ax.invert_xaxis()
        ax.axvline(1.0, color='gray', ls=':', lw=1.2, alpha=0.6)
        ax.set_xlabel('Collision Factor ρ  (← no contention        heavy contention →)')
        ax.set_ylabel('Offloading Ratio')
        ax.set_title('Offloading Ratio under Varying D2D Contention')
        ax.legend()
        plt.tight_layout(); savefig(fig, 'fig11_contention')
        _fig_pbar.update(1)
        
        # ── fig12: Coverage & Fairness ──
        x = np.arange(len(COV_ALGOS)); bw = 0.36
        COVERAGE  = [np.mean(cov_res[a]['cov'])*100 for a in COV_ALGOS]
        JAIN      = [np.mean(cov_res[a]['jain'])     for a in COV_ALGOS]
        ccov = [S[a]['color'] for a in COV_ALGOS]
        fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
        ax2 = ax1.twinx()
        b1 = ax1.bar(x-bw/2, COVERAGE, bw, label='Coverage (%)',
                     color=ccov, edgecolor='black', lw=0.7, alpha=0.88)
        b2 = ax2.bar(x+bw/2, JAIN, bw, label="Jain's Index",
                     color=ccov, edgecolor='black', lw=0.7, alpha=0.52, hatch='//')
        for bar, v in zip(b1, COVERAGE):
            ax1.text(bar.get_x()+bar.get_width()/2, v+0.5,
                     f'{v:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        for bar, v in zip(b2, JAIN):
            ax2.text(bar.get_x()+bar.get_width()/2, v+0.002,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax1.set_xticks(x); ax1.set_xticklabels(COV_ALGOS, rotation=12, ha='right')
        ax1.set_ylabel('Request Coverage (%)'); ax2.set_ylabel("Jain's Fairness Index")
        ax1.set_ylim(0, max(COVERAGE)*1.25); ax2.set_ylim(0.75, max(JAIN)*1.05)
        ax1.set_title('Coverage and Fairness across Algorithms')
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='lower center',
                   bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True)
        plt.tight_layout(); savefig(fig, 'fig12_coverage_fairness')
        _fig_pbar.update(1)
        
        # ── fig13: Violin ──
        vnames = ['RL-CWCA', 'CFCA', 'SAA', 'Greedy', 'Popular Cache']
        vdata  = [sig_res[k] for k in vnames]   # 10 seeds each
        vcols  = [S[k]['color'] for k in vnames]
        fig, ax = plt.subplots(figsize=(9, 5.0))
        parts = ax.violinplot(vdata, positions=range(len(vnames)),
                              showmeans=True, showextrema=True, widths=0.65)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(vcols[i]); pc.set_alpha(0.60)
        for comp in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
            parts[comp].set_linewidth(1.6)
        for i, (nm, vals) in enumerate(zip(vnames, vdata)):
            ax.scatter([i]*len(vals), vals, color=vcols[i],
                       s=28, zorder=5, alpha=0.75, edgecolors='white', lw=0.4)
        ax.set_xticks(range(len(vnames))); ax.set_xticklabels(vnames, rotation=12, ha='right')
        ax.set_ylabel('Offloading Ratio')
        ax.set_title('Offloading Ratio Distribution across Seeds')
        plt.tight_layout(); savefig(fig, 'fig13_violin')
        _fig_pbar.update(1)
        
        # ── fig14: SUMO mobility characterisation ──
        print('>>> fig14_sumo_mobility')
        all_ct = ACT[ACT > 0].flatten()
        sorted_ct = np.sort(all_ct)
        cdf = np.arange(1, len(sorted_ct)+1) / len(sorted_ct)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(sorted_ct, cdf, color=S['RL-CWCA']['color'], lw=2.2)
        ax1.axvline(np.mean(all_ct), color='gray', ls='--', lw=1.5,
                    label=f'Mean={np.mean(all_ct):.0f}s')
        ax1.axvline(np.median(all_ct), color='#9467BD', ls=':', lw=1.5,
                    label=f'Median={np.median(all_ct):.0f}s')
        ax1.set_xlabel('Contact Duration (s)'); ax1.set_ylabel('CDF')
        ax1.set_title('Contact Duration CDF'); ax1.legend()
        sample = min(50, ND)
        im = ax2.imshow(NCT[:sample, :sample], cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, ax=ax2, label='Contact Count')
        ax2.set_xlabel('Device ID'); ax2.set_ylabel('Device ID')
        ax2.set_title(f'Contact Frequency Heatmap (top {sample} devices)')
        plt.tight_layout(); savefig(fig, 'fig14_sumo_mobility')
        _fig_pbar.update(1)
        
        # ── fig15: Bit rate sweep ──
        print('>>> fig15_off_bitrate')
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for nm in SHOW:
            ax.plot(BITRATES, brate_res[nm], label=nm, **lp(nm))
        ax.set_xlabel('Bit Rate (MB/s)'); ax.set_ylabel('Offloading Ratio')
        ax.set_title('Offloading Ratio vs D2D Bit Rate')
        ax.legend(ncol=2); plt.tight_layout(); savefig(fig, 'fig15_off_bitrate')
        _fig_pbar.update(1)
        
        # ── fig16: 5-metric radar ──
        print('>>> fig16_radar')
        RADAR_ALGOS = ['RL-CWCA', 'CFCA', 'SAA', 'Greedy', 'Popular Cache']
        METRICS_R = ['Offloading', 'CHR', 'Cache\nUtil.', 'D2D\nSuccess', 'Local\nHit Rate']
        radar_vals = {}
        for a in RADAR_ALGOS:
            radar_vals[a] = [
                np.mean(main_res[a]['off']),
                np.mean(main_res[a]['chr']),
                np.mean(main_res[a]['cu']),
                np.mean(main_res[a]['d2d']),
                np.mean(main_res[a]['lr']),
            ]
        Nm = len(METRICS_R)
        angles = np.linspace(0, 2*np.pi, Nm, endpoint=False).tolist(); angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for a in RADAR_ALGOS:
            vals = radar_vals[a] + radar_vals[a][:1]
            ax.plot(angles, vals, 'o-', lw=2, color=S[a]['color'], label=a, ms=S[a]['ms']*0.7)
            ax.fill(angles, vals, alpha=0.08, color=S[a]['color'])
        ax.set_thetagrids(np.degrees(angles[:-1]), METRICS_R)
        ax.set_ylim(0, 1); ax.set_title('Multi-Metric Comparison', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.12))
        plt.tight_layout(); savefig(fig, 'fig16_radar')
        _fig_pbar.update(1)
        
        # ── fig17: Violin — both Offloading and CHR ──
        print('>>> fig17_violin_both')
        vnames_h = ['RL-CWCA', 'CFCA', 'SAA', 'Greedy', 'Popular Cache']
        vcols_h  = [S[k]['color'] for k in vnames_h]
        fig, (av1, av2) = plt.subplots(1, 2, figsize=(14, 5))
        for ax, metric, title in [(av1, 'off', 'Offloading Ratio'), (av2, 'chr', 'Cache Hit Ratio')]:
            vdata_h = [main_res[a][metric] for a in vnames_h]
            parts = ax.violinplot(vdata_h, showmeans=True, showextrema=True, widths=0.65)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(vcols_h[i]); pc.set_alpha(0.60)
            for comp in ['cmeans','cmaxes','cmins','cbars']:
                parts[comp].set_linewidth(1.6)
            for i, (nm, vals) in enumerate(zip(vnames_h, vdata_h)):
                ax.scatter([i]*len(vals), vals, color=vcols_h[i],
                           s=30, zorder=5, alpha=0.75, edgecolors='white', lw=0.4)
            ax.set_xticks(range(len(vnames_h)))
            ax.set_xticklabels(vnames_h, rotation=12, ha='right')
            ax.set_ylabel(title); ax.set_title(f'{title} Distribution across Seeds')
        plt.suptitle('Robustness: Offloading and CHR across Seeds', fontsize=16, fontweight='bold')
        plt.tight_layout(); savefig(fig, 'fig17_violin_both')
        _fig_pbar.update(1)
        
        # ── fig18: HP sensitivity — training episodes ──
        print('>>> fig18_hp_episodes')
        cfca_ref = np.mean(main_res['CFCA']['off'])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(EP_VALS, hp_offs, 's-', color=S['RL-CWCA']['color'], lw=2.5, ms=10)
        ax.axhline(cfca_ref, color=S['CFCA']['color'], ls='--', lw=1.8, label='CFCA baseline')
        for ep_x, ep_y in zip(EP_VALS, hp_offs):
            ax.text(ep_x, ep_y+0.003, f'{ep_y:.4f}', ha='center', va='bottom', fontsize=12)
        ax.set_xlabel('DQN Training Episodes'); ax.set_ylabel('Offloading Ratio')
        ax.set_title('Hyperparameter Sensitivity: Number of Training Episodes')
        ax.legend(); plt.tight_layout(); savefig(fig, 'fig18_hp_episodes')
        _fig_pbar.update(1)
        
        # ── fig19: NNPM (net profit margin vs cache size) ──
        print('>>> fig19_nnpm')
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for nm in ['RL-CWCA', 'CFCA', 'SAA']:
            ax.plot(CACHE_SIZES, nnpm_res[nm], label=nm, **lp(nm))
        ax.axhline(0, color='gray', ls='--', alpha=0.5, lw=1)
        ax.set_xlabel('Cache Size (MB)'); ax.set_ylabel('Net Profit Margin (%)')
        ax.set_title('Net Profit Margin (NNPM) vs Cache Size')
        # Zoom y-axis to make RL-CWCA / SAA comparison readable; CFCA's deep
        # negative tail (down to ≈ -1050%) is clipped off-screen on purpose.
        ax.set_ylim(-300, 50)
        ax.legend(ncol=2); plt.tight_layout(); savefig(fig, 'fig19_nnpm')
        _fig_pbar.update(1)
        
        # ── fig20: Deployment time efficiency ──
        print('>>> fig20_efficiency')
        TIME_ALGOS = ['RL-CWCA', 'CFCA', 'SAA', 'Greedy', 'Popular Cache']
        mean_times = [np.mean(DEPLOY_TIMES.get(a, [0]))*1000 for a in TIME_ALGOS]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(TIME_ALGOS, mean_times,
                      color=[S[a]['color'] for a in TIME_ALGOS],
                      edgecolor='black', lw=0.7, alpha=0.88)
        for bar, t in zip(bars, mean_times):
            ax.text(bar.get_x()+bar.get_width()/2, t+0.5,
                    f'{t:.1f} ms', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_ylabel('Deployment Time (ms)')
        ax.set_title('Deployment Decision Time\n(single call, 172 devices)')
        plt.tight_layout(); savefig(fig, 'fig20_efficiency')
        _fig_pbar.update(1)
        
        # ── fig21: DRL improvement bars at 2 GB ──
        print('>>> fig21_drl_improvement_2gb')
        DRL_SHOW = ['DQN-Interest', 'DQN-Weak', 'DQN-Pop', 'CFCA', 'SAA']
        rl_off_2gb = np.mean(cache_res['RL-CWCA'][2000]['off'])
        rl_chr_2gb = np.mean(cache_res['RL-CWCA'][2000]['chr'])
        imp_o2 = [(rl_off_2gb - np.mean(cache_res[b][2000]['off'])) /
                   max(np.mean(cache_res[b][2000]['off']), 1e-9) * 100 for b in DRL_SHOW]
        imp_c2 = [(rl_chr_2gb - np.mean(cache_res[b][2000]['chr'])) /
                   max(np.mean(cache_res[b][2000]['chr']), 1e-9) * 100 for b in DRL_SHOW]
        x = np.arange(len(DRL_SHOW)); bw = 0.36
        fig, ax = plt.subplots(figsize=(10, 5.2))
        b1 = ax.bar(x-bw/2, imp_o2, bw, label='Offloading ↑',
                    color='#E63946', edgecolor='black', lw=0.7, alpha=0.88)
        b2 = ax.bar(x+bw/2, imp_c2, bw, label='CHR ↑',
                    color='#1D6FA4', edgecolor='black', lw=0.7, alpha=0.88)
        for bar in list(b1)+list(b2):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.4,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=11.5, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(DRL_SHOW, rotation=12, ha='right')
        ax.set_ylabel('Improvement (%)'); ax.legend()
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_title('RL-CWCA vs Baselines at 2 GB Cache')
        plt.tight_layout(); savefig(fig, 'fig21_drl_improvement_2gb')
        _fig_pbar.update(1)
        
        
        # ──────────────────────────────────────────────────────────────
        _fig_pbar.close()
        _rm.mark_phase(RUN_DIR, 'phase11_figures', {'count': 21})
        
        # PHASE 12: GENERATE ALL TABLES
        # ──────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("PHASE 12: Generating tables")
        print("=" * 60)
        
        import time as _time
        
        # Table 1: Main results (3 seeds)
        rows = []
        for nm in MAIN_ALGOS:
            om = np.mean(main_res[nm]['off']); osd = np.std(main_res[nm]['off'])
            cm = np.mean(main_res[nm]['chr']); csd = np.std(main_res[nm]['chr'])
            rows.append({'Algorithm': nm,
                         'Offloading': f'{om:.4f}+/-{osd:.4f}',
                         'CHR':        f'{cm:.4f}+/-{csd:.4f}'})
        pd.DataFrame(rows).to_csv(f'{TD}/table1_main.csv', index=False)
        print("  [OK] table1_main.csv")
        
        # Table 2: Cache sweep
        drl_rows = []
        for nm in DRL_ORDER:
            row = {'Algorithm': nm}
            for cm_ in [500, 1000, 1500, 2000]:
                row[f'Off@{cm_}'] = f"{np.mean(cache_res[nm][cm_]['off']):.4f}"
            drl_rows.append(row)
        pd.DataFrame(drl_rows).to_csv(f'{TD}/table2_cache_sweep.csv', index=False)
        print("  [OK] table2_cache_sweep.csv")
        
        # Table 3: Ablation
        abl_rows = [{'Configuration': k.replace('\n',' '),
                     'Offloading': f'{v[0]:.4f}',
                     'Std': f'{v[1]:.4f}',
                     'Delta': f'{(abl_res[list(abl_res.keys())[0]][0]-v[0])/abl_res[list(abl_res.keys())[0]][0]*100:.1f}%'}
                    for k, v in abl_res.items()]
        pd.DataFrame(abl_rows).to_csv(f'{TD}/table3_ablation.csv', index=False)
        print("  [OK] table3_ablation.csv")
        
        # Table 4: Significance (10 seeds)
        sig_rows = []
        for b in SIG_NAMES[1:]:
            i_rl = SIG_NAMES.index('RL-CWCA'); i_b = SIG_NAMES.index(b)
            p = SIG_P[i_rl, i_b]
            u, _ = stats.mannwhitneyu(sig_res['RL-CWCA'], sig_res[b], alternative='two-sided')
            n1, n2 = len(sig_res['RL-CWCA']), len(sig_res[b])
            r_rb = abs(1 - 2*u / (n1*n2))
            sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
            sig_rows.append({'Comparison': f'RL-CWCA vs {b}',
                             'U': f'{u:.0f}', 'p': f'{p:.4f}',
                             'Significance': sig, 'r': f'{r_rb:.3f}'})
        pd.DataFrame(sig_rows).to_csv(f'{TD}/table4_significance.csv', index=False)
        print("  [OK] table4_significance.csv")
        
        # Table 5: Controller comparison
        ctrl_rows = []
        for nm in ['RL-CWCA', 'PPO', 'A2C', 'DQN', 'CFCA']:
            m1, s1 = ctrl_res[1000][nm]; m2, s2 = ctrl_res[2000][nm]
            label = 'RL-CWCA (A2C, full)' if nm == 'RL-CWCA' else f'{nm} (CW only)' if nm in ('PPO', 'A2C', 'DQN') else nm
            ctrl_rows.append({'Controller': label,
                              'Off@1GB': f'{m1:.4f}+/-{s1:.4f}',
                              'Off@2GB': f'{m2:.4f}+/-{s2:.4f}'})
        pd.DataFrame(ctrl_rows).to_csv(f'{TD}/table5_controller.csv', index=False)
        print("  [OK] table5_controller.csv")
        
        # Table 6: Coverage & Fairness
        cov_rows = [{'Algorithm': a,
                     'Offloading': f'{np.mean(cov_res[a]["off"]):.4f}',
                     'Coverage_%': f'{np.mean(cov_res[a]["cov"])*100:.1f}',
                     'Jain':       f'{np.mean(cov_res[a]["jain"]):.4f}'}
                    for a in COV_ALGOS]
        pd.DataFrame(cov_rows).to_csv(f'{TD}/table6_coverage_fairness.csv', index=False)
        print("  [OK] table6_coverage_fairness.csv")
        
        # Table 7: Contention sensitivity
        coll_rows = [{'rho': cf,
                      'RL-CWCA': f'{coll_res["RL-CWCA"][i]:.4f}',
                      'CFCA':    f'{coll_res["CFCA"][i]:.4f}',
                      'SAA':     f'{coll_res["SAA"][i]:.4f}',
                      'Delta_RL_CFCA': f'{(coll_res["RL-CWCA"][i]-coll_res["CFCA"][i])/coll_res["CFCA"][i]*100:+.1f}%'}
                     for i, cf in enumerate(COLL_FACTORS)]
        pd.DataFrame(coll_rows).to_csv(f'{TD}/table7_contention.csv', index=False)
        print("  [OK] table7_contention.csv")
        
        # Table 8: Aligned-threshold comparison across cache sizes (reviewer request)
        # Shows RL-CWCA (aligned, I>=0.05) vs DQN-Interest at same threshold
        aln_rows = []
        for cm_ in [500, 1000, 1500, 2000]:
            rl_aln  = np.mean(cache_res['RL-CWCA-aligned'][cm_]['off'])
            rl_full = np.mean(cache_res['RL-CWCA'][cm_]['off'])
            dqi     = np.mean(cache_res['DQN-Interest'][cm_]['off'])
            lead_pct = (rl_aln - dqi) / max(dqi, 1e-9) * 100
            aln_rows.append({
                'Cache_MB':          cm_,
                'RL-CWCA (full, 0.005)':     f'{rl_full:.4f}',
                'RL-CWCA (aligned, 0.05)':   f'{rl_aln:.4f}',
                'DQN-Interest (0.05)':       f'{dqi:.4f}',
                'Lead_aligned_vs_DQNInt_%':  f'{lead_pct:+.1f}%',
            })
        pd.DataFrame(aln_rows).to_csv(f'{TD}/table8_aligned_comparison.csv', index=False)
        print("  [OK] table8_aligned_comparison.csv")
        print()
        print("  Aligned threshold (RL-CWCA, I>=0.05) vs DQN-Interest at matched threshold:")
        for row in aln_rows:
            print(f"    {row['Cache_MB']} MB: RL-CWCA-aligned={row['RL-CWCA (aligned, 0.05)']}"
                  f"  DQN-Interest={row['DQN-Interest (0.05)']}"
                  f"  Lead={row['Lead_aligned_vs_DQNInt_%']}")
        
        # Summary markdown
        with open('results/summary.md', 'w', encoding='utf-8') as f:
            f.write(f"# RL-CWCA Results Summary\n\n")
            f.write(f"Generated: {_time.strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"## Main Results (1 GB, γ=0.6, 3 seeds)\n\n")
            f.write("| Algorithm | Offloading | CHR |\n|---|---|---|\n")
            for nm in MAIN_ALGOS:
                om = np.mean(main_res[nm]['off']); cm_ = np.mean(main_res[nm]['chr'])
                f.write(f"| {nm} | {om:.4f} | {cm_:.4f} |\n")
            f.write(f"\n## Aligned Threshold Comparison (I_uc >= 0.05, matched)\n\n")
            f.write("| Cache (MB) | RL-CWCA full | RL-CWCA aligned | DQN-Interest | Lead (aligned) |\n")
            f.write("|---|---|---|---|---|\n")
            for row in aln_rows:
                f.write(f"| {row['Cache_MB']} | {row['RL-CWCA (full, 0.005)']} "
                        f"| {row['RL-CWCA (aligned, 0.05)']} "
                        f"| {row['DQN-Interest (0.05)']} "
                        f"| {row['Lead_aligned_vs_DQNInt_%']} |\n")
            f.write(f"\n## Speed Constraint Ablation (bug-fixed)\n\n")
            wo_speed_key = 'w/o\\nSpeed Constraint'
            for k, (m, sd) in abl_res.items():
                if 'Speed' in k:
                    f.write(f"- w/o Speed Constraint (fixed): offloading={m:.4f}+/-{sd:.4f}\n")
            f.write(f"\n## Significance (10 seeds)\n\n")
            for row in sig_rows:
                f.write(f"- {row['Comparison']}: p={row['p']} ({row['Significance']}), r={row['r']}\n")
        
        print("\n  [OK] summary.md")
        print("\n" + "=" * 60)
        print(f"  COMPLETE: 13 figures + 8 tables")
        print(f"  Figures:  {FD}/")
        print(f"  Tables:   {TD}/")
        print(f"  Key new outputs:")
        print(f"    table3_ablation.csv        — speed constraint result (bug-fixed)")
        print(f"    table8_aligned_comparison.csv — aligned-threshold vs DQN-Interest")
        print(f"    results/summary.md         — includes aligned & bug-fix sections")
        print("=" * 60)
        
        
        # ──────────────────────────────────────────────────────────────
    
    else:
        # _P13_ONLY mode: initialize stubs so _rm.complete_run doesn't KeyError
        MAIN_ALGOS  = ['RL-CWCA', 'DRL-Binary',
                       'DQN-Interest', 'DQN-Weak', 'DQN-Pop',
                       'CFCA', 'SAA', 'Greedy', 'Popular Cache']
        SWEEP_ALGOS = MAIN_ALGOS + ['RL-CWCA-aligned']
        main_res    = {nm: {'off': [0.0], 'chr': [0.0]} for nm in MAIN_ALGOS}
        cache_res   = {a: {sz: {'off': [0.0]} for sz in [500,750,1000,1500,2000]}
                       for a in SWEEP_ALGOS}
        HEURISTICS  = {'CFCA': a_cfca, 'SAA': a_saa,
                       'Greedy': a_greedy, 'Popular Cache': a_pop}
    
    # PHASE 13: TRANSFER VALIDATION — MICROSOFT T-DRIVE (REAL DATA)
    # ──────────────────────────────────────────────────────────────
    # Validates policy transfer to a real-world GPS contact trace:
    # 10,357 Beijing taxis, one week, no registration required.
    # Source: Yuan et al. (2010), ACM SIGSPATIAL. 800+ citations.
    #
    # One-time setup (~3 min):
    #   python fetch_tdrive.py \
    #       --n-taxis 200 \
    #       --out results/tdrive/tdrive_contact_data.npz
    #
    # Differences from Marylebone (SUMO):
    #   - Real GPS traces, not simulation
    #   - All vehicles (taxis), no pedestrians
    #   - Beijing urban topology vs London Marylebone
    #   - Sparser contact graph (taxis cover larger area)
    #   - Citation: Yuan et al. (2010) T-Drive, SIGSPATIAL GIS
    # ──────────────────────────────────────────────────────────────
    _TDRIVE_NPZ = Path('results/tdrive/tdrive_contact_data.npz')

    # Auto-generate NPZ from zip if not already built
    if not _TDRIVE_NPZ.exists():
        _ZIP_CANDIDATES = [
            Path('tdrive.zip'),
            Path('results/tdrive/TaxiData.zip'),
            Path('TaxiData.zip'),
        ]
        _zip_found = next((p for p in _ZIP_CANDIDATES if p.exists()), None)
        if _zip_found:
            print(f"\n  [Phase 13] NPZ not found — building from {_zip_found} ...")
            import fetch_tdrive as _ft
            _parsed   = _ft.parse_zip(_zip_found, n_taxis=200, seed=42)
            _ep_pos   = _ft.bin_trajectories(_parsed, epoch_s=30)
            _tids     = list(_parsed.keys())
            _contacts = _ft.compute_contacts(_ep_pos, _tids, 120.0, 30, 30.0)
            _avg, _nct = _ft.build_matrices(len(_tids), _contacts)
            _vt       = np.array(['car'] * len(_tids), dtype=object)
            _TDRIVE_NPZ.parent.mkdir(parents=True, exist_ok=True)
            np.savez(_TDRIVE_NPZ, avg_ct=_avg, n_ct=_nct, vtypes_arr=_vt)
            print(f"  [Phase 13] NPZ saved -> {_TDRIVE_NPZ}")
        else:
            raise FileNotFoundError(
                f"Phase 13 requires T-Drive data.\n"
                f"Place tdrive.zip in the project root, or run:\n"
                f"  python fetch_tdrive.py --out {_TDRIVE_NPZ}"
            )

    print("\n" + "=" * 60)
    print("PHASE 13: Transfer validation (T-Drive, Beijing taxis)")
    print("=" * 60)

    if True:  # always runs — NPZ guaranteed above
    
        _td      = np.load(_TDRIVE_NPZ, allow_pickle=True)
        _ACT_TD  = _td['avg_ct']
        _NCT_TD  = _td['n_ct']
        _VT_TD   = list(_td['vtypes_arr'])
        _ND_TD   = len(_VT_TD)
        _n_cars  = sum(1 for t in _VT_TD if t == 'car')
        print(f"  T-Drive: N={_ND_TD} taxis")
        print(f"  Active pairs: {(_NCT_TD > 0).sum() // 2}")
        act_td = _ACT_TD[_ACT_TD > 0]
        if len(act_td):
            print(f"  Mean contact: {act_td.mean():.1f}s  Median: {np.median(act_td):.1f}s")
    
        class SimTDrive(Sim):
            """Sim using T-Drive (Beijing taxi) contact matrices.
    
            Delegates all init logic to Sim via _contacts so any future
            attribute added to Sim.__init__ is automatically inherited.
    
            pri is normalised to [0,1] via norm_pri() after super().__init__().
            Raw value is kept in pri_raw for diagnostics.
    
            Why normalise: T-Drive has sparse contacts so raw π̄_u ≈ 0.01–0.08.
            The CW pull term (1 + 5*π̄_u) spans only 1.05–1.40 (1.33×)
            instead of Marylebone's 1.25–5.00 (4×). Normalising restores
            discriminative power without changing the CW score formula.
            """
            def __init__(self, F, gm, cmb, fmb, br, seed=42, coll_factor=1.0):
                super().__init__(F, gm, cmb, fmb, br, seed=seed,
                                 coll_factor=coll_factor,
                                 _contacts={'act': _ACT_TD, 'nct': _NCT_TD,
                                            'vt': _VT_TD,  'nd': _ND_TD})
                self.pri_raw = self.pri.copy()
                self.pri     = norm_pri(self.pri_raw)
    
        # ── Zero-shot: Marylebone-trained weights on T-Drive topology ──
        XFER_ALGOS = ['RL-CWCA', 'CFCA', 'SAA', 'Greedy', 'Popular Cache']
        xfer_res   = {a: defaultdict(list) for a in XFER_ALGOS}
        xfer_cache = [500, 1000, 2000]
    
        _p13 = tqdm(xfer_cache, desc='Phase 13 T-Drive', ncols=80)
        for cm in _p13:
            _p13.set_postfix(cache=f'{cm}MB')
            for sd in tqdm(MAIN_SEEDS, desc=f'  seeds@{cm}MB',
                           leave=False, ncols=60):
                for name in XFER_ALGOS:
                    sc = SimTDrive(200, .6, cm, 60, 1, seed=sd)
                    if name in HEURISTICS:
                        HEURISTICS[name](sc)
                    else:
                        net, bfn, filt, uo = agents[name]
                        deploy(sc, net, bfn, filt, uo)
                    r = sc.exchange()
                    xfer_res[name][cm].append(r['off'])
    
        # ── Fine-tune: 30 A2C episodes on T-Drive topology (cold-start) ──
        # Empirical finding: warm-starting from Marylebone weights at the
        # standard lr=3e-4 fine-tune rate actually HURTS T-Drive offloading.
        # The Marylebone optimum sits in a different basin (172 mixed devices,
        # dense contacts ~166s mean) than T-Drive (200 taxis, sparse contacts).
        # 30 episodes at low lr can't escape the SUMO basin but does inject
        # gradient noise, ending slightly below zero-shot. Cold-start at the
        # main lr (3e-3) lets the network find a T-Drive-specific optimum.
        # This matches the historically-working pipeline.
        print("  Retraining RL-CWCA on T-Drive topology (30 A2C eps, cold-start, lr=3e-3)...")
        _sc_ft = SimTDrive(200, .6, 1000, 60, 1, seed=GLOBAL_SEED)
        net_ft, _ = train_a2c(_sc_ft, 30, b_cw_score, 'full', o_pri,
                               seed=GLOBAL_SEED, desc='RL-CWCA-TDrive-FT',
                               w_off=W_OFF, w_chr=W_CHR)
        agents['RL-CWCA-TDrive-FT'] = (net_ft, b_cw_score, 'full', o_pri)
    
        xfer_ft = defaultdict(list)
        for sd in tqdm(MAIN_SEEDS, desc='  Fine-tune eval',
                       leave=False, ncols=60):
            sc = SimTDrive(200, .6, 1000, 60, 1, seed=sd)
            deploy(sc, net_ft, b_cw_score, 'full', o_pri)
            xfer_ft['off'].append(sc.exchange()['off'])
    
        # ── Diagnostic: show what norm_pri actually did ──
        _diag = SimTDrive(200, .6, 1000, 60, 1, seed=GLOBAL_SEED)
        _raw  = _diag.pri_raw
        _norm = _diag.pri
        _pull_raw  = 1 + 5 * _raw
        _pull_norm = 1 + 5 * _norm
        tqdm.write(f"\n  π̄_u diagnostic (T-Drive):")
        tqdm.write(f"    Raw  π̄_u: [{_raw.min():.4f}, {_raw.max():.4f}]  "
                   f"pull span: {_pull_raw.min():.2f}–{_pull_raw.max():.2f}  "
                   f"ratio: {_pull_raw.max()/_pull_raw.min():.2f}×")
        tqdm.write(f"    Norm π̄_u: [{_norm.min():.4f}, {_norm.max():.4f}]  "
                   f"pull span: {_pull_norm.min():.2f}–{_pull_norm.max():.2f}  "
                   f"ratio: {_pull_norm.max()/_pull_norm.min():.2f}×")
        _mary_range = 4.0   # from analysis (Marylebone pull ratio)
        tqdm.write(f"    Marylebone pull ratio (reference): {_mary_range:.2f}×")
    
        # ── Print results ──
        print("\n  Transfer results (1 GB, T-Drive Beijing taxis):")
        print(f"  {'Algorithm':<28} Offloading  Lead vs CFCA  Notes")
        cfca_zs = np.mean(xfer_res['CFCA'][1000])
        for name in XFER_ALGOS:
            zs   = np.mean(xfer_res[name][1000])
            lead = (zs - cfca_zs) / max(cfca_zs, 1e-9) * 100
            note = '← norm_pri active' if name == 'RL-CWCA' else ''
            print(f"  {name:<28} {zs:.4f}      {lead:+.1f}%        {note}")
        ft_m    = np.mean(xfer_ft['off'])
        lead_ft = (ft_m - cfca_zs) / max(cfca_zs, 1e-9) * 100
        print(f"  {'RL-CWCA (fine-tune, norm)':<28} {ft_m:.4f}      {lead_ft:+.1f}%"
              f"        <- 30 A2C eps trained on T-Drive (cold-start, lr=3e-3)")
    
        # ── Table 9: transfer across cache sizes ──
        xfer_rows = []
        for cm in xfer_cache:
            row = {'Cache_MB': cm}
            for name in XFER_ALGOS:
                row[name] = f"{np.mean(xfer_res[name][cm]):.4f}"
            cf = np.mean(xfer_res['CFCA'][cm])
            rl = np.mean(xfer_res['RL-CWCA'][cm])
            row['RL-CWCA_ft_norm@1GB'] = f"{ft_m:.4f}" if cm == 1000 else '-'
            row['Lead_RL-CWCA_vs_CFCA_%']    = f'{(rl-cf)/max(cf,1e-9)*100:+.1f}%'
            row['Lead_RL-CWCA-ft_vs_CFCA_%'] = (
                f'{(ft_m-cf)/max(cf,1e-9)*100:+.1f}%' if cm == 1000 else '-')
            row['norm_pri_applied'] = 'yes'
            xfer_rows.append(row)
        pd.DataFrame(xfer_rows).to_csv(f'{TD}/table9_transfer_tdrive.csv',
                                        index=False)
        print(f"\n  [OK] table9_transfer_tdrive.csv")
    
        _rm.mark_phase(RUN_DIR, 'phase13_transfer_tdrive', {
            'dataset':                  'T-Drive (Yuan et al. 2010, SIGSPATIAL)',
            'n_taxis':                  _ND_TD,
            'norm_pri':                 'enabled',
            'pull_ratio_raw':           f'{_pull_raw.max()/_pull_raw.min():.2f}x',
            'pull_ratio_norm':          f'{_pull_norm.max()/_pull_norm.min():.2f}x',
            'rl_cwca_zero_shot_1gb':    f"{np.mean(xfer_res['RL-CWCA'][1000]):.4f}",
            'rl_cwca_finetune_norm_1gb':f"{ft_m:.4f}",
            'cfca_1gb':                 f"{np.mean(xfer_res['CFCA'][1000]):.4f}",
        })

    # ──────────────────────────────────────────────────────────────
    # PHASE 14: CROSS-CITY + DIURNAL EVAL (SUMO ALT TRACES)
    # ──────────────────────────────────────────────────────────────
    # Zero-shot evaluation of the Marylebone-trained agents on any
    # additional SUMO traces the user has dropped into
    # results/sumo_sim/contact_data_*.npz (e.g. LuST full / morning
    # / evening). The phase auto-discovers traces and skips cleanly
    # if none are present, so the pipeline runs unchanged in
    # environments where only the canonical Marylebone NPZ exists.
    #
    # Each alt trace must follow the same NPZ schema as the canonical
    # contact_data.npz:
    #     avg_ct (N,N) float32, n_ct (N,N) int32, vtypes_arr (N,) obj
    # Use build_sumo_contact_npz.py to produce one from a SUMO FCD.
    #
    # Outputs:
    #     {TD}/table10_cross_city.csv
    #     {LD}/phase14_console.txt
    # ──────────────────────────────────────────────────────────────
    _ALT_DIR    = Path('results/sumo_sim')
    # Exclude both the historical 500-generic NPZ and the active canonical
    # trace (_SUMO_NPZ); only NPZs *other than* the one used for training
    # qualify as alt traces.
    _CANONICAL_NAMES = {'contact_data.npz', _SUMO_NPZ.name}
    _ALT_NPZS   = sorted(p for p in _ALT_DIR.glob('contact_data_*.npz')
                         if p.name not in _CANONICAL_NAMES)
    if _ALT_NPZS:
        print("\n" + "=" * 60)
        print(f"PHASE 14: Cross-city / diurnal eval ({len(_ALT_NPZS)} alt trace(s))")
        print("=" * 60)

        _ALT_CACHES = [500, 1000, 2000]
        _ALT_ALGOS  = ['RL-CWCA', 'CFCA', 'SAA', 'Greedy', 'Popular Cache']
        # Re-use heuristics dict from earlier; fall back to a fresh one if missing.
        try:
            _HEURISTICS_14 = HEURISTICS
        except NameError:
            _HEURISTICS_14 = {'CFCA': a_cfca, 'SAA': a_saa,
                              'Greedy': a_greedy, 'Popular Cache': a_pop}

        def _make_sim_class(act, nct, vt, nd):
            """Factory: returns a Sim subclass bound to this trace's contacts."""
            class _SimAlt(Sim):
                def __init__(self, F, gm, cmb, fmb, br, seed=42, coll_factor=1.0):
                    super().__init__(F, gm, cmb, fmb, br, seed=seed,
                                     coll_factor=coll_factor,
                                     _contacts={'act': act, 'nct': nct,
                                                'vt':  vt,  'nd':  nd})
                    # Same norm_pri trick as SimTDrive: makes the pull term
                    # comparable across topologies with very different
                    # contact-frequency scales.
                    self.pri_raw = self.pri.copy()
                    self.pri     = norm_pri(self.pri_raw)
            return _SimAlt

        _alt_rows = []   # one row per (trace, algo, cache)
        for _npz in _ALT_NPZS:
            # Human-readable label: prefer the curated map, fall back to a
            # title-cased version of the stem so unknown traces still show up
            # cleanly in the table (e.g. contact_data_my_city -> "My City").
            _label = _TRACE_LABEL_MAP.get(
                _npz.stem,
                _npz.stem.replace('contact_data_', '').replace('_', ' ').title())
            print(f"\n  Trace: {_label}  ({_npz.name})")
            _td   = np.load(_npz, allow_pickle=True)
            _act  = _td['avg_ct']
            _nct  = _td['n_ct']
            _vt   = list(_td['vtypes_arr'])
            _nd   = len(_vt)
            _td.close()
            _cars = sum(1 for t in _vt if str(t).lower() not in
                        {'pedestrian','ped','bike','bicycle','walking'})
            print(f"    N={_nd}  vehicles={_cars}  "
                  f"active pairs={(_nct > 0).sum()//2}")
            _SimCls = _make_sim_class(_act, _nct, _vt, _nd)

            for _cm in _ALT_CACHES:
                _r = {a: defaultdict(list) for a in _ALT_ALGOS}
                for _sd in MAIN_SEEDS:
                    for _name in _ALT_ALGOS:
                        _s = _SimCls(250, .6, _cm, 80, 1, seed=_sd)
                        if _name in _HEURISTICS_14:
                            _HEURISTICS_14[_name](_s)
                        else:
                            _net, _bfn, _filt, _uo = agents[_name]
                            deploy(_s, _net, _bfn, _filt, _uo)
                        _res = _s.exchange()
                        for _k, _v in _res.items():
                            _r[_name][_k].append(_v)
                for _name in _ALT_ALGOS:
                    _off_m = float(np.mean(_r[_name]['off']))
                    _off_s = float(np.std (_r[_name]['off']))
                    _dd_m  = float(np.mean(_r[_name]['dd']))
                    _cu_m  = float(np.mean(_r[_name]['cu']))
                    _nnpm  = (_dd_m - _cu_m * _cm * 250) / max(_dd_m, 1e-9) * 100
                    _alt_rows.append({
                        'trace':     _label,
                        'algorithm': _name,
                        'cache_MB':  _cm,
                        'off_mean':  round(_off_m, 4),
                        'off_std':   round(_off_s, 4),
                        'nnpm_pct':  round(_nnpm,  2),
                    })

        # ── Table 10: cross-city + diurnal ──
        _df14 = pd.DataFrame(_alt_rows)
        _t10  = Path(TD) / 'table10_cross_city.csv'
        _df14.to_csv(_t10, index=False)
        print(f"\n  [OK] {_t10}")

        # ── Pivoted console summary: trace x algo at 1 GB ──
        print("\n  Zero-shot offloading @ 1 GB cache:")
        _pivot = (_df14[_df14.cache_MB == 1000]
                  .pivot_table(index='trace', columns='algorithm', values='off_mean'))
        print(_pivot.to_string(float_format='%.4f'))

        _rm.mark_phase(RUN_DIR, 'phase14_cross_city', {
            'traces':       [p.name for p in _ALT_NPZS],
            'n_traces':     len(_ALT_NPZS),
            'norm_pri':     'enabled',
            'rl_cwca_1gb':  {row['trace']: row['off_mean']
                             for _, row in _df14[(_df14.cache_MB == 1000) &
                                                  (_df14.algorithm == 'RL-CWCA')].iterrows()},
        })
    else:
        print(f"\n  [Phase 14] No additional SUMO traces in {_ALT_DIR}/  (skipping).")
        print(f"             Canonical trace ({_SUMO_NPZ.name}) is already covered")
        print(f"             by the main Phase 2-10 eval. To enable cross-city /")
        print(f"             diurnal eval, drop additional contact_data_<label>.npz")
        print(f"             files there (e.g. contact_data_lust.npz,")
        print(f"             contact_data_cologne.npz). Known labels:")
        for k in sorted(_TRACE_LABEL_MAP):
            if k == _SUMO_NPZ.stem: continue
            print(f"               {k}.npz  ->  '{_TRACE_LABEL_MAP[k]}'")

    # ──────────────────────────────────────────────────────────────
    # PHASE 15: DELIVERY-THRESHOLD + NEIGHBOUR-LIMIT SENSITIVITY
    # ──────────────────────────────────────────────────────────────
    # The 30% per-neighbour fragment threshold and 8-neighbour retrieval cap
    # are both taken from the CFCA baseline paper. This phase tests whether
    # the RL-CWCA-over-CFCA margin is stable across reasonable alternatives:
    #
    #   delivery_threshold in {0.10, 0.30, 0.50}
    #   max_neighbours     in {4, 8, 16}
    #
    # Evaluated at 1 GB cache, gamma=0.6, 3 seeds, for RL-CWCA and CFCA only
    # (the two algorithms that drive the headline lead). Outputs two tables:
    #   table_a2_threshold_sweep.csv  -- threshold sweep at K=8
    #   table_a3_neighbour_sweep.csv  -- neighbour sweep at tau=0.30
    # Plus a combined 3x3 grid CSV for the supplementary appendix.
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 15: Delivery-threshold + neighbour-limit sensitivity")
    print("=" * 60)

    _P15_THRESHOLDS = [0.10, 0.30, 0.50]
    _P15_NEIGHBOURS = [4, 8, 16]
    _P15_ALGOS      = ['RL-CWCA', 'CFCA']
    _p15_grid_rows  = []   # one row per (tau, K, algo)

    for _tau in tqdm(_P15_THRESHOLDS, desc='Phase 15 grid', ncols=80):
        for _K in _P15_NEIGHBOURS:
            for _alg in _P15_ALGOS:
                _offs = []
                for _sd in MAIN_SEEDS:
                    _s = Sim(250, 0.6, 1000, 80, 1, seed=_sd)
                    if _alg == 'CFCA':
                        a_cfca(_s)
                    else:
                        _net, _bfn, _filt, _uo = agents['RL-CWCA']
                        deploy(_s, _net, _bfn, _filt, _uo)
                    _offs.append(_s.exchange(delivery_threshold=_tau,
                                              max_neighbours=_K)['off'])
                _p15_grid_rows.append({
                    'tau':       _tau,
                    'K':         _K,
                    'algorithm': _alg,
                    'off_mean':  round(float(np.mean(_offs)), 4),
                    'off_std':   round(float(np.std (_offs)), 4),
                })

    _df15 = pd.DataFrame(_p15_grid_rows)

    # ── Threshold sweep (K=8, paper default) ──
    _df_thresh = (_df15[_df15.K == 8]
                  .pivot_table(index='tau', columns='algorithm',
                                values='off_mean'))
    _df_thresh['Delta_pct'] = ((_df_thresh['RL-CWCA'] - _df_thresh['CFCA'])
                               / _df_thresh['CFCA'].clip(lower=1e-9) * 100).round(2)
    _t_a2 = Path(TD) / 'table_a2_threshold_sweep.csv'
    _df_thresh.to_csv(_t_a2)
    print(f"\n  [OK] {_t_a2}")
    print(f"\n  Threshold sweep (K=8):")
    print(_df_thresh.to_string(float_format='%.4f'))

    # ── Neighbour sweep (tau=0.30, paper default) ──
    _df_nbr = (_df15[_df15.tau == 0.30]
               .pivot_table(index='K', columns='algorithm', values='off_mean'))
    _df_nbr['Delta_pct'] = ((_df_nbr['RL-CWCA'] - _df_nbr['CFCA'])
                            / _df_nbr['CFCA'].clip(lower=1e-9) * 100).round(2)
    _t_a3 = Path(TD) / 'table_a3_neighbour_sweep.csv'
    _df_nbr.to_csv(_t_a3)
    print(f"\n  [OK] {_t_a3}")
    print(f"\n  Neighbour sweep (tau=0.30):")
    print(_df_nbr.to_string(float_format='%.4f'))

    # ── Combined 3x3 grid (full sweep) for the appendix ──
    _t_grid = Path(TD) / 'table_a4_threshold_neighbour_grid.csv'
    _df15.to_csv(_t_grid, index=False)
    print(f"\n  [OK] {_t_grid}")

    # Detect whether the RL-CWCA-over-CFCA margin is stable.
    _deltas = []
    for _tau in _P15_THRESHOLDS:
        for _K in _P15_NEIGHBOURS:
            _rl = _df15[(_df15.tau == _tau) & (_df15.K == _K) &
                        (_df15.algorithm == 'RL-CWCA')]['off_mean'].iloc[0]
            _cf = _df15[(_df15.tau == _tau) & (_df15.K == _K) &
                        (_df15.algorithm == 'CFCA')]['off_mean'].iloc[0]
            _deltas.append((_rl - _cf) / max(_cf, 1e-9) * 100)
    _delta_range = max(_deltas) - min(_deltas)
    print(f"\n  RL-CWCA lead range across 3x3 grid: "
          f"{min(_deltas):+.1f}% to {max(_deltas):+.1f}%  "
          f"(spread: {_delta_range:.1f} percentage points)")
    if _delta_range < 5.0:
        print(f"  Verdict: STABLE - threshold/neighbour choices do not drive the result.")
    else:
        print(f"  Verdict: PARAMETER-SENSITIVE - margin varies > 5pp; investigate.")

    _rm.mark_phase(RUN_DIR, 'phase15_sensitivity', {
        'thresholds':         _P15_THRESHOLDS,
        'neighbours':         _P15_NEIGHBOURS,
        'lead_range_pct':     f'{min(_deltas):+.1f} to {max(_deltas):+.1f}',
        'lead_spread_pp':     round(_delta_range, 1),
    })

    # ──────────────────────────────────────────────────────────────
    # PHASE 16: PER-CONTACT-TYPE OFFLOADING BREAKDOWN
    # ──────────────────────────────────────────────────────────────
    # Decomposes RL-CWCA's and CFCA's D2D offloading into four communication
    # types per the paper:
    #   P2P  pedestrian-to-pedestrian (IEEE 802.15.8)
    #   V2V  vehicle-to-vehicle       (IEEE 802.11p)
    #   V2P  vehicle-to-pedestrian    (IEEE 802.11p)
    #   P2V  pedestrian-to-vehicle    (IEEE 802.11p)
    # Bicycles are bucketed with pedestrians (non-motor-vehicle), matching
    # the paper's 4-type scheme.
    #
    # Each transfer is attributed by-provider: when multiple providers
    # contribute to one delivery, the bytes are split among them.
    # 1 GB cache, gamma=0.6, 3 seeds; RL-CWCA and CFCA only.
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 16: Per-contact-type offloading breakdown")
    print("=" * 60)

    _P16_TYPES = ['P2P', 'V2V', 'V2P', 'P2V']
    _P16_ALGOS = ['RL-CWCA', 'CFCA']
    _p16_rows  = []
    for _alg in _P16_ALGOS:
        _agg_dh = {t: 0   for t in _P16_TYPES}
        _agg_dd = {t: 0.0 for t in _P16_TYPES}
        _agg_lh = 0; _agg_total = 0
        for _sd in MAIN_SEEDS:
            _s = Sim(250, 0.6, 1000, 80, 1, seed=_sd)
            if _alg == 'CFCA':
                a_cfca(_s)
            else:
                _net, _bfn, _filt, _uo = agents['RL-CWCA']
                deploy(_s, _net, _bfn, _filt, _uo)
            _r = _s.exchange()
            for t in _P16_TYPES:
                _agg_dh[t] += _r['type_dh'][t]
                _agg_dd[t] += _r['type_dd'][t]
            _agg_lh    += _r['lh']
            _agg_total += _r['tr']

        _total_d2d_mb       = sum(_agg_dd.values()) or 1.0
        _total_d2d_n        = sum(_agg_dh.values()) or 1
        _total_offloaded_n  = _agg_lh + _total_d2d_n   # local hits + D2D
        for t in _P16_TYPES:
            _p16_rows.append({
                'algorithm':         _alg,
                'transfer_type':     t,
                'transfers':         _agg_dh[t],
                'mean_MB_per_tx':    round(_agg_dd[t] / max(_agg_dh[t], 1), 2),
                'pct_of_d2d_n':      round(100 * _agg_dh[t] / _total_d2d_n, 2),
                'pct_of_d2d_mb':     round(100 * _agg_dd[t] / _total_d2d_mb, 2),
                # Partial offloading: fraction of ALL offloaded transfers
                # (local hits + every D2D type) attributable to this type.
                # Local hits have no contact type so they don't appear in
                # any of the 4 partial_phi values; the sum of the 4 plus
                # the local-hit share equals 100%.
                'partial_phi_pct':   round(100 * _agg_dh[t] / max(_total_offloaded_n, 1), 4),
            })

    _df16  = pd.DataFrame(_p16_rows)
    _t_a5  = Path(TD) / 'table_a5_contact_type_breakdown.csv'
    _df16.to_csv(_t_a5, index=False)
    print(f"\n  [OK] {_t_a5}")
    print(f"\n  Per-contact-type breakdown (1 GB cache, gamma=0.6, 3 seeds):")
    _pivot16 = _df16.pivot_table(index='transfer_type', columns='algorithm',
                                  values='pct_of_d2d_mb')
    print(_pivot16.to_string(float_format='%.2f'))
    print('  (numbers are % of total D2D bytes delivered)')

    _rm.mark_phase(RUN_DIR, 'phase16_contact_type', {
        'algorithms': _P16_ALGOS,
        'types':      _P16_TYPES,
    })


    # ── Finalise run record ────────────────────────────────────────
    if not _P13_ONLY:
        _rm.complete_run(RUN_DIR, results_summary={
            'rl_cwca_1gb':     f"{np.mean(main_res['RL-CWCA']['off']):.4f}",
            'rl_cwca_2gb':     f"{np.mean(cache_res['RL-CWCA'][2000]['off']):.4f}",
            'rl_cwca_aligned_1gb': f"{np.mean(cache_res['RL-CWCA-aligned'][1000]['off']):.4f}",
            'rl_cwca_aligned_2gb': f"{np.mean(cache_res['RL-CWCA-aligned'][2000]['off']):.4f}",
            'figures_dir':     FD,
            'tables_dir':      TD,
        })
        print(f"\n  Run files: {RUN_DIR}")
        print(f"  Registry:  {_rm.base / 'registry.json'}")
        print(f"  Latest:    {_rm.base / 'latest.txt'}")
        
        # ── Show all runs table ────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  All runs:")
        print(_rm.list_runs().to_string(index=False))
        print("=" * 60)


# ──────────────────────────────────────────────────────────────
# REWARD WEIGHT COMPARISON
# ──────────────────────────────────────────────────────────────
def run_reward_weight_comparison():
    """
    Train PPO (same setup as RL-CWCA Phase 1) under three reward weight
    configurations and compare training dynamics and final offloading.

    Configs
    -------
    1:1 — gr = res['off']*100 + res['chr']*100
    2:1 — gr = res['off']*200 + res['chr']*100
    3:1 — gr = res['off']*300 + res['chr']*100

    Outputs
    -------
    {TD}/reward_weight_comparison.csv
    {FD}/fig_reward_weights.pdf + .png
    console summary table
    """
    _CONFIGS = [
        ('1:1', 100, 100),
        ('2:1', 200, 100),
        ('3:1', 300, 100),
    ]
    _SEEDS = [42, 43, 44]
    _EPS   = 100
    _CFG_STYLE = {
        '1:1': dict(color='#1D6FA4', ls='--',  lw=2.1, marker='o', ms=6),
        '2:1': dict(color='#E63946', ls='-',   lw=2.8, marker='*', ms=11),
        '3:1': dict(color='#2CA02C', ls='-.',  lw=2.1, marker='^', ms=6),
    }

    csv_rows   = []
    final_off  = {lbl: [] for lbl, _, _ in _CONFIGS}
    mean_curve = {}   # label → np.ndarray(eps,) — mean smoothed reward across seeds

    print("\n" + "=" * 60)
    print("REWARD WEIGHT COMPARISON")
    print("=" * 60)

    for lbl, w_off, w_chr in _CONFIGS:
        print(f"\n  Config {lbl}  (off×{w_off}  chr×{w_chr})")
        seed_smoothed = []

        for sd in _SEEDS:
            # Train A2C under this weight config — train_a2c handles seeding,
            # best-weights tracking and rolls the critic back onto the actor
            # via _ctrl_critic for downstream checkpointing.
            s = Sim(250, 0.6, 1000, 80, 1, seed=sd)
            actor, rews = train_a2c(s, _EPS, b_cw_score, 'full', o_pri,
                                     seed=sd, desc=f'  {lbl} seed={sd}',
                                     w_off=w_off, w_chr=w_chr)

            # 5-episode smoothed reward
            smoothed = (pd.Series(rews)
                          .rolling(5, min_periods=1)
                          .mean()
                          .tolist())
            seed_smoothed.append(smoothed)

            # Final offloading: deploy best weights and evaluate
            s_eval = Sim(250, 0.6, 1000, 80, 1, seed=sd)
            deploy(s_eval, actor, b_cw_score, 'full', o_pri)
            off_val = s_eval.exchange()['off']
            final_off[lbl].append(off_val)

            for ep_i, (raw, sm) in enumerate(zip(rews, smoothed)):
                csv_rows.append({
                    'weight_config':    lbl,
                    'seed':             sd,
                    'episode':          ep_i + 1,
                    'raw_reward':       raw,
                    'smoothed_reward':  sm,
                    'final_offloading': off_val,
                })

        mean_curve[lbl] = np.mean(seed_smoothed, axis=0)

    # ── CSV ──────────────────────────────────────────────────────
    df = pd.DataFrame(csv_rows)
    csv_path = Path(TD) / 'reward_weight_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n  [OK] {csv_path}')

    # ── Figure ───────────────────────────────────────────────────
    episodes = np.arange(1, _EPS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for lbl, _, _ in _CONFIGS:
        st = _CFG_STYLE[lbl]
        ax1.plot(episodes, mean_curve[lbl], label=lbl,
                 color=st['color'], ls=st['ls'], lw=st['lw'],
                 marker=st['marker'], ms=st['ms'], markevery=9)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Smoothed Reward (5-ep window)')
    ax1.set_title('Training Reward vs Episode')
    ax1.legend(title='Weight (off:chr)', loc='lower right')

    labels  = [lbl for lbl, _, _ in _CONFIGS]
    means   = [np.mean(final_off[lbl]) for lbl in labels]
    errs    = [np.std(final_off[lbl])  for lbl in labels]
    colors  = [_CFG_STYLE[lbl]['color'] for lbl in labels]
    ax2.bar(labels, means, yerr=errs, capsize=6,
            color=colors, width=0.5, edgecolor='k', linewidth=0.8)
    ax2.set_xlabel('Reward Weight (off : chr)')
    ax2.set_ylabel('Final Offloading Ratio')
    ax2.set_title('Final Offloading by Reward Config')

    fig.tight_layout()
    savefig(fig, 'fig_reward_weights')

    # ── Console summary + pick best config ──────────────────────
    hdr = f"  {'Weight config':<14} | {'Mean final reward':>18} | {'Mean final offloading':>22}"
    print("\n  Reward Weight Comparison Summary")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    best_lbl, best_off, best_w_off, best_w_chr = None, -1.0, 200, 100
    for lbl, w_off_c, w_chr_c in _CONFIGS:
        mean_rew = float(mean_curve[lbl][-1])
        mean_off = float(np.mean(final_off[lbl]))
        marker = ' ←' if mean_off > best_off else ''
        print(f"  {lbl:<14} | {mean_rew:>18.2f} | {mean_off:>22.4f}{marker}")
        if mean_off > best_off:
            best_off, best_lbl = mean_off, lbl
            best_w_off, best_w_chr = w_off_c, w_chr_c
    print(f"\n  Selected reward weights: {best_lbl}  "
          f"(off×{best_w_off} + chr×{best_w_chr})  "
          f"→ offloading {best_off:.4f}")
    print()
    return best_w_off, best_w_chr


if __name__ == '__main__':
    main()
