#!/usr/bin/env python3
"""
══════════════════════════════════════════
  RL-CWCA: Complete Experimental Pipeline  
  ────────────────────────────────────────
  Run management:
    Each execution creates a timestamped run directory:
      results/runs/<YYYYMMDD_HHMMSS>/figures/
      results/runs/<YYYYMMDD_HHMMSS>/tables/
      results/runs/<YYYYMMDD_HHMMSS>/logs/
      results/runs/<YYYYMMDD_HHMMSS>/meta.json
    A registry at results/runs/registry.json tracks all runs.
    The latest run ID is written to results/runs/latest.txt.

  Phases:
    1  Train all DRL agents + A2C/PPO controllers
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
import os, random, time, warnings, json
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
        with open(self.registry_path, 'w') as f:
            json.dump(reg, f, indent=2)


# Initialise run
_RUN_CONFIG = {
    'seeds_main': [42, 43, 44],
    'seeds_sig':  list(range(42, 52)),
    'episodes':   45,
    'cache_1gb':  1000,
    'gamma':      0.6,
    'speed_fix':  'nospeed=th_gt_for_all (Rev3)',
    'aligned_threshold': 0.05,
}
_rm      = RunManager()
RUN_ID, RUN_DIR = _rm.new_run(config=_RUN_CONFIG)
FD = str(RUN_DIR / 'figures')
TD = str(RUN_DIR / 'tables')
LD = str(RUN_DIR / 'logs')

# ── Seeding ────────────────────────────────────────────────────
GLOBAL_SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(GLOBAL_SEED)

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
    'CFCA':          dict(color='#1D6FA4', marker='o',  ls='--',  lw=2.1, ms=6,  z=3),
    'SAA':           dict(color='#2CA02C', marker='^',  ls='-.',  lw=2.1, ms=6,  z=3),
    'Greedy':        dict(color='#9467BD', marker='D',  ls=':',   lw=1.9, ms=5,  z=2),
    'Popular Cache': dict(color='#8C8C8C', marker='v',  ls=':',   lw=1.7, ms=5,  z=2),
    'A2C':           dict(color='#17BECF', marker='p',  ls='--',  lw=2.1, ms=7,  z=3),
    'PPO':           dict(color='#BCBD22', marker='h',  ls='-.',  lw=2.1, ms=7,  z=3),
    'DQN (CW score)': dict(color='#8B4513', marker='D',  ls='--',  lw=2.0, ms=6,  z=3),
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
data = np.load('../results/sumo_sim/contact_data.npz', allow_pickle=True)
ACT = data['avg_ct']
NCT = data['n_ct']
VT  = list(data['vtypes_arr'])
ND  = len(VT)
SD  = 30                            # state-vector dimension
MULTS = np.array([.2, .5, 1., 2., 5.])
print(f"Loaded SUMO: {ND} devices\n")


class Sim:
    def __init__(s, F, gm, cmb, fmb, br, seed=42, coll_factor=1.0):
        rng = np.random.RandomState(seed)
        s.N = ND; s.F = F; s.br = br; s.avg_ct = ACT
        s.coll = coll_factor          # ← contention model
        mx = NCT.max() if NCT.max() > 0 else 1
        s.mp = NCT / mx; np.fill_diagonal(s.mp, 0)
        s.ints = rng.dirichlet(np.ones(15), s.N)
        mn  = np.minimum(s.ints[:, None, :], s.ints[None, :, :]).sum(2)
        mx2 = np.maximum(s.ints[:, None, :], s.ints[None, :, :]).sum(2)
        s.isim = np.divide(mn, mx2, out=np.zeros_like(mn), where=mx2 > 0)
        np.fill_diagonal(s.isim, 0)
        s.vr = np.array([1. if t == 'car' else 0. for t in VT])
        rk = np.arange(1, F + 1, dtype=float)
        rp = rk ** (-gm); s.fp = rp / rp.sum()
        s.fc = np.arange(F) % 15
        s.fs = rng.uniform(max(10, fmb * .3), fmb * 2., F)
        v = ACT[ACT > 0]
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

    def exchange(s, nreq=15):
        """
        D2D exchange with:
        - per-neighbor 30% fragment threshold (not aggregate)
        - coll_factor scales effective bit rate (contention model)
        - returns per-user offloading for Jain fairness
        """
        rng = np.random.RandomState(99)
        tr = lh = dh = dd = 0.
        eff_br = s.br * s.coll
        per_u  = np.zeros(s.N)
        req_u  = np.zeros(s.N)

        for i in range(s.N):
            nr = rng.poisson(nreq)
            w  = s.fp * (.2 + .8 * s.ints[i, s.fc]); w /= w.sum()
            for fid in rng.choice(s.F, nr, p=w):
                tr += 1; req_u[i] += 1
                if s.cache[i, fid]:
                    lh += 1; per_u[i] += 1.0; continue
                pv = np.where(s.cache[:, fid])[0]
                if len(pv) == 0: continue
                pr = s.mp[i, pv]
                k  = min(8, len(pv))
                ti = np.argpartition(-pr, k)[:k] if k < len(pv) else np.arange(len(pv))
                delta = 0.0
                for idx in ti:
                    j = pv[idx]
                    if rng.random() < pr[idx]:
                        ct   = s.avg_ct[i, j]
                        frac = min(s.fs[fid], eff_br * ct) / s.fs[fid]
                        if frac >= 0.3:          # per-neighbor threshold
                            delta = min(1.0, delta + frac)
                            if delta >= 1.0: break
                if delta > 0:
                    dh += 1; dd += delta * s.fs[fid]
                per_u[i] += delta

        off = (lh + dh) / max(tr, 1)
        ch  = (lh + .7 * dh) / max(tr, 1)
        cu  = s.cu.mean() / s.tc.mean()
        d2d = dh / max(dh + max(tr - lh - dh, 0), 1)
        lr  = lh / max(tr, 1)

        # per-user offloading ratio for Jain
        phi = np.where(req_u > 0, per_u / req_u, 0.0)
        active = phi[req_u > 0]
        jain = (active.sum()**2 / (len(active) * (active**2).sum())
                if len(active) > 1 else 1.0)
        cov  = (per_u > 0).sum() / max(s.N, 1)   # fraction of users served

        return dict(off=off, chr=ch, lh=lh, dh=dh, tr=tr, dd=dd,
                    cu=cu, d2d=d2d, lr=lr, jain=jain, cov=cov)


def st_vec(s, u, f):
    return np.concatenate([
        s.ints[u],
        [s.mp[u].mean(), s.mp[u].max(), s.isim[u].mean(), s.isim[u].max(),
         s.cu[u] / max(s.tc[u], 1), s.cache[u].sum() / max(s.F, 1)],
        [s.vr[u], 0, 0],
        [s.vr[u], s.avg_ct[u].mean() / 300.,
         s.fp[f] * 100, s.fs[f] / 250.,
         s.cache[:, f].sum() / max(s.N, 1), s.fc[f] / 15.]
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
# 4. SCORING FUNCTIONS & ORDERING
# ──────────────────────────────────────────────────────────────
def b_cw_score(s, u, f):  return s.fp[f] * s.ints[u, s.fc[f]] * (1 + s.pri[u] * 5)
def b_interest(s, u, f): return s.fp[f] * (1 + s.ints[u, s.fc[f]] * 2)
def b_weak(s, u, f):     return s.fp[f] * (1 + s.ints[u, s.fc[f]] * .5)
def b_pop(s, u, f):      return s.fp[f]
def o_pri(s):  return np.argsort(-s.pri)
def o_rand(s): idx = np.arange(s.N); np.random.shuffle(idx); return idx


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
}


# ──────────────────────────────────────────────────────────────
# ── Checkpoint support ──────────────────────────────────────────
_CKPT = Path('results/checkpoints/agents.pt')
_BFN_MAP  = {'b_cw_score': b_cw_score, 'b_interest': b_interest,
             'b_weak': b_weak, 'b_pop': b_pop}
_UORD_MAP = {'o_pri': o_pri, 'o_rand': o_rand}
_NET_MAP  = {'PolicyNet': PolicyNet, 'DQNet': DQNet}

def _save_checkpoint(agents):
    _CKPT.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {}
    for name, (net, bfn, filt, uord) in agents.items():
        ckpt[name] = {
            'state_dict': net.state_dict(),
            'net_type': net.__class__.__name__,
            'bfn':  bfn.__name__,
            'filt': filt,
            'uord': uord.__name__,
        }
    torch.save(ckpt, _CKPT)
    print(f'  [CKPT] Saved {len(ckpt)} agents → {_CKPT}')

def _load_checkpoint():
    ckpt = torch.load(_CKPT, map_location='cpu')
    agents = {}
    for name, d in ckpt.items():
        net = _NET_MAP[d['net_type']]()
        net.load_state_dict(d['state_dict'])
        agents[name] = (net, _BFN_MAP[d['bfn']], d['filt'], _UORD_MAP[d['uord']])
    print(f'  [CKPT] Loaded {len(agents)} agents from {_CKPT}')
    return agents

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


def train_dqn(s, eps, base_fn, filt, uord_fn, seed=GLOBAL_SEED, desc='DQN'):
    set_seed(seed)
    net = DQNet(); tgt = DQNet(); tgt.load_state_dict(net.state_dict())
    opt = optim.Adam(net.parameters(), lr=1e-3)
    buf = deque(maxlen=12000); best_r = -1e9; best_w = None; rews = []; ev = 1.

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
        res = s.exchange(); gr = res['off'] * 200 + res['chr'] * 100

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


def train_a2c(s, eps, base_fn, filt, uord_fn, seed=GLOBAL_SEED,
              lr=3e-4, gamma=0.95, ent=0.01, vc=0.5, desc='A2C'):
    set_seed(seed)
    actor = PolicyNet(); critic = ValueNet()
    opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    best_r = -1e9; best_w = None; rews = []

    for ep in trange(eps, desc=f'  {desc}', leave=False, ncols=80,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ep [{elapsed}<{remaining}]'):
        s.clear()
        cands = _candidates(s, filt, uord_fn)
        svs, acts_t, cand_pairs = [], [], []
        for u, fi in cands:
            sv_t = torch.FloatTensor(st_vec(s, u, fi)).unsqueeze(0)
            with torch.no_grad():
                logits = actor(sv_t)
                probs  = torch.softmax(logits, -1)
                act    = torch.multinomial(probs, 1).item()
            svs.append(st_vec(s, u, fi)); acts_t.append(act)
            cand_pairs.append((u, fi, base_fn(s, u, fi) * MULTS[act]))

        cand_pairs.sort(key=lambda x: -x[2])
        for u, fi, _ in cand_pairs: s.put(u, fi)
        res = s.exchange(); gr = res['off'] * 200 + res['chr'] * 100; rews.append(gr)

        if svs:
            sb = torch.FloatTensor(np.array(svs))
            ab = torch.LongTensor(acts_t)
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
    return actor, rews


def train_ppo(s, eps, base_fn, filt, uord_fn, seed=GLOBAL_SEED,
              lr=3e-4, clip=0.2, ppo_ep=4, ent=0.01, vc=0.5, desc='PPO'):
    set_seed(seed)
    actor = PolicyNet(); critic = ValueNet()
    opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    best_r = -1e9; best_w = None; rews = []

    for ep in trange(eps, desc=f'  {desc}', leave=False, ncols=80,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ep [{elapsed}<{remaining}]'):
        s.clear()
        cands = _candidates(s, filt, uord_fn)
        svs, acts_t, old_lps, cand_pairs = [], [], [], []
        for u, fi in cands:
            sv_t = torch.FloatTensor(st_vec(s, u, fi)).unsqueeze(0)
            with torch.no_grad():
                logits = actor(sv_t)
                probs  = torch.softmax(logits, -1)
                act    = torch.multinomial(probs, 1).item()
                olp    = torch.log(probs[0, act] + 1e-8).item()
            svs.append(st_vec(s, u, fi)); acts_t.append(act); old_lps.append(olp)
            cand_pairs.append((u, fi, base_fn(s, u, fi) * MULTS[act]))

        cand_pairs.sort(key=lambda x: -x[2])
        for u, fi, _ in cand_pairs: s.put(u, fi)
        res = s.exchange(); gr = res['off'] * 200 + res['chr'] * 100; rews.append(gr)

        if svs:
            sb  = torch.FloatTensor(np.array(svs))
            ab  = torch.LongTensor(acts_t)
            olb = torch.FloatTensor(old_lps)
            ret = torch.FloatTensor([gr / max(len(svs), 1)] * len(svs))
            for _ in range(ppo_ep):
                logits = actor(sb); vals = critic(sb)
                lp_all = torch.log_softmax(logits, -1)
                nlp    = lp_all.gather(1, ab.unsqueeze(1)).squeeze()
                adv    = (ret - vals.detach())
                adv    = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio  = torch.exp(nlp - olb)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
                pl = -torch.min(s1, s2).mean()
                vl = nn.MSELoss()(vals, ret)
                ent_l = -(torch.softmax(logits, -1) * lp_all).sum(1).mean()
                loss  = pl + vc * vl - ent * ent_l
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.)
                opt.step()

        if gr > best_r:
            best_r = gr; best_w = {k: v.clone() for k, v in actor.state_dict().items()}

    if best_w: actor.load_state_dict(best_w)
    return actor, rews


def deploy(s, net, base_fn, filt, uord_fn):
    # net can be DQNet (outputs Q-values) or PolicyNet (outputs logits)
    # argmax on either gives the greedy/deterministic deployment action
    s.clear(); cands = []; bases = []; net.eval()
    for u in uord_fn(s):
        if filt == 'none':
            th = s.gt * 1.5
        elif filt == 'nospeed':
            # FIX: remove vehicle penalty only (same fix as _candidates)
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




# ──────────────────────────────────────────────────────────────
# PHASE 1: TRAIN ALL DRL AGENTS
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: Training DRL agents (45 eps, seed=42)")
print("=" * 60)

if _P13_ONLY:
    if not _CKPT.exists():
        print("  WARNING: --phase13 set but no checkpoint found.")
        print("  Run without --phase13 first to train and save agents.")
        import sys; sys.exit(1)
    agents = _load_checkpoint()
    curves = {}
    print("  [SKIP] Phase 1 — loaded from checkpoint")
else:
    ABL_CONFIGS = {
        'w/o Interest Filter':  (b_cw_score, 'none',    'pri'),
        'w/o Speed Constraint': (b_cw_score, 'nospeed', 'pri'),
        'w/o Priority Order':   (b_cw_score, 'full',    'rand'),
        'w/o CW Score (DQN+Pop)':(b_pop,    'full',    'pri'),
    }
    
    uord_map = {'pri': o_pri, 'rand': o_rand}
    agents  = {}
    curves  = {}
    
    TRAINER_MAP = {'a2c': train_a2c, 'ppo': train_ppo, 'dqn': train_dqn}
    
    for name, (bfn, filt, uo, ctrl) in tqdm(AGENT_CONFIGS.items(),
                                              desc='Phase 1 agents', ncols=80):
        t0 = time.time()
        s  = Sim(250, .6, 1000, 80, 1, seed=GLOBAL_SEED)
        net, rews = TRAINER_MAP[ctrl](s, 45, bfn, filt, uord_map[uo],
                                       seed=GLOBAL_SEED, desc=f'{name} [{ctrl.upper()}]')
        agents[name] = (net, bfn, filt, uord_map[uo])
        curves[name] = rews
        tqdm.write(f"  {name} [{ctrl.upper()}]: {time.time()-t0:.0f}s  off={max(rews)/300:.4f}")
    
    # Controller comparison: DQN and PPO with CW score base
    print("\nTraining DQN and PPO (CW score base, for controller comparison)...")
    for ctrl_name, trainer_fn in tqdm([('DQN (CW score)', train_dqn), ('PPO', train_ppo)],
                                        desc='Phase 1 ctrl', ncols=80):
        t0 = time.time()
        s  = Sim(250, .6, 1000, 80, 1, seed=GLOBAL_SEED)
        net, rews = trainer_fn(s, 45, b_cw_score, 'full', o_pri,
                                seed=GLOBAL_SEED, desc=ctrl_name)
        agents[ctrl_name] = (net, b_cw_score, 'full', o_pri)
        curves[ctrl_name] = rews
        tqdm.write(f"  {ctrl_name}: {time.time()-t0:.0f}s")
    
    # RL-CWCA with aligned threshold (EXP 2)
    print("\nTraining RL-CWCA (aligned threshold 0.05)...")
    s = Sim(250, .6, 1000, 80, 1, seed=GLOBAL_SEED)
    net_aln, _ = train_a2c(s, 45, b_cw_score, 'aligned', o_pri,
                             seed=GLOBAL_SEED, desc='RL-CWCA-aligned')
    agents['RL-CWCA-aligned'] = (net_aln, b_cw_score, 'aligned', o_pri)
    
    # Ablation variants
    print("\nTraining ablation variants...")
    for name, (bfn, filt, uo) in tqdm(ABL_CONFIGS.items(),
                                        desc='Phase 1 ablation', ncols=80):
        s = Sim(250, .6, 1000, 80, 1, seed=GLOBAL_SEED)
        net, _ = train_a2c(s, 45, bfn, filt, uord_map[uo],
                            seed=GLOBAL_SEED, desc=name[:25])
        agents[name] = (net, bfn, filt, uord_map[uo])
        tqdm.write(f"  {name}: done")
    
    _rm.mark_phase(RUN_DIR, 'phase1_training',
                   {'agents_trained': list(agents.keys())})
    
    _save_checkpoint(agents)



# ──────────────────────────────────────────────────────────────
if not _P13_ONLY:
    # PHASE 2: MAIN EVALUATION  (3 same seeds)
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Main evaluation (3 seeds)")
    print("=" * 60)
    
    MAIN_ALGOS = ['RL-CWCA', 'DQN-Interest', 'DQN-Weak', 'DQN-Pop',
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
        tqdm.write(f"  {name}: off={np.mean(r['off']):.4f}±{np.std(r['off']):.4f}")
    
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
        tqdm.write(f"  {name}: {np.mean(r['off']):.4f}±{np.std(r['off']):.4f}")
    
    # Aligned threshold (EXP 2 — same seeds)
    r = defaultdict(list)
    for sd in tqdm(MAIN_SEEDS, desc='  Aligned thresh', leave=False, ncols=60):
        s = Sim(250, .6, 1000, 80, 1, seed=sd)
        net, bfn, filt, uo = agents['RL-CWCA-aligned']
        deploy(s, net, bfn, filt, uo)
        for k, v in s.exchange().items(): r[k].append(v)
    abl_res['Aligned\nThreshold'] = (np.mean(r['off']), np.std(r['off']))
    tqdm.write(f"  Aligned threshold: {np.mean(r['off']):.4f}±{np.std(r['off']):.4f}")
    for k, (m, sd) in abl_res.items():
        tqdm.write(f"  {k.replace(chr(10),' ')}: {m:.4f}±{sd:.4f}")
    
    _rm.mark_phase(RUN_DIR, 'phase4_ablation',
                   {k.replace('\n',' '): f'{v[0]:.4f}' for k,v in abl_res.items()})
    
    
    # ──────────────────────────────────────────────────────────────
    # PHASE 5: SENSITIVITY SWEEPS
    # ──────────────────────────────────────────────────────────────
    print("\n>>> Sensitivity sweeps")
    SHOW = ['RL-CWCA', 'DQN-Interest', 'CFCA', 'SAA', 'Greedy', 'Popular Cache']
    
    # Zipf γ
    ZIPF_VALS = [0.6, 0.8, 1.0, 1.2]
    zipf_res = {a: defaultdict(list) for a in SHOW}
    for gm in tqdm(ZIPF_VALS, desc='  Zipf γ sweep', ncols=80):
        for sd in MAIN_SEEDS:
            for name in SHOW:
                s = Sim(250, gm, 1000, 80, 1, seed=sd)
                if name in HEURISTICS: HEURISTICS[name](s)
                else:
                    net, bfn, filt, uo = agents[name]; deploy(s, net, bfn, filt, uo)
                zipf_res[name][gm].append(s.exchange()['off'])
    
    # File size
    FSIZES = [30, 70, 100, 150, 250]
    fsize_res = {a: [] for a in SHOW}
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
                                seed=GLOBAL_SEED, desc=f'HP ep={n_ep}')
        s2 = Sim(250, .6, 1000, 80, 1, seed=GLOBAL_SEED)
        deploy(s2, net_hp, b_cw_score, 'full', o_pri)
        hp_offs.append(s2.exchange()['off'])
        tqdm.write(f"  {n_ep} eps: {hp_offs[-1]:.4f}")
    
    # NNPM: net profit margin
    print(">>> NNPM computation")
    C_NET = 1.0
    nnpm_res = {a: [] for a in ['RL-CWCA', 'CFCA', 'SAA', 'Greedy', 'Popular Cache']}
    for cm in tqdm(CACHE_SIZES, desc='  NNPM', ncols=80):
        for name in nnpm_res:
            dd_m = np.mean(cache_res[name][cm]['dd'])
            cu_m = np.mean(cache_res[name][cm]['cu'])
            tr_m = np.mean(cache_res[name][cm]['tr'])
            lh_m = np.mean(cache_res[name][cm]['lh'])
            dh_m = np.mean(cache_res[name][cm]['dh'])
            rev  = dd_m
            cost = C_NET * (cu_m + max(tr_m - lh_m - dh_m, 0) * 0.5)
            nnpm_res[name].append((rev - cost) / max(rev, 1e-9) * 100)
    
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
    flib_res = {a: [] for a in ['RL-CWCA', 'CFCA']}
    for fl in tqdm(FLIB, desc='Phase 9 scalability', ncols=80):
        for name in flib_res:
            offs_fl = []
            for sd_fl in MAIN_SEEDS:
                s = Sim(fl, .6, 1000, 80, 1, seed=sd_fl)
                if name == 'CFCA': a_cfca(s)
                else:
                    net, bfn, filt, uo = agents[name]; deploy(s, net, bfn, filt, uo)
                offs_fl.append(s.exchange()['off'])
            flib_res[name].append(float(np.mean(offs_fl)))
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
        for name in tqdm(['RL-CWCA', 'DQN (CW score)', 'PPO', 'CFCA'],
                         desc='  controllers', leave=False, ncols=60):
            offs = []
            for sd in MAIN_SEEDS:
                s = Sim(250, .6, cm, 80, 1, seed=sd)
                if name == 'CFCA': a_cfca(s)
                else:
                    net, bfn, filt, uo = agents[name]; deploy(s, net, bfn, filt, uo)
                offs.append(s.exchange()['off'])
            ctrl_res[cm][name] = (np.mean(offs), np.std(offs))
            tqdm.write(f"  {name} @ {cm}MB: {np.mean(offs):.4f}±{np.std(offs):.4f}")
    _rm.mark_phase(RUN_DIR, 'phase10_controller', {'status': 'done'})
    
    
    # ──────────────────────────────────────────────────────────────
    # PHASE 11: GENERATE ALL FIGURES
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 11: Generating all figures")
    print("=" * 60)
    _fig_pbar = tqdm(total=21, desc='Phase 11 figures', ncols=80, unit='fig')
    
    # ── fig01: Training curves ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    w = 5
    drl_agents = ['RL-CWCA', 'DQN (CW score)', 'DQN-Interest', 'DQN-Weak', 'DQN-Pop']
    for nm in drl_agents:
        rw = curves[nm]
        sm = np.convolve(rw, np.ones(w)/w, 'valid')
        ax1.plot(range(w-1, len(rw)), sm, label=nm,
                 color=S[nm]['color'], lw=S[nm]['lw'], ls=S[nm]['ls'], zorder=S[nm]['z'])
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward (smoothed, w=5)')
    ax1.set_title('DRL Training Convergence')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)
    rw = curves['RL-CWCA']
    std_c = [np.std(rw[max(0, i-9):i+1]) for i in range(len(rw))]
    ax2.plot(std_c, color=S['RL-CWCA']['color'], lw=2.2)
    ax2.fill_between(range(len(std_c)), std_c, alpha=0.18, color=S['RL-CWCA']['color'])
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Reward Std Dev (w=10)')
    ax2.set_title('RL-CWCA Training Stability')
    plt.tight_layout(rect=[0, 0.15, 1, 1]); savefig(fig, 'fig01_training_curves')
    _fig_pbar.update(1)
    
    # ── fig02: Offloading vs Cache Size ──
    DRL_ORDER = ['RL-CWCA', 'DQN-Interest', 'DQN-Weak', 'DQN-Pop', 'CFCA', 'SAA']
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
    cnames = ['A2C\n(RL-CWCA)', 'DQN\n(CW score)', 'PPO', 'CFCA']
    cmap_  = {'A2C\n(RL-CWCA)': 'RL-CWCA', 'DQN\n(CW score)': 'DQN (CW score)', 'PPO': 'PPO', 'CFCA': 'CFCA'}
    x = np.arange(len(cnames)); bw = 0.36
    m1 = [ctrl_res[1000][cmap_[k]][0] for k in cnames]
    s1 = [ctrl_res[1000][cmap_[k]][1] for k in cnames]
    m2 = [ctrl_res[2000][cmap_[k]][0] for k in cnames]
    s2 = [ctrl_res[2000][cmap_[k]][1] for k in cnames]
    ccols = ['#E63946', '#17BECF', '#BCBD22', '#1D6FA4']
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
    ax.set_title('Controller Comparison: A2C vs DQN vs PPO')
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
    for nm in ['RL-CWCA', 'CFCA', 'SAA', 'Greedy', 'Popular Cache']:
        ax.plot(CACHE_SIZES, nnpm_res[nm], label=nm, **lp(nm))
    ax.axhline(0, color='gray', ls='--', alpha=0.5, lw=1)
    ax.set_xlabel('Cache Size (MB)'); ax.set_ylabel('Net Profit Margin (%)')
    ax.set_title('Network Normalised Profit Margin vs Cache Size')
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
                     'Offloading': f'{om:.4f}±{osd:.4f}',
                     'CHR':        f'{cm:.4f}±{csd:.4f}'})
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
    for nm in ['RL-CWCA','DQN (CW score)','PPO','CFCA']:
        m1,s1 = ctrl_res[1000][nm]; m2,s2 = ctrl_res[2000][nm]
        ctrl_rows.append({'Controller': nm,
                          'Off@1GB': f'{m1:.4f}±{s1:.4f}',
                          'Off@2GB': f'{m2:.4f}±{s2:.4f}'})
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
                f.write(f"- w/o Speed Constraint (fixed): offloading={m:.4f}±{sd:.4f}\n")
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
    MAIN_ALGOS  = ['RL-CWCA', 'DQN-Interest', 'DQN-Weak', 'DQN-Pop',
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

if _TDRIVE_NPZ.exists():
    print("\n" + "=" * 60)
    print("PHASE 13: Transfer validation (T-Drive, Beijing taxis)")
    print("=" * 60)

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

        Key difference from plain Sim:
            self_.pri is normalised to [0,1] via norm_pri().
            Raw value is kept in self_.pri_raw for diagnostics.

        Why: T-Drive has sparse contacts so raw π̄_u ≈ 0.01–0.08.
        The CW pull term (1 + 5*π̄_u) spans only 1.05–1.40 (1.33×)
        instead of Marylebone's 1.25–5.00 (4×).  Normalising restores
        discriminative power without changing the CW score formula.
        """
        def __init__(self_, F, gm, cmb, fmb, br, seed=42, coll_factor=1.0):
            rng = np.random.RandomState(seed)
            self_.N      = _ND_TD
            self_.F      = F
            self_.br     = br
            self_.avg_ct = _ACT_TD
            self_.coll   = coll_factor
            mx = _NCT_TD.max() if _NCT_TD.max() > 0 else 1
            self_.mp = _NCT_TD / mx
            np.fill_diagonal(self_.mp, 0)
            self_.ints = rng.dirichlet(np.ones(15), self_.N)
            mn  = np.minimum(self_.ints[:, None, :], self_.ints[None, :, :]).sum(2)
            mx2 = np.maximum(self_.ints[:, None, :], self_.ints[None, :, :]).sum(2)
            self_.isim = np.divide(mn, mx2, out=np.zeros_like(mn), where=mx2 > 0)
            np.fill_diagonal(self_.isim, 0)
            self_.vr   = np.array([1. if t == 'car' else 0. for t in _VT_TD])
            rk = np.arange(1, F + 1, dtype=float)
            rp = rk ** (-gm); self_.fp = rp / rp.sum()
            self_.fc   = np.arange(F) % 15
            self_.fs   = rng.uniform(max(10, fmb * .3), fmb * 2., F)
            v = _ACT_TD[_ACT_TD > 0]
            self_.gt   = br * np.mean(v) if len(v) > 0 else br * 30
            self_.cache = np.zeros((self_.N, F), bool)
            self_.cu    = np.zeros(self_.N)
            self_.tc    = np.full(self_.N, float(cmb))
            self_.comb  = self_.mp * self_.isim
            raw_pri     = self_.comb.mean(1)
            self_.pri_raw = raw_pri.copy()           # keep raw for comparison
            self_.pri     = norm_pri(raw_pri)        # ← normalised [0,1]

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

    # ── Fine-tune: 20 A2C episodes on T-Drive topology ──
    print("  Retraining RL-CWCA on T-Drive topology (20 eps fine-tune)...")
    _sc_ft = SimTDrive(200, .6, 1000, 60, 1, seed=GLOBAL_SEED)
    net_ft, _ = train_a2c(_sc_ft, 20, b_cw_score, 'full', o_pri,
                           seed=GLOBAL_SEED, desc='RL-CWCA-TDrive-FT')
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
          f"        ← retrained with norm_pri")

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

else:
    print("\n  [SKIP] Phase 13: T-Drive NPZ not found at", _TDRIVE_NPZ)
    print("  To run transfer validation (no login required, ~3 min):")
    print("    python fetch_tdrive.py \\")
    print("        --n-taxis 200 \\")
    print("        --out results/tdrive/tdrive_contact_data.npz")
    print("  Then re-run this script.")
    _rm.mark_phase(RUN_DIR, 'phase13_transfer_tdrive',
                   {'status': 'skipped — NPZ not found'})

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
