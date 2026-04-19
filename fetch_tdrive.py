#!/usr/bin/env python3
"""
fetch_tdrive.py
────────────────────────────────────────────────────────────────
Downloads the Microsoft T-Drive taxi GPS dataset, extracts it,
computes D2D contact events (pairs within 120 m simultaneously),
and saves an NPZ in the format expected by rlgaca_eval_updated.py.

Dataset
───────
  Yuan et al. (2010) "T-Drive: Driving Directions Based on Taxi
  Trajectories". ACM SIGSPATIAL GIS. 800+ citations.
  Published by Microsoft Research. No registration required.

  URL:  https://download.microsoft.com/download/F/4/8/
        F4894AA5-FDBC-481E-9285-D5F8C4C4F039/TaxiData.zip

  Content: GPS traces of 10,357 Beijing taxis, ~17 million points,
  one week of data (Feb 2 – Feb 8, 2008). Each file is one taxi.
  Format per line: taxi_id, timestamp, longitude, latitude

Contact model (matching the paper's SUMO model)
────────────────────────────────────────────────
  Two taxis are "in contact" when their Haversine distance < 120 m
  and the overlap duration >= 30 s (matches the 30% fragment
  threshold in the paper for a 100s mean contact window).

  We sample N=200 taxis and bin their traces to 30-second epochs
  over the full week, then compute pairwise proximity for each epoch.

Output
──────
  results/tdrive/tdrive_contact_data.npz
    avg_ct      (N, N) float32  mean contact duration per pair (s)
    n_ct        (N, N) int32    number of contact events per pair
    vtypes_arr  (N,)   object   all 'car' (taxis)

Usage
─────
  python fetch_tdrive.py [--n-taxis 200] [--out results/tdrive/tdrive_contact_data.npz]

  Typical runtime: ~3 min on a laptop (download + parse + proximity)
────────────────────────────────────────────────────────────────
"""
import argparse, io, os, sys, time, zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import urllib.request as _ur
except ImportError:
    import urllib.request as _ur

# ── Published T-Drive download URL (Microsoft Research) ─────────
TDRIVE_URL = (
    "https://download.microsoft.com/download/F/4/8/"
    "F4894AA5-FDBC-481E-9285-D5F8C4C4F039/TaxiData.zip"
)
RADIO_M   = 120.0    # D2D radio range in metres
MIN_DUR_S = 30.0     # minimum contact duration (seconds)
EPOCH_S   = 30       # time bin size in seconds


# ── Haversine distance (metres) ──────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def parse_args():
    p = argparse.ArgumentParser(description='Fetch T-Drive and build contact NPZ')
    p.add_argument('--n-taxis',  type=int, default=200,
                   help='Number of taxis to sample (default: 200)')
    p.add_argument('--out', default='results/tdrive/tdrive_contact_data.npz',
                   help='Output NPZ path')
    p.add_argument('--zip-cache', default='results/tdrive/TaxiData.zip',
                   help='Local path to cache the downloaded ZIP')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# ── Download with progress ────────────────────────────────────────
def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Using cached ZIP: {dest}")
        return
    print(f"  Downloading T-Drive from Microsoft Research (~35 MB)...")
    print(f"  URL: {url}")

    def _progress(block, bsize, total):
        done = block * bsize
        pct  = min(100, done * 100 // total) if total > 0 else 0
        bar  = '#' * (pct // 4) + '.' * (25 - pct // 4)
        sys.stdout.write(f"\r  [{bar}] {pct:3d}%  ({done//1024} KB)")
        sys.stdout.flush()

    try:
        _ur.urlretrieve(url, dest, reporthook=_progress)
    except Exception as e:
        print(f"\n\n  ERROR: Download failed — {e}")
        print("  The Microsoft Research URL is no longer active.")
        print("  Please download T-Drive manually from one of:")
        print("    https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/")
        print("    https://www.kaggle.com/datasets/arashnic/microsoft-research-t-drive-trajectory-data")
        print(f"  Then place the ZIP at: {dest}")
        sys.exit(1)
    print(f"\n  Saved: {dest}  ({dest.stat().st_size // 1024} KB)")


# ── Parse GPS files from ZIP ─────────────────────────────────────
def parse_zip(zip_path, n_taxis, seed):
    """
    Returns dict: taxi_id -> list of (epoch_seconds, lat, lon)
    All timestamps are normalised to seconds-since-dataset-start.
    """
    from datetime import datetime
    rng      = np.random.RandomState(seed)
    t0_ref   = None   # earliest timestamp across all taxis
    raw      = {}     # taxi_id -> [(ts, lat, lon)]

    print(f"  Parsing ZIP (sampling {n_taxis} taxis)...")
    with zipfile.ZipFile(zip_path) as zf:
        txt_files = [n for n in zf.namelist()
                     if n.endswith('.txt') and not n.startswith('__')]
        selected  = sorted(rng.choice(len(txt_files),
                                      min(n_taxis, len(txt_files)),
                                      replace=False))
        for file_idx in selected:
            fname = txt_files[file_idx]
            try:
                data = zf.read(fname).decode('utf-8', errors='ignore')
            except Exception:
                continue
            pts = []
            for line in data.splitlines():
                parts = line.strip().split(',')
                if len(parts) < 4:
                    continue
                try:
                    ts_str = parts[1].strip()
                    # T-Drive format: "2008-02-02 13:34:05"
                    ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                    lat = float(parts[3]); lon = float(parts[2])
                    # Basic Beijing bounding box check
                    if not (39.0 < lat < 41.0 and 115.0 < lon < 117.5):
                        continue
                    pts.append((ts, lat, lon))
                except (ValueError, IndexError):
                    continue
            if len(pts) < 10:
                continue
            pts.sort(key=lambda x: x[0])
            taxi_id = Path(fname).stem
            raw[taxi_id] = pts
            if t0_ref is None or pts[0][0] < t0_ref:
                t0_ref = pts[0][0]

    if t0_ref is None or not raw:
        raise RuntimeError("No valid GPS data found in ZIP.")

    # Convert to (epoch_int, lat, lon) with epoch = seconds since t0_ref
    parsed = {}
    for tid, pts in raw.items():
        parsed[tid] = [(int((t - t0_ref).total_seconds()), la, lo)
                       for (t, la, lo) in pts]

    print(f"  Parsed {len(parsed)} taxis with valid GPS data.")
    return parsed


# ── Bin to regular 30-s epochs ────────────────────────────────────
def bin_trajectories(parsed, epoch_s=30):
    """
    For each taxi, keep the last known position in each 30-s epoch.
    Returns dict: epoch -> dict: taxi_id -> (lat, lon)
    """
    from collections import defaultdict
    epoch_pos = defaultdict(dict)
    for tid, pts in parsed.items():
        for (ts, lat, lon) in pts:
            ep = ts // epoch_s
            epoch_pos[ep][tid] = (lat, lon)
    return epoch_pos


# ── Compute pairwise contacts ─────────────────────────────────────
def compute_contacts(epoch_pos, taxi_ids, radio_m, epoch_s, min_dur_s):
    """
    Finds contact events: pairs within radio_m for >= min_dur_s seconds.
    Returns list of (i, j, duration_s).
    """
    taxis  = list(taxi_ids)
    N      = len(taxis)
    t_idx  = {t: i for i, t in enumerate(taxis)}

    # Track open contacts: (i,j) -> start_epoch
    open_ct = {}
    contacts = []
    min_epochs = int(np.ceil(min_dur_s / epoch_s))

    epochs_sorted = sorted(epoch_pos.keys())
    total = len(epochs_sorted)
    print(f"  Computing pairwise proximity over {total:,} epochs "
          f"({total * epoch_s / 3600:.1f} h of data)...")

    for prog, ep in enumerate(epochs_sorted):
        pos_ep = epoch_pos[ep]
        present = [t for t in taxis if t in pos_ep]

        # For each pair both present in this epoch
        for k1 in range(len(present)):
            for k2 in range(k1 + 1, len(present)):
                ta, tb = present[k1], present[k2]
                la1, lo1 = pos_ep[ta]
                la2, lo2 = pos_ep[tb]
                dist = haversine(la1, lo1, la2, lo2)
                key  = (t_idx[ta], t_idx[tb])
                if dist <= radio_m:
                    if key not in open_ct:
                        open_ct[key] = ep
                else:
                    if key in open_ct:
                        dur_epochs = ep - open_ct.pop(key)
                        if dur_epochs >= min_epochs:
                            contacts.append((*key, dur_epochs * epoch_s))

        if (prog + 1) % 5000 == 0:
            sys.stdout.write(f"\r    {prog+1}/{total} epochs, "
                             f"{len(contacts):,} contacts found...")
            sys.stdout.flush()

    # Close any still-open contacts
    for (i, j), start_ep in open_ct.items():
        dur_epochs = epochs_sorted[-1] - start_ep
        if dur_epochs >= min_epochs:
            contacts.append((i, j, dur_epochs * epoch_s))

    print(f"\n  Total contact events: {len(contacts):,}")
    return contacts


# ── Build NPZ matrices ────────────────────────────────────────────
def build_matrices(N, contacts):
    dur_sum = defaultdict(float)
    cnt     = defaultdict(int)
    for i, j, dur in contacts:
        key = (min(i, j), max(i, j))
        dur_sum[key] += dur
        cnt[key]     += 1

    avg_ct = np.zeros((N, N), dtype=np.float32)
    n_ct   = np.zeros((N, N), dtype=np.int32)
    for (i, j), total in dur_sum.items():
        c = cnt[(i, j)]
        mean_dur = total / c
        avg_ct[i, j] = avg_ct[j, i] = mean_dur
        n_ct[i, j]   = n_ct[j, i]   = c

    return avg_ct, n_ct


def print_summary(N, avg_ct, n_ct):
    act = avg_ct[avg_ct > 0]
    print(f"\n  T-Drive contact trace summary:")
    print(f"    Unique taxis (N):    {N}")
    print(f"    Active pairs:        {(n_ct > 0).sum() // 2}")
    print(f"    Graph density:       {(n_ct>0).sum()/max(N*(N-1),1)*100:.1f}%")
    if len(act):
        print(f"    Mean duration:       {act.mean():.1f} s")
        print(f"    Median duration:     {np.median(act):.1f} s")
        print(f"    Max contacts/pair:   {n_ct.max()}")
    else:
        print("    WARNING: 0 contacts found.")
        print("    The sampled taxis may not overlap spatially.")
        print("    Try --n-taxis 400 or lower --radio-m threshold.")


def main():
    args = parse_args()

    print("=" * 60)
    print("T-Drive contact trace builder")
    print("  Dataset: Microsoft Research T-Drive (Yuan et al. 2010)")
    print("  No registration required.")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Download
    download(TDRIVE_URL, args.zip_cache)

    # Step 2: Parse
    parsed = parse_zip(args.zip_cache, args.n_taxis, args.seed)
    if not parsed:
        print("ERROR: No taxis parsed. Check ZIP format.")
        sys.exit(1)

    # Step 3: Bin
    epoch_pos = bin_trajectories(parsed, epoch_s=EPOCH_S)

    # Step 4: Contacts
    taxi_ids = list(parsed.keys())
    N        = len(taxi_ids)
    contacts = compute_contacts(epoch_pos, taxi_ids,
                                RADIO_M, EPOCH_S, MIN_DUR_S)

    if not contacts:
        print("\nWARNING: No contacts found with current parameters.")
        print("Try: --n-taxis 400")

    # Step 5: Matrices
    avg_ct, n_ct = build_matrices(N, contacts)
    vtypes = np.array(['car'] * N, dtype=object)   # taxis are vehicles
    print_summary(N, avg_ct, n_ct)

    # Step 6: Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, avg_ct=avg_ct, n_ct=n_ct, vtypes_arr=vtypes)
    print(f"\n  Saved: {out}")
    print(f"  Keys: avg_ct{avg_ct.shape}, n_ct{n_ct.shape}, "
          f"vtypes_arr{vtypes.shape}")
    print(f"\n  Total time: {time.time()-t0:.0f} s")
    print(f"\n  Next step — run Phase 13 in rlgaca_eval_updated.py:")
    print(f"    python rlgaca_eval_updated.py")


if __name__ == '__main__':
    main()
