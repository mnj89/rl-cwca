# Data Setup

## SUMO Contact Traces

The evaluation script (`full_eval.py`) requires pre-extracted contact data from a SUMO
traffic simulation of the London Marylebone area.

### Expected File

Place the following file at:

```
data/sumo_sim/contact_data.npz
```

### File Contents

The `.npz` archive must contain three arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `avg_ct` | `(N, N)` | Average contact duration (seconds) between each device pair |
| `n_ct` | `(N, N)` | Number of contacts between each device pair |
| `vtypes_arr` | `(N,)` | Device type string per device: `'car'`, `'bicycle'`, or `'pedestrian'` |

where `N` is the total number of devices in the simulation.

### Reproducing from SUMO

1. **Install SUMO** — https://eclipse.dev/sumo/

2. **Download the Marylebone OpenStreetMap**

   ```
   Area: -0.1256°N : 0.15°N, 51.5206°W : 51.5095°W
   ```

   Export via [openstreetmap.org](https://www.openstreetmap.org/) and convert with JOSM.

3. **Generate mobility traces** using NETEDIT with:
   - Pedestrians: avg speed 5 km/h
   - Bicycles: avg speed 15 km/h
   - Vehicles: avg speed 30 km/h
   - Simulation time: 1000 s
   - Radio range: 120 m

4. **Extract contacts** — For each timestep, detect device pairs within radio range
   and accumulate contact durations. Save as:

   ```python
   import numpy as np
   np.savez('data/sumo_sim/contact_data.npz',
            avg_ct=avg_ct,       # shape (N, N), float
            n_ct=n_ct,           # shape (N, N), int
            vtypes_arr=vtypes)   # shape (N,), str
   ```

### Map Reference

The simulation area corresponds to the map snapshot at `precomputed/figures/map.png`.

---

The city scenario used in the paper (Fig. 1) includes the Marylebone district with
pedestrians, cyclists, and vehicles registered under a single BS for proactive caching.
