#!/usr/bin/env python3
"""
Grid‑search wrapper for prosperity3bt.
--------------------------------------
• Adjust BASKET_PATH, DATASET_ID and PARAM GRIDS as needed.
• Requires pandas.
"""

import itertools, json, subprocess, re, pathlib, pandas as pd, shlex

# --- edit these ----------------------------------------------------------------
BASKET_PATH = pathlib.Path("trader/basket.py").resolve()
DATASET_ID  = 2                               # ← prosperity3 round/day id
windows     = [150, 300, 450]
kentries    = [1.5, 2.0, 2.5]
sizes       = [40, 60, 80]
# --------------------------------------------------------------------------------

def run_bt(win, k, size):
    kexit = 0.4 * k
    cmd = (
        f"prosperity3bt {shlex.quote(str(BASKET_PATH))} {DATASET_ID} "
        f"--merge-pnl --match-trades worse "
        f"--window {win} --kentry {k} --kexit {kexit:.2f} --size {size}"
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip())

    # Prosperity prints a JSON line starting with SUMMARY: {...}
    m = re.search(r"SUMMARY:\s*(\{.*\})", proc.stdout)
    if not m:
        raise RuntimeError("Could not find SUMMARY line in output.")
    metrics = json.loads(m.group(1))
    metrics.update(window=win, k_entry=k, k_exit=kexit, size=size)
    return metrics

records = []
for win, k, sz in itertools.product(windows, kentries, sizes):
    print(f"Running win={win} k={k} size={sz} ...", end=" ", flush=True)
    rec = run_bt(win, k, sz)
    records.append(rec)
    print("done.")

df = pd.DataFrame(records)
df.sort_values("final_pnl", ascending=False, inplace=True)
print("\nTop 10 parameter sets:")
print(df.head(10).to_string(index=False))

out = pathlib.Path("grid_results.csv")
df.to_csv(out, index=False)
print(f"\nFull results saved to {out}")
