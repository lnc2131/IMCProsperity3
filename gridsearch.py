#!/usr/bin/env python3
"""
Grid‑search **DJEMBES** Bollinger‑band parameters in *Round2v6.py*.

Iterated parameters
-------------------
* ``bb_window``   → [700, 800, 900]   # example
* ``bb_k_entry`` → [3.8, 4.0, 4.2, 4.6]
* ``bb_k_exit``  → [3.2, 3.4, 3.6, 3.8, 4.0]

For each combination the script:
  1) reads Round2v6.py,
  2) patches the lines within "Product.DJEMBES: { ... }"
     so that bb_window, bb_k_entry, bb_k_exit get the desired values,
  3) saves that as a unique Round2v6_dj_{i}.py,
  4) runs:
       prosperity3bt Round2v6_dj_{i}.py 2 --merge-pnl --match-trades worse
  5) extracts "Total profit: x" from the final line,
  6) appends a row (bb_window, bb_k_entry, bb_k_exit, profit)
     to grid_results_djembes.csv
  7) repeats for the entire parameter grid.

Usage
-----
  python3 gridsearch_djembes.py
"""

import itertools
import os
import re
import subprocess
from pathlib import Path
from textwrap import shorten

import pandas as pd

# ─── 1) PATHS & GRIDS ─────────────────────────────────────────────────────

TRADER_SRC   = Path("/Users/lnc/acad/Prosperity /KINGLUCAS_/Round_2/trader/round3final.py")
BACKTEST_EXE = "prosperity3bt"
DATA_ROUND   = "5"

make_edge = 2
make_probability = 0.800
csi_threshold = [0.3, 0.4, 0.5, 0.6, 0.7]  # Critical Sunlight Index
sunlight_factor = [1,3,5,7,9,11,13,15]  # How strongly sunlight affects pricing
min_conversion_size = [4,8,12]  # Minimum position size to trigger conversion
take_width = [1,2]        # Minimum edge for arb opportunities
long_bias = [-0.5,-0.25,0]


# The param names in your code for DJEMBES
PARAM_KEYS   = ("make_edge", "make_probability", "csi_threshold", "sunlight_factor", "min_conversion_size", "take_width", "long_bias" )

# ─── 2) REGEXS ────────────────────────────────────────────────────────────

# Example final lines might say: "Total profit:  123,456"
PROFIT_RE    = re.compile(r"Total profit:\s*([\-\d,]+)")

# We match JSON-like "bb_window": 123.45 or "bb_k_entry": 2.0
KEY_VAL_RE   = r"[\"']{}[\"']\s*:\s*[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?"

# ──────────────────────────────────────────────────────────────────────────
def locate_djembes_block(src: str) -> tuple[int,int]:
    """
    Return (start, end) substring indices for the 'Product.DJEMBES: { ... }'
    param block in the source code, or raise if not found.
    """
    # We look for something like:  Product.DJEMBES: {
    pattern = r"\s*:\s*\{"
    for m in re.finditer(pattern, src):
        # Once we find '{', parse forward until we match '}' 
        depth = 0
        endpos = None
        for i, ch in enumerate(src[m.end():], start=m.end()):
            if ch == '{':
                depth += 1
            elif ch == '}':
                if depth == 0:
                    endpos = i + 1
                    break
                depth -= 1
        if endpos is None:
            # maybe there's unbalanced braces or we found partial. keep searching
            continue
        block_text = src[m.start():endpos]
        # Check that each param key is in that block (like "bb_window")
        if all(k in block_text for k in PARAM_KEYS):
            return m.start(), endpos
    raise ValueError("Param block for Product.DJEMBES not found or missing required keys")


def patch_source(src: str, w: int, k_ent: float, k_exit: float) -> str:
    """
    For the 'Product.DJEMBES: { ... }' block, re‑substitute lines for
    bb_window, bb_k_entry, bb_k_exit, e.g.  '"bb_window": 123'
    """
    start, end = locate_djembes_block(src)
    block = src[start:end]

    # For each param key, do a single re.sub
    for key, val in zip(PARAM_KEYS, (w, k_ent, k_exit)):
        pat = KEY_VAL_RE.format(key)
        new_string = f'"{key}": {val}'
        block, n = re.subn(pat, new_string, block, 1)
        if n == 0:
            raise ValueError(f"Key {key} not found in the DJEMBES block, or unsubstitutable.")
    
    return src[:start] + block + src[end:]


def run_backtest(pyfile: Path) -> float | None:
    """
    Launch the backtester with:
      prosperity3bt <pyfile> 2 --merge-pnl --match-trades worse
    and parse 'Total profit: X' from stdout 
    """
    cmd = [BACKTEST_EXE, str(pyfile), DATA_ROUND, "--merge-pnl", "--match-trades", "worse"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("❌  CLI error →", shorten(proc.stderr or proc.stdout, 150))
        return None
    
    matches = PROFIT_RE.findall(proc.stdout)
    if not matches:
        # No 'Total profit:' lines found
        snippet = shorten(proc.stdout.replace('\n', ' '), 200)
        print("⚠️  Could not find 'Total profit:' in output. Example:", snippet)
        return None

    # Typically the last line is the final day total
    final_str = matches[-1].replace(",", "")
    return float(final_str)


def main() -> None:
    combos = list(itertools.product(window, z_entry, z_exit))
    original_src = TRADER_SRC.read_text()

    results = []
    for i, (w, ent, xit) in enumerate(combos, 1):
        try:
            new_src = patch_source(original_src, w, ent, xit)
        except ValueError as e:
            print(f"{i}/{len(combos)} Patch fail:", e)
            continue
        
        # Write a throwaway file
        tmpfile = TRADER_SRC.parent / f"Round2v6_dj_{i}.py"
        tmpfile.write_text(new_src)

        # Run
        profit = run_backtest(tmpfile)
        print(f"{i}/{len(combos)} w={w}, k_entry={ent}, k_exit={xit} => {profit}")
        results.append(dict(bb_window=w, bb_k_entry=ent, bb_k_exit=xit, profit=profit))

        # Clean up
        try:
            tmpfile.unlink()
        except OSError:
            pass

    # Summarize
    df = pd.DataFrame(results)
    df.to_csv("grid_results_djembes.csv", index=False)
    best = df.dropna(subset=["profit"]).sort_values("profit", ascending=False).head(10)
    print("\nTop 10 combos for DJEMBES:")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
