#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 02:26:39 2025

@author: lnc
"""

#!/usr/bin/env python3
"""
Grid-search MAGNIFICENT_MACARONS parameters in round3final.py.

Iterated parameters
-------------------
* csi_threshold → [0.3, 0.4, 0.5, 0.6, 0.7]  # Critical Sunlight Index
* sunlight_factor → [1, 3, 5, 7, 9, 11, 13, 15]  # How strongly sunlight affects pricing
* min_conversion_size → [4, 8, 12]  # Minimum position size to trigger conversion
* take_width → [1, 2]  # Minimum edge for arb opportunities
* long_bias → [-0.5, -0.25, 0]  # Bias against long positions due to storage costs

For each combination the script:
  1) reads round3final.py,
  2) patches the lines within "Product.MAGNIFICENT_MACARONS: { ... }"
     so that parameters get the desired values,
  3) saves that as a unique round3final_mm_{i}.py,
  4) runs:
       prosperity3bt round3final_mm_{i}.py 4 --merge-pnl --match-trades worse
  5) extracts "Total profit: x" from the final line,
  6) appends a row (parameters and profit)
     to grid_results_macarons.csv
  7) repeats for the entire parameter grid.

Usage
-----
  python3 gridsearch_macarons.py
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

# Fixed parameters
make_edge = 2
make_probability = 0.800
storage_cost = 0.1
position_limit = 75  # Added position_limit parameter

# Grid search parameters
csi_threshold = [0.3, 0.4, 0.5, 0.6, 0.7]  # Critical Sunlight Index
sunlight_factor = [10]  # How strongly sunlight affects pricing
min_conversion_size = [8]  # Minimum position size to trigger conversion
take_width = [0.5,1, 2,3,4,5]  # Minimum edge for arb opportunities
long_bias = [-0.5]  # Bias against long positions due to storage costs

# The param names in your code for MAGNIFICENT_MACARONS
PARAM_KEYS = ("make_edge", "make_probability", "storage_cost", "csi_threshold", 
              "sunlight_factor", "min_conversion_size", "take_width", "long_bias", "position_limit")

# ─── 2) REGEXS ────────────────────────────────────────────────────────────

# Example final lines might say: "Total profit:  123,456"
PROFIT_RE = re.compile(r"Total profit:\s*([\-\d,]+)")

# We match JSON-like "key": 123.45 or "key": 2.0
KEY_VAL_RE = r"[\"']{}[\"']\s*:\s*[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?"

# ──────────────────────────────────────────────────────────────────────────
def locate_macarons_block(src: str) -> tuple[int, int]:
    """
    Return (start, end) substring indices for the 'Product.MAGNIFICENT_MACARONS: { ... }'
    param block in the source code, or raise if not found.
    """
    # We look for Product.MAGNIFICENT_MACARONS: {
    pattern = r"Product\.MAGNIFICENT_MACARONS\s*:\s*\{"
    match = re.search(pattern, src)
    if not match:
        raise ValueError("Product.MAGNIFICENT_MACARONS block not found")
    
    start = match.start()
    
    # Parse forward until we match the closing brace
    depth = 0
    for i, ch in enumerate(src[match.end():], start=match.end()):
        if ch == '{':
            depth += 1
        elif ch == '}':
            if depth == 0:
                end = i + 1
                return start, end
            depth -= 1
    
    raise ValueError("Could not find closing brace for MAGNIFICENT_MACARONS block")


def patch_source(src: str, params: dict) -> str:
    """
    For the 'Product.MAGNIFICENT_MACARONS: { ... }' block, re-substitute lines
    for all the parameters in the params dict
    """
    try:
        start, end = locate_macarons_block(src)
        block = src[start:end]
        
        # Extract just the parameter block
        param_block_start = block.find('{')
        param_block = block[param_block_start:]
        block_prefix = block[:param_block_start]
        
        # Create a new parameter block with our desired values
        new_param_block = "{\n"
        for key, val in params.items():
            new_param_block += f'        "{key}": {val},\n'
        new_param_block += "    }"
        
        # Combine and return the updated source
        return src[:start] + block_prefix + new_param_block + src[end:]
    
    except ValueError as e:
        # If we didn't find the MAGNIFICENT_MACARONS block, we need to add it
        # First locate the PARAMS dictionary
        params_match = re.search(r"PARAMS\s*=\s*\{", src)
        if not params_match:
            raise ValueError("Could not find PARAMS dictionary")
        
        # Find the end of the PARAMS dictionary
        params_start = params_match.end()
        depth = 1  # We're inside the first brace already
        for i, ch in enumerate(src[params_start:], start=params_start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    # We found the end of PARAMS
                    params_end = i
                    
                    # Add our new product block
                    new_block = '\n    Product.MAGNIFICENT_MACARONS: {\n'
                    for key, val in params.items():
                        new_block += f'        "{key}": {val},\n'
                    new_block += '    },\n'
                    
                    return src[:params_end] + new_block + src[params_end:]
        
        raise ValueError("Could not find end of PARAMS dictionary")


def run_backtest(pyfile: Path) -> float | None:
    """
    Launch the backtester with:
      prosperity3bt <pyfile> 4 --merge-pnl --match-trades worse
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

    # Get the very last number which should be the total
    final_str = matches[-1].replace(",", "")
    return float(final_str)


def main() -> None:
    # Generate all combinations of parameter values
    param_grid = list(itertools.product(
        csi_threshold, 
        sunlight_factor, 
        min_conversion_size, 
        take_width, 
        long_bias
    ))
    
    print(f"Running grid search with {len(param_grid)} parameter combinations")
    
    original_src = TRADER_SRC.read_text()
    results = []
    
    for i, (csi, sf, mcs, tw, lb) in enumerate(param_grid, 1):
        # Create parameter dictionary with fixed and variable parameters
        params = {
            "make_edge": make_edge,
            "make_probability": make_probability,
            "storage_cost": storage_cost,
            "csi_threshold": csi,
            "sunlight_factor": sf,
            "min_conversion_size": mcs,
            "take_width": tw,
            "long_bias": lb,
            "position_limit": position_limit
        }
        
        try:
            # Patch the source code with our new parameters
            new_src = patch_source(original_src, params)
            
            # Write temporary file
            tmpfile = TRADER_SRC.parent / f"round3final_mm_{i}.py"
            tmpfile.write_text(new_src)
            
            # Run backtest
            print(f"Running {i}/{len(param_grid)}: csi={csi}, sf={sf}, mcs={mcs}, tw={tw}, lb={lb}")
            profit = run_backtest(tmpfile)
            
            # Store results
            result = {
                "csi_threshold": csi,
                "sunlight_factor": sf,
                "min_conversion_size": mcs,
                "take_width": tw,
                "long_bias": lb,
                "profit": profit
            }
            results.append(result)
            print(f"  Profit: {profit}")
            
            # Save intermediate results after each run
            pd.DataFrame(results).to_csv("grid_results_macarons.csv", index=False)
            
            # Clean up temporary file
            try:
                tmpfile.unlink()
            except OSError:
                pass
                
        except Exception as e:
            print(f"Error with combination {i}: {e}")
    
    # Final results processing
    df = pd.DataFrame(results)
    df.to_csv("grid_results_macarons.csv", index=False)
    
    # Show best combinations
    if not df.empty:
        best = df.dropna(subset=["profit"]).sort_values("profit", ascending=False).head(10)
        print("\nTop 10 parameter combinations for MAGNIFICENT_MACARONS:")
        print(best.to_string(index=False))
        
        # Extract best parameters
        if len(best) > 0:
            best_params = best.iloc[0]
            print("\nBest parameters:")
            print(f"csi_threshold = {best_params['csi_threshold']}")
            print(f"sunlight_factor = {best_params['sunlight_factor']}")
            print(f"min_conversion_size = {best_params['min_conversion_size']}")
            print(f"take_width = {best_params['take_width']}")
            print(f"long_bias = {best_params['long_bias']}")
            print(f"Profit = {best_params['profit']}")


if __name__ == "__main__":
    main()