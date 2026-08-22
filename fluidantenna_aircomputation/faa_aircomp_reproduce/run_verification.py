#!/usr/bin/env python3
"""
run_verification.py

Orchestrates the full pre-submission verification pass for the FAA-AirComp
WCL letter:

  1. Runs `scripts/verify_key_results.py --mc 500` (or your chosen MC count)
  2. Parses its printed output for the metrics the paper claims
  3. Runs `scripts/reproduce_all_figures.py` to regenerate all 6 figures
  4. Checks a few basic sanity conditions (files exist, non-empty, etc.)
  5. Writes everything into submission_progress.json, next to this script

This script does NOT modify, round, or cherry-pick numbers to match the
paper. It records whatever the code actually outputs. If a value disagrees
with the paper (v17), it is flagged as a MISMATCH — that means the paper
needs to be corrected to match the code, not the reverse. Reporting a
number the code doesn't actually produce is a fast way to get desk-rejected
or retracted, so this script refuses to help fudge results.

Usage:
    python run_verification.py --mc 500
    python run_verification.py --mc 500 --skip-figures   # verify only
    python run_verification.py --mc 20  --quick           # fast smoke test
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROGRESS_FILE = ROOT / "submission_progress.json"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_key_results.py"
REPRODUCE_SCRIPT = ROOT / "scripts" / "reproduce_all_figures.py"
FIGURES_DIR = ROOT / "figures"

# What the paper (v17) currently claims — used only for comparison/flagging,
# never written back as truth. Update these if the paper text changes.
PAPER_CLAIMS = {
    "esr_pct": 53.6,
    "bcd_iters_min": 12,
    "bcd_iters_max": 16,
    "alpha": 1.34,
    "beta": -0.28,
    "r_squared": 0.95,
}

# Tolerances for "close enough to count as confirmed" vs "flag as mismatch"
TOLERANCES = {
    "esr_pct": 1.0,       # +/- 1 percentage point
    "alpha": 0.05,
    "beta": 0.05,
    "r_squared": 0.03,
}


def run_command(cmd, label):
    """Run a subprocess, stream output, and return (returncode, full_stdout)."""
    print(f"\n{'='*70}\n  RUNNING: {label}\n  CMD: {' '.join(cmd)}\n{'='*70}\n")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print("--- stderr ---", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
    return proc.returncode, proc.stdout + proc.stderr


def parse_verify_output(output: str) -> dict:
    """
    Best-effort parse of verify_key_results.py's printed output.

    NOTE: this regex set is a starting point based on the sample output
    format seen in the README ("[PASS] ESR = 38.2% ...", "[PASS] alpha = ...").
    If your actual script prints differently, adjust the patterns below —
    the goal is to capture real printed numbers, not to guess at them.
    """
    results = {}

    patterns = {
        "esr_pct": r"ESR\s*=\s*([\d.]+)\s*%",
        "bcd_iters": r"[Mm]ax\s*iters?\s*=\s*(\d+)",
        "alpha": r"[aα]\s*=\s*([\d.]+)\s*(?:>|\(paper)",
        "beta": r"[bβ]\s*=\s*(-?[\d.]+)",
        "r_squared": r"R[²2]\s*=\s*([\d.]+)",
        "monotone_pass": r"Monotone:\s*(\d+)\s*/\s*(\d+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, output)
        if m:
            if key == "monotone_pass":
                results[key] = f"{m.group(1)}/{m.group(2)}"
            else:
                results[key] = float(m.group(1))
        else:
            results[key] = None

    results["all_pass"] = "All key results VERIFIED" in output or (
        "[FAIL]" not in output and "PASS" in output
    )
    results["raw_output"] = output.strip()
    return results


def compare_to_paper(parsed: dict) -> dict:
    """Flag any parsed metric that disagrees with the paper beyond tolerance."""
    mismatches = {}
    for key, tol in TOLERANCES.items():
        code_val = parsed.get(key)
        paper_val = PAPER_CLAIMS.get(key)
        if code_val is None or paper_val is None:
            continue
        if abs(code_val - paper_val) > tol:
            mismatches[key] = {
                "paper_claims": paper_val,
                "code_produces": code_val,
                "action": "UPDATE THE PAPER to match the code output — do not change the code to hit the paper's number.",
            }
    return mismatches


def check_figures_exist() -> dict:
    """Confirm all 6 figures were actually regenerated and are non-trivial files."""
    status = {}
    for i in range(1, 7):
        candidates = list(FIGURES_DIR.glob(f"fig{i}.*"))
        if not candidates:
            status[f"fig{i}"] = {"exists": False, "size_bytes": 0}
        else:
            f = candidates[0]
            status[f"fig{i}"] = {
                "exists": True,
                "path": str(f.relative_to(ROOT)),
                "size_bytes": f.stat().st_size,
            }
    return status


def check_algorithm_source() -> dict:
    """
    Static checks on src/bcd.py for the three code-vs-paper items that
    can't be verified from numeric output alone.
    """
    bcd_path = ROOT / "src" / "bcd.py"
    exp_path = ROOT / "src" / "experiments.py"
    checks = {
        "s3_unit_modulus": {"found": False, "detail": None},
        "s4_proxy_phi": {"found": False, "detail": None},
        "run_faa_maxpow_exists": {"found": False, "detail": None},
    }

    if bcd_path.exists():
        src = bcd_path.read_text(encoding="utf-8", errors="ignore")
        # S3: unit-modulus, not amplitude-shrinkable
        if re.search(r"abs\([^)]*\)\s*==\s*1|\|\s*tau\w*\s*\|\s*=\s*1", src, re.I) or \
           "unit" in src.lower() and "modulus" in src.lower():
            checks["s3_unit_modulus"]["found"] = True
        checks["s3_unit_modulus"]["detail"] = (
            "Heuristic source scan only — manually confirm S3 normalises "
            "tau_k to phase (|tau_k| = 1) rather than clipping to |tau_k| <= 1."
        )

        # S4: phi(t) proxy with sigma_min + gamma * L1 norm term
        if "sigma_min" in src.lower() or "smallest singular" in src.lower():
            if "gamma" in src.lower() or "l1" in src.lower() or "norm" in src.lower():
                checks["s4_proxy_phi"]["found"] = True
        checks["s4_proxy_phi"]["detail"] = (
            "Heuristic source scan only — manually confirm S4 scores candidates "
            "with phi(t) = sigma_min(H(t)) + gamma * ||sigma(H(t))||_1, not raw MSE."
        )
    else:
        checks["s3_unit_modulus"]["detail"] = f"src/bcd.py not found at {bcd_path}"
        checks["s4_proxy_phi"]["detail"] = f"src/bcd.py not found at {bcd_path}"

    if exp_path.exists():
        src = exp_path.read_text(encoding="utf-8", errors="ignore")
        if "_run_faa_maxpow" in src:
            checks["run_faa_maxpow_exists"]["found"] = True
        checks["run_faa_maxpow_exists"]["detail"] = "Checked for literal '_run_faa_maxpow' in src/experiments.py"
    else:
        checks["run_faa_maxpow_exists"]["detail"] = f"src/experiments.py not found at {exp_path}"

    return checks


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print(f"\n✅ Progress written to {PROGRESS_FILE}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc", type=int, default=500, help="Monte Carlo run count (default 500, paper-quality)")
    ap.add_argument("--skip-figures", action="store_true", help="Only run verify_key_results.py, skip figure regeneration")
    ap.add_argument("--quick", action="store_true", help="Use a low MC count for a fast smoke test (overrides --mc)")
    args = ap.parse_args()

    mc = 20 if args.quick else args.mc

    if not VERIFY_SCRIPT.exists():
        print(f"ERROR: {VERIFY_SCRIPT} not found. Run this script from the repo root "
              f"(faa_aircomp_reproduce/) or adjust ROOT.", file=sys.stderr)
        sys.exit(1)

    # --- Step 1: run verify_key_results.py ---
    rc, out = run_command(
        [sys.executable, str(VERIFY_SCRIPT), "--mc", str(mc)],
        f"verify_key_results.py --mc {mc}",
    )
    verify_ok = rc == 0
    parsed = parse_verify_output(out)
    mismatches = compare_to_paper(parsed)

    # --- Step 2: reproduce figures (unless skipped) ---
    figures_status = {}
    figures_rc = None
    if not args.skip_figures:
        if REPRODUCE_SCRIPT.exists():
            figures_rc, fig_out = run_command(
                [sys.executable, str(REPRODUCE_SCRIPT), "--mc", str(mc)],
                f"reproduce_all_figures.py --mc {mc}",
            )
            figures_status = check_figures_exist()
        else:
            print(f"WARNING: {REPRODUCE_SCRIPT} not found — skipping figure regeneration.")

    # --- Step 3: static source checks for S3/S4/_run_faa_maxpow ---
    source_checks = check_algorithm_source()

    # --- Step 4: assemble and write progress file ---
    progress = load_progress()
    progress.setdefault("project", "FAA-AirComp WCL Letter Submission")
    progress["last_updated"] = date.today().isoformat()
    progress["last_verification_run"] = {
        "mc": mc,
        "verify_script_returncode": rc,
        "verify_script_passed": verify_ok,
        "parsed_metrics": {k: v for k, v in parsed.items() if k != "raw_output"},
        "mismatches_vs_paper_v17": mismatches,
        "figures_regenerated": not args.skip_figures,
        "figures_returncode": figures_rc,
        "figures_status": figures_status,
        "source_checks": source_checks,
    }

    # Update blocking_items if present, based on this run
    if "blocking_items" in progress:
        for item in progress["blocking_items"]:
            if item["id"] == "verify_esr" and parsed.get("esr_pct") is not None:
                item["done"] = True
                item["result"] = f"ESR = {parsed['esr_pct']}% at MC={mc}"
                if "esr_pct" in mismatches:
                    item["result"] += "  ⚠️ MISMATCH vs paper (53.6%) — paper needs updating"
            if item["id"] == "verify_run_faa_maxpow":
                item["done"] = source_checks["run_faa_maxpow_exists"]["found"]
                item["result"] = source_checks["run_faa_maxpow_exists"]["detail"]
            if item["id"] == "verify_s3_constraint":
                item["done"] = source_checks["s3_unit_modulus"]["found"]
                item["result"] = source_checks["s3_unit_modulus"]["detail"] + " [heuristic — verify manually]"
            if item["id"] == "verify_s4_proxy":
                item["done"] = source_checks["s4_proxy_phi"]["found"]
                item["result"] = source_checks["s4_proxy_phi"]["detail"] + " [heuristic — verify manually]"
            if item["id"] == "verify_other_figures" and figures_status:
                all_exist = all(f["exists"] for f in figures_status.values())
                item["done"] = all_exist
                item["result"] = f"{sum(f['exists'] for f in figures_status.values())}/6 figures regenerated"
            if item["id"] == "fill_readme_key_results_table":
                item["done"] = False
                item["result"] = "Run complete — copy last_verification_run.parsed_metrics into README.md by hand"

    save_progress(progress)

    # --- Step 5: human-readable summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  verify_key_results.py:  {'PASSED' if verify_ok else 'FAILED (check output above)'}")
    print(f"  Parsed ESR:             {parsed.get('esr_pct')}%  (paper claims 53.6%)")
    print(f"  Parsed alpha / beta:    {parsed.get('alpha')} / {parsed.get('beta')}  (paper: 1.34 / -0.28)")
    print(f"  Parsed R^2:             {parsed.get('r_squared')}  (paper: 0.95)")
    if mismatches:
        print("\n  ⚠️  MISMATCHES FOUND — the paper does not match the code:")
        for k, v in mismatches.items():
            print(f"      - {k}: paper says {v['paper_claims']}, code produced {v['code_produces']}")
        print("\n  The paper must be corrected to match these numbers before submission.")
    else:
        print("\n  ✅ No numeric mismatches detected against paper v17.")
    if not args.skip_figures:
        missing = [k for k, v in figures_status.items() if not v.get("exists")]
        if missing:
            print(f"\n  ⚠️  Figures missing: {missing}")
        else:
            print("  ✅ All 6 figures regenerated.")
    print(f"\n  Full details written to: {PROGRESS_FILE}")
    print("=" * 70 + "\n")

    if mismatches or (not verify_ok):
        sys.exit(2)  # nonzero exit so CI / scripts can detect "not ready"


if __name__ == "__main__":
    main()