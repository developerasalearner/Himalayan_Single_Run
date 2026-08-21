#!/usr/bin/env python3
"""Run every stage in order and write a run manifest.

    python scripts/run_all.py                 # reuse frozen feature sets
    python scripts/run_all.py --research      # re-run the stochastic search

Any config key can be overridden, e.g.:

    python scripts/run_all.py --set seed=7 --set compute.device=cpu
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import REPO_ROOT, build_parser, setup  # noqa: E402

STAGES = [
    ("01_build_dataset.py", []),
    ("02_baselines.py", []),
    ("03_feature_selection.py", []),
    ("04_imbalance.py", []),
    ("05_calibration_final.py", []),
    ("06_leakage_ablation.py", []),
]


def package_versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "imbalanced-learn",
                 "matplotlib", "seaborn", "pyarrow"):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:  # noqa: BLE001
            versions[name] = "not installed"
    return versions


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--research", action="store_true",
                        help="force a fresh feature search in stage 3")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()
    cfg, device = setup(args)

    common = []
    if args.config:
        common += ["--config", args.config]
    for override in args.overrides:
        common += ["--set", override]
    if args.no_cache:
        common.append("--no-cache")
    if args.quiet:
        common.append("--quiet")

    stages = list(STAGES)
    if args.skip_ablation:
        stages = [s for s in stages if s[0] != "06_leakage_ablation.py"]

    timings = {}
    started_at = datetime.now(timezone.utc).isoformat()

    for script, extra in stages:
        if script == "03_feature_selection.py" and args.research:
            extra = extra + ["--research"]

        command = [sys.executable, str(REPO_ROOT / "scripts" / script), *common, *extra]
        print(f"\n{'=' * 70}\n  {script}\n{'=' * 70}")

        started = time.time()
        result = subprocess.run(command, cwd=REPO_ROOT)
        timings[script] = round(time.time() - started, 2)

        if result.returncode != 0:
            print(f"\n{script} failed with exit code {result.returncode}")
            return result.returncode

    manifest = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "seed": cfg.get_dotted("seed"),
        "device": device,
        "termreason_excluded": cfg.get_dotted("leakage.exclude_termreason"),
        "stage_seconds": timings,
        "versions": package_versions(),
        "config": dict(cfg),
    }

    path = cfg.get_path("paths.results_dir") / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\nAll stages complete. Manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
