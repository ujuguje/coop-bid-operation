"""Single source of truth for every path in this repository.

All code and notebooks derive locations from the repo root (the parent of
code/), so the repository is fully relocatable — no absolute paths anywhere.

Layout:
    data/raw/                    original CAISO market data (LMP, solar/net-load)
    data/processed/rl_inputs/            LSTM-forecast RL environment inputs
    data/processed/expert_actions/       OJPD expert demonstrations (MILP)
    data/processed/optimization_results/ full MILP solution variables
    data/processed/legacy_inputs/        pre-LSTM (persistence) inputs, kept for
                                         row alignment in gen_sac_inputs_LSTM.py
    results/                     training outputs (checkpoints, caches, forecasts)
    outputs/                     final paper artifacts (figures, tables)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW      = os.path.join(ROOT, "data", "raw")
RL_INPUTS     = os.path.join(ROOT, "data", "processed", "rl_inputs")
EXPERT_DIR    = os.path.join(ROOT, "data", "processed", "expert_actions")
OPT_RESULTS   = os.path.join(ROOT, "data", "processed", "optimization_results")
LEGACY_INPUTS = os.path.join(ROOT, "data", "processed", "legacy_inputs")

RESULTS       = os.path.join(ROOT, "results")
RES_BASELINE  = os.path.join(RESULTS, "baseline")
RES_SENS      = os.path.join(RESULTS, "sensitivity")
RES_DASAC      = os.path.join(RESULTS, "coop_dasac")
RES_DABC      = os.path.join(RESULTS, "coop_dabc")
RES_FORECAST  = os.path.join(RESULTS, "forecast")
RES_OOD       = os.path.join(RESULTS, "ood_analysis")

OUTPUTS       = os.path.join(ROOT, "outputs")
