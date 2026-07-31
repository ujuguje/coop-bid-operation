# Cooperative Bid and Operation Strategy in Two-Settlement Energy Market

Code, data, and results for:

> Seong, W., et al. (2026). **Cooperative Bid and Operation Strategy in
> Two-Settlement Energy Market through Dual-Agent Imitation Learning on Joint
> Optimized Policy.** *Applied Energy* (accepted, July 2026).

A solar + storage plant participates in a two-settlement (day-ahead bid /
real-time operation) electricity market. Two agents — a bidding agent and an
operation agent — are trained cooperatively by imitating a perfect-foresight
jointly optimized policy (OJPD), and benchmarked against single-agent SAC,
independent dual-agent SAC, rolling-horizon MPC, and the OJPD upper bound on
CAISO market data.

If you use this code, please cite the paper above.

## Repository layout (in data-flow order)

```
├─ data/
│   ├─ raw/                          ① Raw CAISO market data
│   │    ├─ FinalData_FMM_LMP.csv         15-min FMM LMP prices
│   │    ├─ Interval_LMP_15min_avg.csv    5-min LMP averaged to 15-min
│   │    └─ Output_Solar_NetLoad_15min.csv  15-min solar generation / net load
│   └─ processed/                    ② Processed data
│        ├─ rl_inputs/                    RL environment inputs (LSTM forecasts)
│        │    LMP_Bid_LSTM, LMP_Ope_LSTM, LMP_Set, Solar_Bid_LSTM, Solar_Ope
│        ├─ expert_actions/               OJPD expert demonstrations
│        │    Offline_Expert_Action_joint_deg{D}_tol{T}.csv (12 configurations)
│        │    2-column projection of optimization_results:
│        │      Bid_Action = B_Cha − B_Dis
│        │      Ope_Action = (O_Cha − O_Dis) − Bid_Action
│        │    The training code reads these files.
│        ├─ optimization_results/         Full MILP solutions (17 columns:
│        │    trades, SoC, profit decomposition). Source records for
│        │    expert_actions; kept for archival (30–40 min to regenerate each).
│        └─ legacy_inputs/                Pre-LSTM (persistence) inputs; used only
│                                         for row alignment in gen_sac_inputs_LSTM.py
├─ code/                             ③ Full pipeline
│   ├─ paths.py                          ★ All paths defined here (repo-relative)
│   ├─ Forecast/                         LSTM/Transformer/SARIMA forecast models
│   ├─ optimization/                     OJPD MILP + 4 MPC benchmark scripts
│   ├─ envs/                             Two-settlement market RL environments
│   ├─ models/, algorithms/              Networks / SAC·BC algorithms + eval_utils
│   ├─ data/settings.py                  Data loading, splits, battery parameters
│   ├─ run_training.py                   Single-SAC / Inde-DASAC training runner
│   ├─ run_coop_dasac.py                 Coop-DASAC (Case C) runner
│   └─ run_coop_dabc.py                  Coop-DABC (Case D, proposed) runner
├─ results/                          ④ Training & experiment outputs
│   ├─ baseline/                         Main checkpoints (10 seeds) + MPC cache
│   ├─ sensitivity/                      Sensitivity-sweep checkpoints (5 seeds)
│   ├─ coop_dasac/  coop_dabc/           Cooperative-method checkpoints
│   ├─ forecast/                         Forecast weights, predictions, benchmarks
│   └─ ood_analysis/                     OOD (Table 5) comparisons + MPC cache
├─ outputs/                          ⑤ Final paper artifacts (figures, tables)
└─ analysis/                         ⑥ Notebooks reproducing every table/figure
```

**Data flow**: `raw` → (Forecast/LSTM) → `processed/rl_inputs` → (MILP) →
`processed/expert_actions` → (RL training) → `results` → (analysis notebooks) →
`outputs`

## Reproducing the paper (notebooks B1–B5)

Open the notebooks in `analysis/` and run top to bottom. All five have been
execution-verified against the manuscript (B1·B2 match exactly, B3 within ±1
rounding), including from a fresh clone of this repository.

| Notebook | Paper artifact | Runtime |
|---|---|---|
| `B1_Table4_Main_Results` | **Table 4** (profit decomposition + Welch / one-sample t + separation) | ~1–3 min |
| `B2_Sensitivity_TableB1_B2` | **Tables B.1 / B.2** (sensitivity, 5 seeds) | ~3–5 min |
| `B3_OOD_Table5` | **Table 5** (out-of-distribution, 10 seeds) | ~5–15 min |
| `B4_Computational_Cost` | **Table B.3** (training / inference time) | ~5–10 min (GPU) |
| `B5_Paper_Figures` | **Figs 7, 9, B.1, B.2** → `outputs/` | ~2–4 min |

Auxiliary notebook: `A1` (Appendix C forecast-model comparison).

### Key facts (easy to get wrong)

1. **All final numbers use eval-mode (dropout OFF) re-evaluation.** Validation
   values in training logs evaluated SAC-family policies in train mode (a bug);
   notebooks B1–B3 (via `algorithms/eval_utils.py`) are the ground truth.
2. **Where the paper's methods live**: Coop-DASAC = `algorithms/coop_dasac.py`,
   Coop-DABC (proposed) = `algorithms/coop_dabc.py` (both with the monotonic
   QMIX hypernetwork fix, `w_final.abs()`).
3. **Checkpoint naming**: `actor_{tag}_deg{D}_tol{T}_seed{S}.pth`
   (tag: `single_sac` / `inde_sac` / `coop_dasac` / `coop_dabc`).
   In the matching `result_*.json` files, the `test_*` fields are eval-mode
   re-evaluations (the paper's standard); `val_returns` are the training-time
   monitoring curves that Figure 9 plots.
4. **Seeds**: main results (Tables 4, 5) use 10 seeds; sensitivity (B.1, B.2)
   uses 5. MPC and OJPD are deterministic. Single-SAC has no tolerance sweep
   (stated in the manuscript).
5. **Parameter rule**: defaults β_deg = 5 $/MWh, ε_tol = 5 MW. Before creating
   an environment, `Cap_Pcs_Ope = tol + 2` must be set —
   `eval_utils.set_cap_ope(tol)` handles this.
6. **Significance tests**: seeds are the sample. Welch two-sample vs stochastic
   benchmarks, one-sample vs deterministic MPC. All one-sided p < 0.001 with
   full separation (lowest Coop-DABC seed beats every benchmark's best run).

## Retraining and heavy-cache regeneration

| Target | How |
|---|---|
| Single-SAC / Inde-DASAC | `python code/run_training.py --algo single_sac --deg_cost 5.0 --tol 5.0 --seed 0` (same for `inde_sac`) |
| Coop-DASAC | `python code/run_coop_dasac.py --tol 5 --deg 5 --seed 0` |
| Coop-DABC (proposed) | `python code/run_coop_dabc.py --tol 5 --deg 5 --seed 0` |
| OJPD expert demonstrations | `python code/optimization/run_optimization.py --mode joint` → `data/processed/expert_actions/` |
| LSTM forecasts + RL inputs | notebook `A1` (training) → `python code/Forecast/gen_sac_inputs_LSTM.py` |
| MPC caches (30–40 min per config) | `code/optimization/mpc_components_{tol,deg}.py`, `mpc_partB.py`, `mpc_sensitivity.py` |

## Environment

```bash
pip install -r requirements.txt
```

CUDA GPU recommended. The MILP runs on cvxpy's default MI-capable solver.
All paths are repository-relative (`code/paths.py`); notebooks assume they are
opened from `analysis/`.

## License

Released under the [MIT License](LICENSE).
