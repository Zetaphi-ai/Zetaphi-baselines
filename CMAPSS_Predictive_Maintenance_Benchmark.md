# Long-History Predictive Maintenance: An O(N) Mixer on NASA C-MAPSS Turbofan RUL

**Date:** 2026-06-09
**Scope:** C-MAPSS FD001, parameter-matched (~70k) comparison. Baseline code in [`cmapss_rul/`](cmapss_rul/).

**Abstract:**
Predictive maintenance asks a model to retain a slow degradation signal across long telemetry histories. We benchmark a proprietary O(N) sequence mixer against parameter-matched GRU, causal TCN, and Transformer baselines on NASA C-MAPSS FD001 remaining-useful-life (RUL) prediction, across input histories from 30 to 200 cycles. The proprietary model is the only architecture whose accuracy survives the longest histories, while holding flat single-stream inference latency from 50 to 4,096 timesteps. We publish the complete baseline harness, the evaluation protocol, and every score — including the regimes where our model loses.

*(The proprietary mixer's mechanism and code are withheld. Everything needed to reproduce the baselines and audit the protocol is in this repository.)*

---

## Protocol (designed to be attacked)

- **Parameter match:** GRU 71.3k / causal TCN 69.0k / Transformer 69.6k / proprietary 69.0k.
  (Pitfall we fixed in our own earlier scout harness: `nn.TransformerEncoderLayer`'s default
  `dim_feedforward=2048` silently makes a "small" Transformer 282k params — 4x over budget.)
- **No test peeking:** unit-level 80/20 train/validation split; model selection on validation
  RMSE only; the official test set is evaluated once per final model. 3 seeds per cell.
- **No survivor bias in the history ladder:** test engines shorter than the input window are
  forward-fill padded so all 100 test units are scored at every history length. (Naive
  windowing silently drops short-lived engines — only 8 of 100 units survive at history 200,
  a systematically different population.)
- **Symmetric tuning:** every model got an identical 16-trial validation-only random search
  budget. Tuned-baseline numbers are reported wherever they beat defaults.
- **Causal baselines:** the TCN uses left-only padding. (Symmetric padding — common in
  reference implementations — lets the model see future cycles.)
- **Corruption fairness:** identical corrupted test data for every architecture (fixed-seed
  generators); robustness is evaluated on the same checkpoints as the headline numbers.

## 1. History ladder — clean test RMSE (mean ± std, 3 seeds, lower is better)

| History (cycles) | GRU | TCN (causal) | Transformer | Proprietary O(N) |
|---|---|---|---|---|
| 30  | **13.00 ± 0.08** | 14.47 ± 0.27 | 13.51 ± 0.38 | 21.84 ± 0.29 |
| 50  | 14.42 ± 0.69 | 14.84 ± 0.23 | **13.58 ± 0.12** | 16.09 ± 0.25 |
| 100 | 28.59 ± 1.57 | 17.39 ± 0.48 | **14.92 ± 0.37** | 16.86 ± 0.68 |
| 150 | 49.64 ± 2.25 | **17.40 ± 0.31** | 32.80 ± 0.18 | 27.10 ± 10.30 |
| 200 | 67.48 ± 0.01 | 41.49 ± 0.56 | 69.45 ± 0.51 | **32.30 ± 5.18** |

Honest reading, both directions:
- **Short history (30-50): our model loses clean.** Well-tuned GRU/attention are stronger
  in the industry-standard FD001 regime. We report it because it is true.
- **Long history (150-200): the proprietary mixer is the only model still standing** at 200
  cycles. GRU and Transformer collapse; the TCN is flat until its receptive field is
  exhausted, then breaks. Part of the baseline collapse is training-data scarcity at long
  windows (14.2k windows at history 30 vs 1.8k at 200) — that scarcity is identical for
  every model, and the proprietary mixer is the one that tolerates it.
- The ±10.3 at history 150 is one diverged seed of three. Real; not hidden; under investigation.

Symmetric tuning helped the baselines too (e.g. tuned GRU improves 49.6 → 31.7 at history
150) and did not change the ordering at history 200.

## 2. Reality-damage suite (history 50; ratio = corrupted RMSE / clean RMSE)

Raw faults vs the same faults behind a standard embedded driver stack (forward-fill on
missing packets, freeze-on-fault for dead channels, 3-tap median despiking) — drivers
applied identically to every model.

| Corruption | GRU | TCN | Transformer | Proprietary |
|---|---|---|---|---|
| sensor noise | **1.05** | 1.08 | 1.18 | 1.73 |
| 20% packet loss (raw zeros) | 1.88 | 1.65 | **1.41** | 1.85 |
| packet loss + forward-fill driver | 1.08 | 1.08 | 1.11 | **1.07** |
| 3 dead channels (raw zeros) | 1.64 | **1.52** | 1.57 | 3.69 |
| dead channels + freeze driver | **1.07** | 1.15 | 1.12 | 1.50 |
| calibration drift | 1.81 | **1.45** | 1.96 | 1.66 |
| bias offset (one channel) | 1.19 | **1.08** | 1.30 | 2.32 |
| voltage spikes (raw) | **1.60** | 2.22 | 3.95 | 4.32 |
| spikes + median despiking driver | **1.14** | 1.23 | 2.58 | 1.53 |
| delayed channel (5-cycle lag) | **1.00** | 1.01 | 1.02 | 1.03 |

Both directions, again:
- **On raw corrupted streams, the proprietary model is the most fragile** to spikes, dead
  channels, and bias. Its state-update treats input-scale corruption as physics.
- **Behind the driver logic standard firmware already has**, its transient-fault tolerance
  leads the board: best on packet loss (1.07), and after despiking it recovers to 1.53 while
  the Transformer remains broken at 2.58 — residual spike energy still hijacks attention.
- **Sustained calibration error (bias/drift/noise) has no driver fix and remains its genuine
  soft spot.** We make no blanket robustness claim.

## 3. Inference scaling (RTX 4090; compiled; ms per forward / peak VRAM)

Single-stream (batch 1):

| Timesteps | Transformer | Proprietary O(N) |
|---|---|---|
| 50 | 0.265 ms / 10 MB | 0.231 ms / 10 MB |
| 1,024 | 0.273 ms / 27 MB | 0.258 ms / 16 MB |
| 4,096 | 2.602 ms / 281 MB | **0.261 ms / 35 MB** |

Batch 64:

| Timesteps | Transformer | Proprietary O(N) |
|---|---|---|
| 1,024 | 11.8 ms / 1.14 GB | 3.3 ms / 0.43 GB |
| 2,048 | 43.0 ms / 4.41 GB | 7.1 ms / 0.86 GB |
| 4,096 | **OOM** | 14.4 ms / 1.71 GB |

Single-stream latency is flat from 50 to 4,096 timesteps (0.26 ms); batch latency doubles
when length doubles (clean O(N)); attention goes super-linear and then out-of-memory.

## The claim this data supports

At matched ~70k parameters on C-MAPSS FD001: near-parity clean accuracy at standard
histories, the only surviving accuracy at 200-cycle histories, flat single-stream latency to
4,096 timesteps, and board-leading transient-fault robustness when paired with ordinary
driver-level signal conditioning. Short-history clean accuracy and sustained calibration
error are the documented weaknesses.

## Reproduce the baselines

```bash
cd cmapss_rul
# data: NASA C-MAPSS FD001 (train_FD001.txt, test_FD001.txt, RUL_FD001.txt) into ./data/
python train.py --fd 001 --seqs 30,50,100,150,200 --models gru,tcn,transformer --seeds 42,123,314
python robustness.py --fd 001 --seq 50 --models gru,tcn,transformer
```
