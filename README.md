# ZetaPhi Baselines

Reproducible baseline code and benchmark scores from [ZetaPhi](https://zetaphi.ai)'s
work on linear-scaling architectures — attention replacements with O(N) sequence
mixing instead of the O(N²) cost of standard self-attention.

This repository exists so the baselines our models are compared against can be
independently reproduced. **It contains the baseline code (Transformer, ConvMixer)
and the measured scores. It does not contain ZetaPhi's proprietary architecture
code or mechanism — only its results.**

---

## CIFAR-100

Matched parameter budget (~0.45M), identical training recipe, 100 epochs, 3 seeds.
Metric: best top-1 eval accuracy over training, mean ± std.

| Model | Token mixing | Params | Top-1 acc | Code |
|---|---|---:|---:|:---:|
| Transformer (multi-head self-attention) | O(N²) | 700,773 | 56.23 ± 0.83 | ✅ public |
| ConvMixer (k=5, tuned) | O(N) | 455,781 | 56.57 ± 0.38 | ✅ public |
| **ZetaPhi mixer (ours)** | **O(N)** | **447,845** | **60.39 ± 0.33** | 🔒 proprietary |

At a matched ~0.45M-parameter budget, the ZetaPhi mixer outperforms a
parameter-matched Transformer by **+4.2 points** and a tuned ConvMixer baseline by
**+3.8 points**, using ~36% fewer parameters than the Transformer — while showing
the lowest seed-to-seed variance and the smallest train/eval overfitting gap of the
models tested.

---

## UCI HAR — continuous edge kinematics

50Hz 9-axis IMU streams, 6 activity classes, parameter-matched, 5 seeds.
Full report: [Robotics_Continuous_Kinematics_Benchmark.md](Robotics_Continuous_Kinematics_Benchmark.md)
(includes a 2026-06-09 correction: clean accuracy is a statistical tie at N=5 seeds,
not the single-seed win previously reported).

| Model | Params | Clean acc (N=5) | Voltage-spike acc | Streaming VRAM |
|---|---:|---:|---:|---:|
| Transformer | 611,462 | 90.45 ± 0.60 | 16.2% | grows with context (OOM @ 32k steps) |
| **ZetaPhi mixer (ours)** | **542,246** | 90.32 ± 0.67 | **76.0%** | **< 5 MB, flat** |

The honest headline is not accuracy (tie) — it is fault robustness and O(1) memory.

---

## NASA C-MAPSS — long-history predictive maintenance

Turbofan RUL prediction, ~70k params all models, validation-only selection, 3 seeds.
Full report: [CMAPSS_Predictive_Maintenance_Benchmark.md](CMAPSS_Predictive_Maintenance_Benchmark.md).
Baseline harness: [`cmapss_rul/`](cmapss_rul/).

| History | GRU | TCN (causal) | Transformer | ZetaPhi (ours) |
|---|---|---|---|---|
| 30 cycles | **13.00** | 14.47 | 13.51 | 21.84 |
| 50 | 14.42 | 14.84 | **13.58** | 16.09 |
| 200 | 67.48 | 41.49 | 69.45 | **32.30** |

Test RMSE, lower is better. We lose short-history clean accuracy and say so; we are the
only architecture still standing at 200-cycle histories, with flat 0.26 ms batch-1
latency from 50 to 4,096 timesteps (attention: 10x slower at 4,096, OOM at batch scale).

---

## EuRoC MAV — drone IMU odometry (NEW)

200Hz raw IMU -> relative pose (position delta + rotation), cross-sequence
split, test = MH_05_difficult, ~69k params all models, 3 seeds.
Full report: [EuRoC_IMU_Odometry_Benchmark.md](EuRoC_IMU_Odometry_Benchmark.md).
Baseline harness: [`euroc_imu/`](euroc_imu/). Per-seed numbers in
[`euroc_imu/results/`](euroc_imu/results/).

| Window | best baseline rot (deg) | ZetaPhi rot (deg) | ZetaPhi b1 latency |
|---|---|---|---|
| 1s (200 steps) | 0.684 ± 0.039 (Transformer) | **0.467 ± 0.025** | 0.277 ms |
| 2s (400) | 1.226 ± 0.164 (Transformer) | **0.791 ± 0.062** | 0.293 ms |
| 4s (800) | 2.212 ± 0.136 (Transformer) | **1.553 ± 0.107** | 0.280 ms |

Rotation wins 30-35% at every window length with flat ~0.28ms batch-1
latency (Transformer training VRAM: 4.4GB at 4s windows vs our 0.77GB).
Honest negative, stated in the report: a GRU wins clean position RMSE at
all lengths (our best variant comes within ~7%); sustained calibration
faults degrade us more than baselines. Behind a standard embedded driver
stack we are the most damage-stable model measured (packet loss ratio 1.00,
conditioned spikes 0.98 — below clean).

---

## Reproduce the baselines

```bash
pip install torch torchvision numpy

# Transformer baseline (3 seeds, 100 epochs)
python baselines.py --lanes transformer --seeds 1 2 3 --epochs 100 --lr 3e-4

# ConvMixer baseline (tuned)
python baselines.py --lanes convmixer_k5 --seeds 1 2 3 --epochs 100 --lr 1e-3
```

Results write to `./baseline_results/` with a `SUMMARY.json`. CIFAR-100 downloads
automatically on first run. A single RTX 4090 runs each lane in a few minutes.

The exact numbers we measured are in [`results/cifar100_summary.json`](results/cifar100_summary.json),
including the full training protocol.

---

## Scope and honesty

- This is a **matched-budget CIFAR-100 comparison at ~0.45M parameters.** It is not
  a claim of universal superiority across scales, datasets, or tasks.
- All compared models use an **identical** training recipe, data pipeline, and
  (within ~2%) parameter count, so the comparison is apples-to-apples.
- We report **best eval accuracy over training**. With no learning-rate schedule,
  final-epoch accuracy is lower for every model due to late overfitting; we do not
  use it as the headline number.
- The ZetaPhi mixer's code and mechanism are intentionally not published here. The
  baselines are, so anyone can verify what we are comparing against.

---

## About

ZetaPhi develops attention-replacement architectures with linear scaling for
long-context and compute-constrained settings. More: [zetaphi.ai](https://zetaphi.ai)

License: baseline code in this repository is released under the MIT License
(see [LICENSE](LICENSE)).
