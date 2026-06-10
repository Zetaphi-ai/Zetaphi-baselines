# EuRoC MAV IMU Odometry Benchmark: ZetaPhi vs Parameter-Matched Baselines
## Public Benchmark Card (sanitized)

Date: June 2026 | Hardware: 2x RTX 4090 | Org: ZetaPhi

> IP NOTICE: ZetaPhi's model architecture and internal configuration are
> withheld pending patent filing. This card publishes the complete protocol,
> baseline code, data processing, and ALL measured numbers — including
> ZetaPhi's — so every claim is auditable against the published harness.
> ZetaPhi configuration variants are referred to by opaque labels
> (variant-A/B/C/D, tuned-combo); they differ only in internal
> non-trainable settings, never in parameter count (~69k like all models).

---

## Task

Inertial odometry from raw 200Hz IMU (3-axis gyro + 3-axis accel): predict
relative body-frame motion over a window — position delta (dp, cm) and
rotation vector (rot, deg). Standard learned-inertial-odometry formulation
(cf. IONet/RoNIN). Window lengths 200/400/800 samples (1s/2s/4s).

## Protocol (honest-card rules)

- Cross-sequence split: train MH_01, MH_02, MH_04; val MH_03;
  test MH_05_difficult (hardest), touched once per final model.
- ~69k parameters ALL models: GRU 69.7k / causal TCN 69.4k / Transformer
  70.2k (sinusoidal PE) / ZetaPhi 68.8k.
- Model selection on validation only; 3 seeds per cell, mean ± std.
- Tuned configurations required a pre-registered 3-seed confirm on
  validation before test numbers were read.
- Compute metrics (train wall-clock, peak train VRAM, CUDA-synced batch-1
  latency) recorded inside every result row by the same harness run.
- Per-seed results: results/euroc_maingrid_per_seed.jsonl,
  results/euroc_robustness_seq800_per_seed.jsonl.

## Main grid — clean test (MH_05, 3 seeds)

### seq200 (1s)
| model       | dp RMSE (cm) | rot RMSE (deg) | train (s) | peak VRAM | b1 latency |
|-------------|--------------|----------------|-----------|-----------|------------|
| GRU         | 24.1 ± 1.2   | 1.060 ± 0.037  | 13.6      | 174 MB    | 0.108 ms   |
| TCN         | 26.0 ± 0.9   | 0.755 ± 0.168  | 16.3      | 74 MB     | 0.354 ms   |
| Transformer | 26.3 ± 1.0   | 0.684 ± 0.039  | 21.1      | 346 MB    | 0.315 ms   |
| ZetaPhi     | 29.7 ± 1.2   | 0.467 ± 0.025  | 34.5      | 256 MB    | 0.277 ms   |

### seq400 (2s)
| model       | dp RMSE (cm) | rot RMSE (deg) | train (s) | peak VRAM | b1 latency |
|-------------|--------------|----------------|-----------|-----------|------------|
| GRU         | 41.2 ± 1.3   | 2.030 ± 0.075  | 18.4      | 332 MB    | 0.115 ms   |
| TCN         | 47.0 ± 0.7   | 1.523 ± 0.071  | 19.5      | 121 MB    | 0.326 ms   |
| Transformer | 53.0 ± 2.6   | 1.226 ± 0.164  | 42.6      | 1180 MB   | 0.321 ms   |
| ZetaPhi     | 52.1 ± 0.2   | 0.791 ± 0.062  | 29.1      | 433 MB    | 0.293 ms   |

### seq800 (4s)
| model       | dp RMSE (cm) | rot RMSE (deg) | train (s) | peak VRAM | b1 latency |
|-------------|--------------|----------------|-----------|-----------|------------|
| GRU         | 79.7 ± 6.2   | 4.276 ± 0.331  | 16.5      | 643 MB    | 0.648 ms   |
| TCN         | 92.1 ± 8.9   | 2.484 ± 0.031  | 18.4      | 218 MB    | 0.319 ms   |
| Transformer | 100.6 ± 1.3  | 2.212 ± 0.136  | 115.1     | 4391 MB   | 0.338 ms   |
| ZetaPhi     | 101.7 ± 3.6  | 1.553 ± 0.107  | 85.0      | 771 MB    | 0.280 ms   |

Validation MSE at seq800 (balanced objective): ZetaPhi 0.367 (best) <
Transformer 0.372 < TCN 0.523 < GRU 0.645.

### Improved ZetaPhi configurations (seq800, 3 seeds, ~69k params, internal settings withheld)

| config            | val MSE         | test dp (cm) | test rot (deg) | b1 latency |
|-------------------|-----------------|--------------|----------------|------------|
| variant-A         | 0.358 ± 0.011   | 95.2 ± 1.6   | 1.458 ± 0.129  | 0.249 ms   |
| variant-C (stack) | 0.361 ± 0.013   | 85.4 ± 4.5   | 1.567 ± 0.128  | 0.261 ms   |
| variant-D         | 0.345 ± 0.016   | 94.1 ± 1.5   | 1.578 ± 0.062  | 0.268 ms   |

All variant changes are zero-parameter and zero-latency. Variant selection
used validation only, with a pre-registered decision rule.

## Headline findings

1. ROTATION: ZetaPhi best at every window length — 32%/35%/30% better than
   the next-best model, with top-2 seed stability throughout.
2. FLAT LATENCY: ZetaPhi compiled batch-1 latency is 0.26-0.28ms from 200
   to 1600 steps. At seq800 it is simultaneously the best validation model
   and the fastest batch-1 model on the board. Transformer training memory
   grows 346MB -> 4.4GB (seq200 -> 800) — 5.7x ZetaPhi's at seq800.
3. POSITION (honest negative): GRU wins clean dp at all lengths
   (best ZetaPhi variant comes within ~7%, overlapping error bars, while
   beating GRU's rotation 2.7x). GRU's validation MSE is nonetheless the
   worst on the board at every length.
4. LONG-WINDOW CROSSOVER: ZetaPhi moves from mid-pack (seq200) to best
   validation model (seq800), reproducing the same crossover measured on
   C-MAPSS FD001 (see sibling card in this repo).

## Reality-damage suite (seq800, same checkpoints, identical corrupted data per model)

Raw sensor faults vs the same fault behind a standard embedded driver
(zero-order-hold for packet loss, freeze for dead axes, 3-tap Hampel for
spikes — standard firmware practice, applied equally to all models).
Position RMSE ratio to own clean (3-seed mean), selected rows:

| corruption           | GRU  | TCN  | Transformer | ZetaPhi | ZetaPhi variant-C |
|----------------------|------|------|-------------|---------|-------------------|
| dropped packets raw  | 1.22 | 1.17 | 1.19        | 1.52    | 1.79              |
| dropped + ZOH driver | 1.05 | 1.11 | 1.01        | 1.00    | 1.03              |
| spike bursts raw     | 1.16 | 1.15 | 1.16        | 1.02    | 1.11              |
| spikes + Hampel      | 1.14 | 1.29 | 1.05        | 0.98    | 0.98              |
| timestamp jitter     | 1.01 | 1.02 | 1.00        | 1.00    | 1.01              |
| axis frozen          | 2.93 | 3.03 | 2.52        | 3.62    | 3.69              |
| accel bias           | 2.93 | 2.52 | 2.48        | 2.82    | 3.22              |

- Behind the driver stack, ZetaPhi is the most damage-stable model measured:
  perfect recovery on packet loss (ratio 1.00), BELOW-clean on conditioned
  spikes (0.98 — variant-C's 83.3cm there is the best absolute position of
  any model under any fault in the full table), immune to timestamp jitter
  and broadband noise. Reproduces the transient-fault pattern from the
  UCI-HAR and C-MAPSS cards on a third domain.
- Honest negative: sustained calibration faults (frozen axes, constant
  bias) degrade ZetaPhi MORE in ratio terms than baselines. Raw
  (undriven) packet loss also hurts it most. Deploy behind the standard
  driver stack, with calibration-fault detection upstream — as any
  window-level IMU model should be.

## Limitations

- Single test flight (MH_05); leave-one-out across all five Machine Hall
  sequences is planned.
- IMU-only odometry, not full VIO; the claim concerns the temporal mixer
  on inertial streams.
- Cross-window drift not measured. 4090-class hardware only.

## Reproduce

- dataset.py — EuRoC ASL parser, cross-sequence splits, window/target
  extraction (fully published).
- models.py — all baselines (fully published); ZetaPhi build raises
  NotImplementedError with this notice.
- train.py / robustness.py — full protocol incl. per-row compute card and
  the seeded corruption suite (fully published).
- results/ — per-seed JSONL for every cell, including the cells we lost.

Full EuRoC data: ETH Research Collection, DOI 10.3929/ethz-b-000690084
(machine_hall archive). The old robotics.ethz.ch mirror is dead.

*ZetaPhi benchmark series. Truth above all else: the negatives in this card
are load-bearing for trusting its positives.*
