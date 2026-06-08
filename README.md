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
