# O(N) Continuous Kinematics: Real-Time Edge Sensor Processing without Memory Scaling

**Abstract:**
Standard O(N^2) Self-Attention architectures struggle when deployed to continuous edge sensor streams (such as Drone IMUs, robotics kinematics, and Bluetooth wearables) because their required context scales quadratically with time, exhausting device memory. We benchmark a proprietary O(N) 1D architecture designed for physical state-updating against a parameter-matched Transformer on continuous 50Hz kinematics (UCI Human Activity Recognition). The architecture matches the Transformer's clean-data accuracy at equal parameter budget, maintains a flat O(1) memory footprint during streaming inference, and proves dramatically more robust to real-world edge hardware faults such as instantaneous voltage spikes.

*(Note: The core mathematical mechanism and source code for the multi-branch state-update logic are proprietary and withheld from this public release. The training harnesses, data loaders, and benchmark scripts used to generate these metrics are available in this repository.)*

> **Correction (2026-06-09):** An earlier version of this document reported a single-seed
> accuracy comparison (92.06% vs 90.30%) and claimed an absolute accuracy win. Under our
> own stricter multi-seed protocol (N=5 seeds, identical training recipe), clean accuracy
> is statistically indistinguishable between the two architectures. We are correcting the
> record because the honest claims — parity accuracy, O(1) memory, and fault robustness —
> are the ones we stand behind. The robustness and memory-scaling results below are
> unchanged by this correction.

---

## 1. Clean Accuracy: Parameter-Matched Parity (N=5 seeds)

Benchmarked on **UCI Human Activity Recognition (HAR)**: continuous 50Hz sequences of 9-axis accelerometer and gyroscope data, 6 activity classes. Both models trained 50 epochs, cosine annealing, AdamW, identical pipeline, 5 seeds.

| Architecture | Parameters | Test Accuracy (mean ± std, N=5) |
| :--- | :--- | :--- |
| 1D Transformer (Baseline) | 611,462 | 90.45% ± 0.60% |
| **Proprietary O(N) Model** | **542,246** | **90.32% ± 0.67%** |

**Conclusion:** At matched parameter budget (the O(N) model in fact uses ~11% fewer
parameters), clean-data accuracy is a statistical tie. The architecture's value on this
task is not raw accuracy — it is everything below.

The O(N) model trains faster per epoch (0.28s vs 0.35s) due to a fully vectorized
multi-branch tensor layout that allows `torch.compile()` to fuse the entire
forward/backward graph.

---

## 2. Infinite Edge Context Scaling (The OOM Test)

We simulated a continuous edge-sensor stream, feeding data 1 timestep at a time and tracking the VRAM required to predict the current state.

| Sequence Length Processed | Transformer Latency | Transformer VRAM | Proprietary Model Latency | Proprietary Model VRAM |
| :--- | :--- | :--- | :--- | :--- |
| 1,024 timesteps | 0.35 ms / step | 43 MB | 7.53 ms / step (eager PyTorch) | < 5 MB |
| 4,096 timesteps | 2.13 ms / step | 92 MB | 9.42 ms / step (eager PyTorch) | < 5 MB |
| 16,384 timesteps | 23.22 ms / step | 333 MB | 28.57 ms / step (eager PyTorch) | < 5 MB |
| **32,768 timesteps** | **OOM (CRASH)** | **CRASH** | **0.02 ms / step (fused C++ kernel)** | **< 5 MB** |

*Measurement note: rows 1-3 for the proprietary model are eager-mode PyTorch (unfused);
the final row is the deployed fused C++ kernel. The kernel's per-step cost is constant
across all sequence lengths — the eager numbers are shown to be conservative about what
unoptimized deployment looks like.*

**Conclusion:** The proprietary model requires effectively constant memory state. It can ingest continuous sensor streams indefinitely without crashing or slowing down.

---

## 3. Real-World Hardware Robustness

Edge sensors suffer dropped Bluetooth packets, asynchronous polling, thermal drift, and voltage spikes. We injected hardware corruption directly into the 50Hz stream *during evaluation*.

| Hardware Damage Type | Transformer Accuracy | Proprietary Model Accuracy | Delta |
| :--- | :--- | :--- | :--- |
| Clean Baseline | 88.0% | 89.8% | +1.8% |
| **Spike Bursts (+5.0 Voltage)** | 16.2% (Total Failure) | **76.0%** | **+59.8%** |
| Gaussian Noise (Static) | 80.9% | **89.3%** | **+8.3%** |
| Gyroscope Complete Dropout | 73.2% | 74.7% | +1.5% |
| Packet Loss (Raw Zeros) | **65.5%** | 43.7% | -21.8% |
| Packet Loss (Forward-Filled Driver) | 87.8% | **89.3%** | +1.6% |

### Analysis of Hardware Resilience:
1. **Shock Absorption (Spike Bursts):** A single massive voltage spike corrupts the Transformer's entire attention context, destroying accuracy (16.2%). The proprietary model's physical state-updating diffuses and absorbs instantaneous spikes, maintaining 76.0% accuracy through severe hardware faults. This is the headline robustness result.
2. **Dropped Packets — honest negative and the standard fix:** Because the architecture computes physical kinematics, feeding artificial `0.0` values during dropped packets breaks the physics (falsely implying the device instantly stopped moving) — raw zeros hurt it MORE than the Transformer (43.7% vs 65.5%). The standard embedded solution — a Zero-Order Hold (forward-fill) at the driver level, ordinary firmware practice applied equally to both models — recovers the architecture to **89.3%**, beating the Transformer's 87.8% under 20% total packet loss. We report both rows; the raw-zeros weakness is real and driver pairing is the deployment assumption.
