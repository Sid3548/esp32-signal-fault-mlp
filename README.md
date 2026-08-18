# esp32-signal-fault-mlp

A tiny fully-connected neural network that tells a healthy sine signal apart from a faulty one, small enough to run on an ESP32 microcontroller. An honest learning artifact on synthetic data — not a benchmark.

**Model:** 64→32→1 MLP · 2,113 params · int8 weights (float32 biases) · ~2.2 KB · runs on ESP32 in [Wokwi](https://wokwi.com) · 0.930 test acc

---

This is a 2,113-parameter multilayer perceptron (64 → 32 → 1, ReLU then sigmoid) that classifies a synthetic sine signal as **normal** or **faulty**, where the fault is an *amplitude sag* — the sine scaled down in height at the same frequency. It was trained in PyTorch on Colab, then int8-quantized (per-tensor symmetric: weights scaled by `max(|w|) / 127`, biases kept in float32) and exported as a self-contained C header (`model_weights.h`). At ~2.2 KB it drops straight onto an ESP32 — I run it on the [Wokwi](https://wokwi.com) ESP32 simulator, which executes real ESP32 firmware on an emulated Xtensa core, so the whole thing runs end-to-end in a browser with no external hardware.

The first fault I tried was an additive distortion, and the model hit ~1.0 accuracy almost immediately — a red flag that the task was trivially separable on one cheap feature. Checking the signals showed why: the additive fault shifted the *mean* of the signal (≈0.044 vs ≈0.000 for a normal signal), so the mean alone separated the two classes. I deliberately switched to an amplitude sag, which multiplies the sine by a random factor in [0.6, 1.0). That keeps the frequency identical and leaves the mean essentially unchanged (~0 for both classes), forcing the model to judge each peak's height instead. Accuracy dropped to 0.930 with a genuine ceiling — the signature of a real problem rather than a leaky one.

Rounding the weights to int8 and back left the PyTorch test accuracy unchanged at 0.930, while shrinking the weights from ~8 KB (float32) to ~2 KB (~2.2 KB for the full model, keeping float32 biases and scales). The errors are also one-directional: of the 14 misclassifications out of 200 test signals, **all 14 are faulty signals called normal — missed faults — and zero are normal signals called faulty**. In fault detection a missed fault is the dangerous direction, which maps directly to real applications like grid and battery-storage (BESS) protection, where calling a fault "normal" is the costly error.

## Honesty notes

- The 0.930 figures are **PyTorch test accuracy in Colab**, on a test set drawn fresh from a separate random seed (`manual_seed(956789095)`) — signals the model never trained on (train 0.938 / test 0.930).
- The int8 check rounds the weights to int8 and back to float and re-runs the float forward pass; it measures weight-rounding error, **not** integer arithmetic on the device. The notebook's job ends at the C-array export (`model_weights.h`); the ESP32 side is the deployment target (run in Wokwi), and a separate on-device accuracy run isn't logged in this repo yet.
- Every missed fault is a *barely-sagged* signal (peak amplitude 0.957–0.995), sitting right against the normal, full-amplitude cluster — which is where the 0.93 ceiling comes from.

## The notebook

`signal_fault_mlp.ipynb` is a working/learning notebook — it keeps earlier dead-ends, including one broken training run where the loss is stuck at 0.7083 (train 0.35 / test 0.50) and some stale comments that say "8 hidden neurons" against a `Linear(64, 32)` layer. The final pipeline is in the lower cells: the model definition that prints `2113 parameters`, then the amplitude-sag signal generator, the 500-epoch training loop, the held-out evaluation, and the int8 C export.

## Files

- `signal_fault_mlp.ipynb` — the full Colab notebook (ground truth for every number above).
- `model_weights.h` — int8 weights + float32 biases + per-tensor scales, exported for on-device C.

## License

MIT — see [LICENSE](LICENSE).
