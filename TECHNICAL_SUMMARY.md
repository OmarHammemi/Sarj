# Edge Arabic TTS — Technical Summary

> Fill in RTF numbers and ablation notes after training.

## 1. Architecture choice

**Model:** FastSpeech2-small (non-autoregressive acoustic model)

**Why suitable for edge CPU deployment:**

- Parallel mel generation (no autoregressive loop over time)
- Parameter count under 20M (see table below)
- Predictable latency; batch size 1 friendly
- Griffin-Lim vocoder avoids separate heavy neural vocoder training

**Parameter breakdown:**

| Component | Parameters |
|-----------|------------|
| Text encoder | TBD |
| Duration predictor | TBD |
| Mel decoder | TBD |
| PostNet | TBD |
| **Total** | TBD (< 20M) |

## 2. Data

- **Dataset:** [HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel](https://huggingface.co/datasets/HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel) (from [Saudi audio HF search](https://huggingface.co/datasets?modality=modality:audio&sort=trending&search=saudi))
- **Speaker:** Single male Saudi speaker, diacritized Modern Standard Arabic
- **Duration:** ~1 hour subset (400–800 short clips, 2–12s each; use `--max-hours 1.0`)
- **Split:** 90% train / 10% val
- **Text:** Fully diacritized Arabic (tashkeel retained)
- **Alignments:** Character-proportional duration bootstrap (no pretrained Arabic acoustic model)

## 3. Training

- Sample rate: 22050 Hz, 80-dim log-mel
- Losses: L1 mel (+ PostNet), log-duration MSE
- Optimizer: AdamW, lr=1e-3
- Steps: 120000 (adjust if early stopping)
- **Overfitting:** Expected with ~1h data; goal is working pipeline, not generalization

## 4. RTF (Real-Time Factor)

RTF = inference_time / audio_duration (lower is faster; RTF < 1 = faster than realtime)

| Device | Avg inference (s) | Audio duration (s) | RTF |
|--------|-------------------|--------------------|-----|
| GPU (T4) | TBD | TBD | TBD |
| CPU (F16as v6, 16 cores) | TBD | TBD | TBD |

**Measurement protocol:** batch size 1, 3 warmup runs, average of 5 runs, includes mel model + Griffin-Lim.

## 5. Ablation analysis (required)

**Ablation:** Remove PostNet (`configs/ablation_no_postnet.yaml`)

**What changed:**

- TBD (e.g. more buzzy / less crisp harmonics)

**Interpretation:**

- PostNet refines mel spectrogram details; without it, Griffin-Lim reconstruction quality drops.

## 6. Production evolution

- Scale to 100+ hours multi-dialect Saudi data
- Train lightweight neural vocoder (HiFi-GAN small) on same speaker
- Export to ONNX + INT8 quantization for mobile/edge
- Add diacritization frontend for undiacritized input
- Streaming inference with chunked mel generation

## 7. Sample sentences

List the 5 Arabic prompts used in `samples/prompts.txt`.
