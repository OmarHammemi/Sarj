# Edge Arabic TTS — Technical Summary

## 1. Architecture choice

**Model:** FastSpeech2-small (non-autoregressive acoustic model)

**Why suitable for edge CPU deployment:**

- Parallel mel generation (no autoregressive loop over time)
- **5,700,673 parameters** (strictly under the 20M constraint)
- Predictable latency; batch size 1 friendly
- Griffin-Lim vocoder avoids training a separate heavy neural vocoder

**Parameter breakdown:**

| Component | Parameters |
|-----------|------------|
| Text encoder | 3,171,840 |
| Duration predictor | 395,009 |
| Mel decoder | 1,599,568 |
| PostNet | 534,256 |
| **Total** | **5,700,673** |

## 2. Data

- **Dataset:** [HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel](https://huggingface.co/datasets/HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel) (from [Saudi audio HF search](https://huggingface.co/datasets?modality=modality:audio&sort=trending&search=saudi))
- **Speaker:** Single male Saudi speaker, diacritized Modern Standard Arabic
- **Duration:** ~1 hour subset (400–800 short clips, 2–12s; `--max-hours 1.0`)
- **Split:** 90% train / 10% val
- **Text:** Fully diacritized Arabic (tashkeel required at inference)
- **Alignments:** Character-proportional duration bootstrap (no pretrained Arabic acoustic model)

## 3. Training

- Sample rate: 22050 Hz, 80-dim log-mel (librosa, matched between train and synthesis)
- Losses: L1 mel + PostNet L1 + log-duration MSE
- Optimizer: AdamW, lr=3e-4 (initial) → 1e-4 (fine-tune), batch 8→4
- Checkpoint: step **10,000**, val loss **2.51**, vocab size **50**
- **Overfitting:** Expected with ~1h data; goal is a working pipeline, not generalization

## 4. RTF (Real-Time Factor)

RTF = inference_time / audio_duration (lower is faster; RTF < 1 = faster than realtime)

| Device | Avg inference (s) | Audio duration (s) | RTF |
|--------|-------------------|--------------------|-----|
| GPU (T4, Kaggle) | 0.8053 | 1.9969 | 0.4033 |
| CPU (Azure VM) | 0.6124 | 1.9621 | 0.3121 |

**Measurement protocol:** batch size 1, 3 warmup runs, average of 5 runs, includes mel model + Griffin-Lim (128 iterations).

## 5. Ablation analysis — PostNet removed

**Ablation:** Disable PostNet at inference (`python synthesize.py --no-postnet`) on the same checkpoint trained with PostNet. Config for training without PostNet: `configs/ablation_no_postnet.yaml`.

**Test sentence:** see `samples/prompts.txt` line 1.

**What changed:**

- **With PostNet:** smoother, more stable output; lower RMS noise floor (0.159)
- **Without PostNet:** harsher, more buzzy timbre; higher RMS (0.196) indicating extra unmodeled energy after Griffin-Lim

**Interpretation:** PostNet adds a residual refinement pass over the decoder mel before vocoding. Without it, the mel lacks fine harmonic detail and Griffin-Lim amplifies artifacts. Speech remains partially intelligible but noticeably rougher — PostNet is the main quality refinement stage in this small-data, vocoder-free pipeline.

Ablation audio: `data/Samples/ablation/with_postnet.wav` vs `data/Samples/ablation/no_postnet.wav`

## 6. Production evolution

- Scale to 100+ hours multi-dialect Saudi data
- Train lightweight neural vocoder (HiFi-GAN small) on same speaker
- Export to ONNX + INT8 quantization for mobile/edge
- Add diacritization frontend for undiacritized input
- Replace bootstrap alignments with MFA or a dedicated aligner

## 7. Sample sentences

Five submission WAV files are in `data/Samples/`. Prompts listed in `samples/prompts.txt`.
