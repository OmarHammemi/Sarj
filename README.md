# Edge Arabic TTS — FastSpeech2-small

**Repo:** [github.com/OmarHammemi/Sarj](https://github.com/OmarHammemi/Sarj) · **Kaggle notebook:** [`notebooks/sarj-1.ipynb`](notebooks/sarj-1.ipynb) · **Colab:** [COLAB.md](COLAB.md)

Acoustic text-to-speech model trained from scratch on ~1 hour of single-speaker Saudi Arabic speech, optimized for edge CPU deployment (<20M parameters).

## Setup

```bash
cd ~/sarj
pip install --break-system-packages -r requirements.txt
```

## Pipeline

### 1. Download dataset (~1 hour)

```bash
python data/download_hesham.py
```

Default dataset: [`HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel`](https://huggingface.co/datasets/HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel) — single male Saudi speaker, diacritized MSA, short clips (2–12s). Streams from Hugging Face and filters to ~1 hour.

Equivalent via the generic downloader:

```bash
python data/download.py --max-hours 1.0
```

Alternative (long-form podcast, undiacritized):

```bash
python data/download.py --dataset AhmedAshrafMarzouk/saudi-podcast-1hr --max-hours 1.0
```

### 2. Preprocess

```bash
python data/preprocess.py --config configs/hesham.yaml
```

Creates `data/train.csv`, `data/val.csv`, `data/vocab.json`, mel spectrograms, and duration targets.

### 3. Train

**On GPU (Kaggle T4 or Colab):**

```bash
python train.py --config configs/hesham.yaml --device cuda
```

Resume / extended training (as in the Kaggle notebook):

```bash
python train.py --config configs/optimized.yaml --device cuda --resume checkpoints/latest.pt
```

Checkpoints saved to `checkpoints/best.pt` and `checkpoints/latest.pt`. **Use `latest.pt` for synthesis** if validation loss plateaus early.

### 4. Synthesize

Input text must include **tashkeel** (diacritics), matching the training data.

```bash
python synthesize.py \
  --text "السَّلَامُ عَلَيْكُمْ" \
  --out samples/01.wav \
  --checkpoint checkpoints/latest.pt \
  --config configs/hesham.yaml \
  --device cpu
```

Higher-quality synthesis (mel clamp + 128 Griffin-Lim iterations):

```bash
python synthesize_v2.py \
  --text "السَّلَامُ عَلَيْكُمْ" \
  --out samples/01.wav \
  --checkpoint checkpoints/latest.pt \
  --config configs/optimized.yaml
```

### 5. RTF benchmark

```bash
python eval/rtf_benchmark.py --device cpu
python eval/rtf_benchmark.py --device cuda   # on Colab
```

## Project layout

```text
configs/          default.yaml, hesham.yaml, optimized.yaml
data/             download_hesham.py, audio_utils.py, preprocess
model/            FastSpeech2-small architecture
notebooks/        sarj-1.ipynb (full Kaggle training notebook)
train.py          Training loop
synthesize.py     Text → WAV (librosa Griffin-Lim)
synthesize_v2.py  Mel clamp + stronger Griffin-Lim
eval/             RTF benchmark
data/Samples/     Submission WAV files (5 sentences + ablation)
samples/          prompts.txt
checkpoints/      Model weights
TECHNICAL_SUMMARY.md
```

## Constraints

| Constraint | Implementation |
|------------|----------------|
| Model < 20M params | Verified in `model/fastspeech.py` |
| No pretrained Arabic acoustic model | Trained from scratch only |
| ~1 hour data | HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel (HF Saudi audio) |

## Ablation

Train without PostNet:

```bash
python train.py --config configs/ablation_no_postnet.yaml --device cuda
```

Compare outputs with the default model on the same sentences:

```bash
python synthesize.py --text "..." --out data/Samples/ablation/with_postnet.wav \
  --checkpoint checkpoints/latest.pt --config configs/optimized.yaml

python synthesize.py --text "..." --out data/Samples/ablation/no_postnet.wav \
  --checkpoint checkpoints/latest.pt --config configs/optimized.yaml --no-postnet
```

## Submission audio

Five synthesized samples: **`data/Samples/`** (diacritized Arabic filenames). See `TECHNICAL_SUMMARY.md` and `samples/prompts.txt`.

## Kaggle / Colab workflow

**Recommended:** upload or open [`notebooks/sarj-1.ipynb`](notebooks/sarj-1.ipynb) on Kaggle (GPU T4). It writes the fixed scripts, downloads Hesham data, preprocesses, trains, and generates 5 submission samples.

Quick CLI on Kaggle:

```bash
python data/download_hesham.py --max-hours 1.0 --min-dur 2.0 --max-dur 12.0
python data/preprocess.py --config configs/hesham.yaml
python train.py --config configs/hesham.yaml --device cuda
```

See [COLAB.md](COLAB.md) for Google Colab steps.
