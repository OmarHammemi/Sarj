# Edge Arabic TTS — FastSpeech2-small

**Repo:** [github.com/OmarHammemi/Sarj](https://github.com/OmarHammemi/Sarj) · **Colab training:** see [COLAB.md](COLAB.md)

Acoustic text-to-speech model trained from scratch on ~1 hour of single-speaker Saudi Arabic speech, optimized for edge CPU deployment (<20M parameters).

## Setup

```bash
cd ~/sarj
pip install --break-system-packages -r requirements.txt
```

## Pipeline

### 1. Download dataset (~1 hour)

```bash
python data/download.py
```

Default dataset: [`AhmedAshrafMarzouk/saudi-podcast-1hr`](https://huggingface.co/datasets/AhmedAshrafMarzouk/saudi-podcast-1hr) (~1 hour, from the [Saudi audio search page](https://huggingface.co/datasets?modality=modality:audio&sort=trending&search=saudi))

Alternative (single speaker, diacritized, slice 1h):

```bash
python data/download.py --dataset HeshamHaroon/arabic-msa-25k-saudi-male-tashkeel --max-hours 1.0
```

### 2. Preprocess

```bash
python data/preprocess.py
```

Creates `data/train.csv`, `data/val.csv`, `data/vocab.json`, mel spectrograms, and duration targets.

### 3. Train

**On GPU (Colab recommended):**

```bash
python train.py --device cuda
```

**Smoke test on CPU:**

```bash
python train.py --device cpu
```

Checkpoints saved to `checkpoints/best.pt` and `checkpoints/latest.pt`.

### 4. Synthesize

```bash
python synthesize.py \
  --text "السَّلامُ عَلَيْكُمْ" \
  --out samples/01.wav \
  --device cpu
```

### 5. RTF benchmark

```bash
python eval/rtf_benchmark.py --device cpu
python eval/rtf_benchmark.py --device cuda   # on Colab
```

## Project layout

```text
configs/          Training configuration
data/             Download, preprocess, dataset
model/            FastSpeech2-small architecture
train.py          Training loop
synthesize.py     Text → WAV (Griffin-Lim)
eval/             RTF benchmark
samples/          Generated audio for submission
checkpoints/      Model weights
TECHNICAL_SUMMARY.md
```

## Constraints

| Constraint | Implementation |
|------------|----------------|
| Model < 20M params | Verified in `model/fastspeech.py` |
| No pretrained Arabic acoustic model | Trained from scratch only |
| ~1 hour data | Saudi HF search datasets (default: saudi-podcast-1hr) |

## Ablation

Train without PostNet:

```bash
python train.py --config configs/ablation_no_postnet.yaml --device cuda
```

Compare outputs with the default model on the same sentences.

## Colab workflow

1. Upload or clone this repo
2. `pip install -r requirements.txt`
3. Run download + preprocess (or upload preprocessed data)
4. `python train.py --device cuda`
5. Download `checkpoints/best.pt`
