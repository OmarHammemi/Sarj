# Google Colab — Train on GPU

Repo: **https://github.com/OmarHammemi/Sarj**

For the full end-to-end workflow (download → train → 5 samples), use the Kaggle notebook: [`notebooks/sarj-1.ipynb`](notebooks/sarj-1.ipynb).

## 1. Open Colab

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Sign in with Google
3. **Runtime → Change runtime type → GPU (T4) → Save**

## 2. Clone and install

Run these cells in order:

```python
!nvidia-smi
```

```python
!git clone https://github.com/OmarHammemi/Sarj.git
%cd /content/Sarj
!pip install -q -r requirements.txt
```

## 3. Prepare data

Processed data is **not** in GitHub (too large). Run on Colab:

```python
!python data/download_hesham.py --max-hours 1.0 --min-dur 2.0 --max-dur 12.0
!python data/preprocess.py
```

Expected: ~400–800 short clips, ~1 hour, diacritized text.

**Note:** Input text at synthesis time must include tashkeel (diacritics), matching the training data.

## 4. Reduce batch size (avoid GPU OOM)

Short clips allow larger batches than the old podcast data. On T4, batch size 8 works well:

```python
import yaml
from pathlib import Path

cfg = yaml.safe_load(open("configs/default.yaml"))
cfg["train"]["batch_size"] = 8
cfg["train"]["num_workers"] = 2
Path("configs/default.yaml").write_text(yaml.dump(cfg, sort_keys=False))
print("batch_size =", cfg["train"]["batch_size"])
```

## 5. Smoke test (optional)

```python
import yaml
from pathlib import Path
cfg = yaml.safe_load(open("configs/default.yaml"))
cfg["train"]["max_steps"] = 20
Path("configs/smoke.yaml").write_text(yaml.dump(cfg, sort_keys=False))
!python train.py --config configs/smoke.yaml --device cuda
```

## 6. Full training

```python
import yaml
from pathlib import Path
cfg = yaml.safe_load(open("configs/default.yaml"))
cfg["train"]["max_steps"] = 120000
Path("configs/default.yaml").write_text(yaml.dump(cfg, sort_keys=False))

!python train.py --device cuda
```

Training takes ~2–5 hours on T4. Checkpoints:

- `checkpoints/best.pt` — best validation loss
- `checkpoints/latest.pt` — resume point

## 7. Save to Google Drive (recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p "/content/drive/MyDrive/Sarj-checkpoints"
!cp checkpoints/best.pt "/content/drive/MyDrive/Sarj-checkpoints/"
```

## 8. Resume if disconnected

```python
%cd /content/Sarj
!python train.py --device cuda --resume checkpoints/latest.pt
```

## 9. GPU RTF + test audio

```python
!python eval/rtf_benchmark.py --device cuda

!python synthesize.py \
  --text "السَّلَامُ عَلَيْكُمْ" \
  --out samples/colab_test.wav \
  --device cuda

from IPython.display import Audio
Audio("samples/colab_test.wav")
```

## 10. Download checkpoint to your VM

```python
from google.colab import files
files.download("checkpoints/best.pt")
```

On your VM:

```bash
mkdir -p ~/sarj/checkpoints
# upload best.pt to ~/sarj/checkpoints/
cd ~/sarj
python3 synthesize.py --text "السلام عليكم" --out samples/01.wav --device cpu
python3 eval/rtf_benchmark.py --device cpu
bash scripts/generate_samples.sh
```

## Private repo?

If the repo is private, use a [GitHub Personal Access Token](https://github.com/settings/tokens):

```python
import getpass
token = getpass.getpass("GitHub token: ")
!git clone https://{token}@github.com/OmarHammemi/Sarj.git
%cd /content/Sarj
```

Or: **File → Upload notebook** and upload a zip of the project instead of cloning.
