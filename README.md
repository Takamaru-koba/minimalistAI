# minimalistAI

Work for The Minimalist Edge AI Challenge. See `The Minimalist Edge AI Challenge.pdf` for the rules.

## Setup

```bash
python3 -m venv edge-ai-env
source edge-ai-env/bin/activate
```

```bash
pip install -r requirements.txt
```

## Data

`data/` is not tracked (EuroSAT is ~230 MB). Download it once:

```bash
python3 -c "from torchvision import datasets; datasets.EuroSAT(root='data', download=True)"
```

## Files

- `checkpoints/` — teacher (ResNet-50) and distilled student (MobileNetV2) weights
- `build/baseline_fp32.onnx` — exported FP32 baseline
- `provisional_split.json` — train/val split used so far
- `evaluate.py`, `measure.py` — accuracy evaluation and cost measurement
