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

____________________________________________________________________________________________

## Step 1: Benchmark Baseline ONNX Metrics

"split": "provisional",
  "accuracy": 0.9925925925925926,
  "worst_class_accuracy": 0.974,
  "per_class_recall": [
    0.9917,
    1.0,
    0.9867,
    1.0,
    0.996,
    0.9875,
    0.974,
    0.9983,
    0.99,
    0.9983
  ],
  "n_val": 5400

 "framework": "onnxruntime",
  "params_million": 2.2196,
  "size_mb": 8.5063,
  "latency_ms": 7.2081,
  "peak_ram_mb": 119.0117,
  "joules_per_inference": 0.0865,
  "codecarbon_cpu_power_w": 3.9675,
  "codecarbon_tracking_mode": "machine"

__________________________________________________________________________________

## Step 2: Safety INT8 Baseline Quantization


