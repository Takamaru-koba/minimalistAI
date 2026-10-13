# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Work for The Minimalist Edge AI Challenge (rules in `The Minimalist Edge AI Challenge.pdf`). The task is to shrink a MobileNetV2 EuroSAT classifier (10 classes, 224×224 input) while keeping accuracy, worst-class recall, latency, RAM and energy competitive. There is no test suite or linter; the scripts are standalone CLI tools, all run from the repo root because paths are relative.

## Setup

```bash
source edge-ai-env/bin/activate        # existing venv
pip install -r requirements.txt
pip install torch-pruning              # needed by prune_sweep.py, NOT in requirements.txt
python3 -c "from torchvision import datasets; datasets.EuroSAT(root='data', download=True)"   # data/ is untracked (~230 MB)
```

`.env` holds a real key (`.env.example` shows the variable name). It is gitignored; never commit it.

## Pipeline and commands

Compression pipeline, in order:

1. **Baseline**: `checkpoints/` holds the ResNet-50 teacher and the distilled MobileNetV2 student (`starting_model.py` loads the student). `build/baseline_fp32.onnx` is its FP32 export.
2. **Pruning** (`prune_sweep.py`, the active work): structured channel pruning with Torch-Pruning, then BN recalibration, then distilled fine-tune from the teacher.
3. **Quantization** (`quantize.py`): static INT8 QDQ via ONNX Runtime, calibrated on stratified *train* images, never val.
4. **Measure**: `evaluate.py` for accuracy, `measure.py` for cost.

```bash
python evaluate.py torch checkpoints/student_mobilenetv2_distilled.pt
python evaluate.py onnx  build/baseline_fp32.onnx
python quantize.py build/baseline_fp32.onnx build/baseline_int8.onnx [--n_calib 500] [--per_tensor]
python measure.py onnx build/baseline_int8.onnx    # run each model in a FRESH process or peak RAM is polluted
python prune_sweep.py --prune_pass A --ratios 0.1 0.2 0.3 [--importance l2|bn|taylor] [--global_pruning] [--ft_epochs N]
python prune_sweep.py --prune_pass B --ratios 0.1 0.2 0.3
```

`baseline_quantization.py` appears to be a copy of `quantize.py`. Edit `quantize.py` as the canonical one.

## Architecture notes that span files

- **Shared preprocessing and split**: `evaluate.py` defines `TRANSFORM`, `NUM_CLASSES` and `load_split()`. `quantize.py` imports them. `prune_sweep.py` and `starting_model.py` re-implement the same transform and read `provisional_split.json` directly. Keep all of these in sync, or accuracy numbers stop being comparable.
- **Split**: `load_split()` prefers `val_split.json` (official, from organizers) and otherwise uses `provisional_split.json` (seeded, stratified, 80/20). Provisional numbers are not comparable with the judges' split. `prune_sweep.py` hardcodes the provisional file (`SPLIT_FILE`); switch it if the official split arrives.
- **Pruned models are whole-model pickles**, not state_dicts, because channel counts change. Load with `torch.load(path, map_location="cpu", weights_only=False)`. Outputs go to `pruned/pruned_pass<A|B>_<importance>_<ratio>.pt` and `sweep_pass<A|B>_<importance>.csv`.
- **Pass A vs Pass B** (`protected_layers()` in `prune_sweep.py`, rationale in `ARCHITECTURE.md`): A protects only the classifier. B also protects the stem conv and every `InvertedResidual` project conv (`block.conv[-2]`), so only the wide hidden expand/depthwise channels are cut. Block outputs are shared through skip connections, so cutting them affects several blocks at once.
- **Pruning details**: `round_to=8` keeps channel counts NPU-friendly. Taylor importance needs a few backward passes before `pruner.step()`.
- **Metrics tracked** (baseline numbers are in `README.md`): accuracy, worst-class recall (a scored metric), params, MACs, size, latency, peak RAM, joules per inference (via codecarbon). Compare pruned and quantized candidates against the README's Step 1 and Step 2 tables.

## Reference docs

- `README.md`: running log of baseline FP32 and INT8 results and the pruning methodology
- `ARCHITECTURE.md`: MobileNetV2 stage-by-stage parameter shares and the Pass A/B pruning plan
