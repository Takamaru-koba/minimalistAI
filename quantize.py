#!/usr/bin/env python3
"""
quantize.py — static INT8 (QDQ) quantization of an FP32 ONNX model with ONNX Runtime.

Steps:
  1. preprocess (shape inference + graph cleanup) — required before quantizing
  2. calibrate on real training images (stratified sample, same transforms/split as evaluate.py)
  3. quantize_static -> QDQ format, per-channel INT8 weights, UINT8 activations

RUN (from the starter-kit folder, next to evaluate.py):
  python quantize.py build/baseline_fp32.onnx build/baseline_int8.onnx
  python quantize.py build/baseline_fp32.onnx build/baseline_int8.onnx --n_calib 500
  python quantize.py build/baseline_fp32.onnx build/baseline_int8_pt.onnx --per_tensor

THEN CHECK ACCURACY:
  python evaluate.py onnx build/baseline_int8.onnx
"""
import argparse
import os
import time

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)

try:
    from onnxruntime.quantization.shape_inference import quant_pre_process
except ImportError:  # older/newer ORT layouts
    from onnxruntime.quantization import quant_pre_process

from torchvision import datasets

# Reuse the kit's exact preprocessing and split logic
from evaluate import NUM_CLASSES, ROOT, TRANSFORM, load_split


def input_info(model_path):
    """Input name and the batch size the model expects (1 if dynamic)."""
    model = onnx.load(model_path)
    inp = model.graph.input[0]
    dim0 = inp.type.tensor_type.shape.dim[0]
    batch = dim0.dim_value if dim0.HasField("dim_value") and dim0.dim_value > 0 else 1
    return inp.name, batch


class EuroSATCalibReader(CalibrationDataReader):
    """Feeds a stratified sample of TRAIN images (never val) to the calibrator."""

    def __init__(self, input_name, batch, n, seed=0):
        dataset = datasets.EuroSAT(root=ROOT / "data", download=False, transform=TRANSFORM)
        split, kind = load_split(dataset.targets)
        targets = np.asarray(dataset.targets)
        train_idx = np.asarray(split["train"])
        rng = np.random.default_rng(seed)

        per_class = max(1, n // NUM_CLASSES)
        chosen = []
        for c in range(NUM_CLASSES):
            idx_c = train_idx[targets[train_idx] == c]
            chosen += rng.choice(idx_c, size=min(per_class, len(idx_c)), replace=False).tolist()
        rng.shuffle(chosen)
        print(f"Calibration: {len(chosen)} train images ({per_class}/class), {kind} split")

        images = [dataset[i][0].numpy() for i in chosen]
        # group into the batch size the model expects (drop a ragged tail)
        n_batches = len(images) // batch
        self.batches = iter([
            {input_name: np.stack(images[b * batch:(b + 1) * batch]).astype(np.float32)}
            for b in range(n_batches)
        ])

    def get_next(self):
        return next(self.batches, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fp32_onnx")
    ap.add_argument("int8_onnx")
    ap.add_argument("--n_calib", type=int, default=200, help="calibration images (spread across classes)")
    ap.add_argument("--per_tensor", action="store_true", help="per-tensor weights instead of per-channel")
    ap.add_argument("--method", choices=["minmax", "entropy", "percentile"], default="minmax")
    args = ap.parse_args()

    t0 = time.time()
    prep_path = args.int8_onnx.replace(".onnx", "_prep.onnx")
    print("1/3 preprocess ...")
    quant_pre_process(args.fp32_onnx, prep_path)

    input_name, batch = input_info(prep_path)
    print(f"    input '{input_name}', batch {batch}")

    print("2/3 loading calibration images ...")
    reader = EuroSATCalibReader(input_name, batch, args.n_calib)

    print("3/3 quantize_static (QDQ) ...")
    method = {"minmax": CalibrationMethod.MinMax,
              "entropy": CalibrationMethod.Entropy,
              "percentile": CalibrationMethod.Percentile}[args.method]
    quantize_static(
        prep_path,
        args.int8_onnx,
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=not args.per_tensor,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=method,
    )
    os.remove(prep_path)

    fp32_mb = os.path.getsize(args.fp32_onnx) / (1024 * 1024)
    int8_mb = os.path.getsize(args.int8_onnx) / (1024 * 1024)
    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"  {args.fp32_onnx}: {fp32_mb:.2f} MB")
    print(f"  {args.int8_onnx}: {int8_mb:.2f} MB  ({fp32_mb / int8_mb:.2f}x smaller)")
    print(f"\nNext: python evaluate.py onnx {args.int8_onnx}")


if __name__ == "__main__":
    main()