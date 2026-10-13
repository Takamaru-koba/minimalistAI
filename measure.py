"""Efficiency measurements shared by baseline and final model (Section 5 protocol).

Usage (run each model in a fresh process so peak RAM is not polluted):
    python measure.py torch checkpoints/student_mobilenetv2_distilled.pt
    python measure.py onnx  student_model.onnx
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import psutil

WARMUP, RUNS, ENERGY_RUNS = 10, 100, 200
INPUT_SHAPE = (1, 3, 224, 224)


class PeakRss:
    """Poll resident memory continuously and keep the maximum."""

    def __init__(self, interval_s=0.005):
        self.proc, self.interval_s = psutil.Process(os.getpid()), interval_s
        self.peak_mb, self._on = self._rss(), False

    def _rss(self):
        return self.proc.memory_info().rss / (1024 * 1024)

    def _loop(self):
        while self._on:
            self.peak_mb = max(self.peak_mb, self._rss())
            time.sleep(self.interval_s)

    def __enter__(self):
        self._on = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._on = False
        self._t.join()


def timed(infer):
    """Return median latency, peak RSS, joules per inference and the codecarbon mode."""
    from codecarbon import EmissionsTracker

    for _ in range(WARMUP):
        infer()
    with PeakRss() as rss:
        latencies = []
        for _ in range(RUNS):
            start = time.perf_counter()
            infer()
            latencies.append((time.perf_counter() - start) * 1000)
    tracker = EmissionsTracker(measure_power_secs=1, log_level="error", save_to_file=False)
    tracker.start()
    for _ in range(ENERGY_RUNS):
        infer()
    tracker.stop()
    data = tracker.final_emissions_data
    return {
        "latency_ms": float(np.median(latencies)),
        "peak_ram_mb": rss.peak_mb,
        "joules_per_inference": data.energy_consumed * 3.6e6 / ENERGY_RUNS,
        "codecarbon_cpu_power_w": float(data.cpu_power),
        "codecarbon_tracking_mode": str(getattr(data, "tracking_mode", "unknown")),
    }


def measure_torch(ckpt):
    import torch
    from thop import profile
    from torch import nn
    from torchvision import models

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    x = torch.randn(*INPUT_SHAPE)
    macs, params = profile(model, inputs=(x,), verbose=False)

    def infer():
        with torch.no_grad():
            model(x)

    return {"framework": "torch", "params_million": params / 1e6, "macs_giga": macs / 1e9,
            "size_mb": Path(ckpt).stat().st_size / 2**20, **timed(infer)}


def measure_onnx(path):
    import onnx
    import onnxruntime as ort

    graph = onnx.load(path).graph
    params = sum(int(np.prod(i.dims)) for i in graph.initializer)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    x = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    return {"framework": "onnxruntime", "params_million": params / 1e6,
            "size_mb": Path(path).stat().st_size / 2**20,
            **timed(lambda: session.run(None, {name: x}))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["torch", "onnx"])
    parser.add_argument("path")
    args = parser.parse_args()
    result = measure_torch(args.path) if args.kind == "torch" else measure_onnx(args.path)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}, indent=2))
