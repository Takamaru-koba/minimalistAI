"""Accuracy and per-class recall on the validation split (Section 5.1 protocol).

Uses val_split.json when present (official). Otherwise falls back to a seeded,
stratified provisional_split.json -- NOT comparable with the judges' split.

    python evaluate.py torch checkpoints/student_mobilenetv2_distilled.pt
    python evaluate.py onnx  build/baseline_fp32.onnx
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).parent
NUM_CLASSES, SEED, VAL_FRACTION = 10, 0, 0.2

TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_split(targets: list[int]) -> tuple[dict[str, list[int]], str]:
    official = ROOT / "val_split.json"
    if official.exists():
        return json.loads(official.read_text()), "official"
    provisional = ROOT / "provisional_split.json"
    if not provisional.exists():
        rng = np.random.default_rng(SEED)
        labels = np.asarray(targets)
        train, val = [], []
        for c in range(NUM_CLASSES):
            idx = rng.permutation(np.flatnonzero(labels == c))
            cut = int(len(idx) * VAL_FRACTION)
            val += idx[:cut].tolist()
            train += idx[cut:].tolist()
        provisional.write_text(json.dumps({"train": sorted(train), "val": sorted(val)}))
    return json.loads(provisional.read_text()), "provisional"


def val_loader(batch_size: int) -> tuple[DataLoader, str]:
    dataset = datasets.EuroSAT(root=ROOT / "data", download=False, transform=TRANSFORM)
    split, kind = load_split(dataset.targets)
    return DataLoader(Subset(dataset, split["val"]), batch_size=batch_size), kind


def torch_predictor(ckpt: str):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    def predict(images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return model(images).argmax(dim=1).numpy()

    return predict, 64


def onnx_predictor(path: str):
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    batch = session.get_inputs()[0].shape[0]
    batch = batch if isinstance(batch, int) else 32

    def predict(images: torch.Tensor) -> np.ndarray:
        return session.run(None, {name: images.numpy()})[0].argmax(axis=1)

    return predict, batch


def evaluate(predict, loader: DataLoader) -> dict:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for images, labels in loader:
        for truth, pred in zip(labels.numpy(), predict(images)):
            confusion[truth, pred] += 1
    recall = confusion.diagonal() / np.maximum(confusion.sum(axis=1), 1)
    return {"accuracy": float(confusion.diagonal().sum() / confusion.sum()),
            "worst_class_accuracy": float(recall.min()),
            "per_class_recall": [round(float(r), 4) for r in recall],
            "n_val": int(confusion.sum())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["torch", "onnx"])
    parser.add_argument("path")
    args = parser.parse_args()
    predict, batch_size = (torch_predictor if args.kind == "torch" else onnx_predictor)(args.path)
    loader, split_kind = val_loader(batch_size)
    print(json.dumps({"split": split_kind, **evaluate(predict, loader)}, indent=2))
