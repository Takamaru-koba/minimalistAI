#!/usr/bin/env python3
"""
prune_sweep.py — structured channel pruning sweep for the Minimalist Edge AI Challenge.

For each prune ratio:
  load distilled MobileNetV2 -> rank filters by importance -> remove channels (Torch-Pruning)
  -> BN recalibration (cheap accuracy recovery, no training) -> evaluate -> save model
Writes sweep.csv: the data for your tradeoff curve.

SETUP (once):
  pip install torch-pruning

RUN (from the starter-kit folder: checkpoints/, data/, provisional_split.json):
  python prune_sweep.py                                  # L2 importance, ratios 0.05 0.10 0.20 0.30
  python prune_sweep.py --importance taylor              # gradient-based importance
  python prune_sweep.py --importance bn --ratios 0.1 0.2 # batch-norm gamma importance
  python prune_sweep.py --global_pruning                 # rank all filters together

OUTPUT:
  pruned/pruned_<importance>_<ratio>.pt   whole model (shape changed, so not a state_dict)
  sweep.csv                               ratio, accuracy, worst class, params, MACs, size

LOAD A PRUNED MODEL LATER:
  model = torch.load("pruned/pruned_l2_0.10.pt", map_location="cpu", weights_only=False)
"""
import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_pruning as tp
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

NUM_CLASSES = 10
STUDENT_CKPT = "checkpoints/student_mobilenetv2_distilled.pt"
SPLIT_FILE = "provisional_split.json"  # switch to val_split.json if organizers release it


# ------------------------------------------------------------------ data + model (Section 4 setup)

def get_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full = datasets.EuroSAT(root=Path("./data"), download=False, transform=transform)
    split = json.loads(Path(SPLIT_FILE).read_text())
    train_loader = DataLoader(Subset(full, split["train"]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full, split["val"]), batch_size=batch_size)
    return train_loader, val_loader


def load_student():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, NUM_CLASSES)
    m.load_state_dict(torch.load(STUDENT_CKPT, map_location="cpu"))
    return m.eval()


# ------------------------------------------------------------------ measurement

@torch.no_grad()
def evaluate(model, loader):
    """Returns (overall accuracy, worst-class recall), both in %."""
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for images, labels in loader:
        preds = model(images).argmax(dim=1)
        for t, p in zip(labels, preds):
            confusion[t, p] += 1
    acc = confusion.diag().sum().item() / confusion.sum().item()
    recall = confusion.diag().float() / confusion.sum(dim=1).clamp(min=1)
    return 100 * acc, 100 * recall.min().item()


def size_mb(model, path):
    torch.save(model, path)
    return os.path.getsize(path) / (1024 * 1024)


# ------------------------------------------------------------------ pruning

def make_importance(name):
    if name == "l2":
        return tp.importance.MagnitudeImportance(p=2)
    if name == "bn":
        return tp.importance.BNScaleImportance()
    if name == "taylor":
        return tp.importance.TaylorImportance()
    raise ValueError(name)


def prune(model, ratio, importance_name, train_loader, global_pruning=False, taylor_batches=10):
    model = copy.deepcopy(model)
    example = torch.randn(1, 3, 224, 224)
    importance = make_importance(importance_name)

    pruner = tp.pruner.MetaPruner(
        model,
        example,
        importance=importance,
        pruning_ratio=ratio,
        ignored_layers=[model.classifier[1]],  # must keep 10 outputs
        round_to=8,                            # NPU-friendly channel counts
        global_pruning=global_pruning,
    )

    if importance_name == "taylor":
        # Taylor importance needs gradients: accumulate a few batches of loss.backward()
        model.train()
        model.zero_grad()
        for i, (x, y) in enumerate(train_loader):
            if i >= taylor_batches:
                break
            F.cross_entropy(model(x), y).backward()

    pruner.step()
    model.zero_grad()
    return model.eval()


@torch.no_grad()
def recalibrate_bn(model, train_loader, batches=100):
    """Re-estimate BatchNorm running stats after pruning. No gradients, no weight updates."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative average over all recal batches
    model.train()
    for i, (x, _) in enumerate(train_loader):
        if i >= batches:
            break
        model(x)
    return model.eval()


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--importance", choices=["l2", "bn", "taylor"], default="l2")
    ap.add_argument("--global_pruning", action="store_true")
    ap.add_argument("--recal_batches", type=int, default=100, help="0 to skip BN recalibration")
    args = ap.parse_args()

    torch.manual_seed(0)
    os.makedirs("pruned", exist_ok=True)
    train_loader, val_loader = get_loaders()
    base = load_student()
    example = torch.randn(1, 3, 224, 224)

    rows = []

    def record(label, ratio, model, acc_before_recal=None):
        acc, worst = evaluate(model, val_loader)
        macs, params = tp.utils.count_ops_and_params(model, example)
        path = f"pruned/pruned_{args.importance}_{ratio:.2f}.pt"
        size = size_mb(model, path)
        row = dict(label=label, ratio=ratio, importance=args.importance,
                   acc_before_recal=acc_before_recal, accuracy=round(acc, 2),
                   worst_class=round(worst, 2), params_million=round(params / 1e6, 3),
                   macs_giga=round(macs / 1e9, 3), size_mb=round(size, 2), path=path)
        rows.append(row)
        print(f"  -> acc {acc:.2f}%  worst {worst:.2f}%  params {params/1e6:.2f}M  "
              f"MACs {macs/1e9:.3f}G  size {size:.2f}MB", flush=True)

    print("== Baseline (ratio 0)")
    record("baseline", 0.0, base)

    for ratio in args.ratios:
        t0 = time.time()
        print(f"== Prune {ratio:.0%} ({args.importance}{', global' if args.global_pruning else ''})")
        pruned = prune(base, ratio, args.importance, train_loader, args.global_pruning)
        before = None
        if args.recal_batches > 0:
            before, _ = evaluate(pruned, val_loader)
            print(f"  before BN recal: {before:.2f}%")
            recalibrate_bn(pruned, train_loader, args.recal_batches)
        record(f"pruned_{ratio:.2f}", ratio, pruned,
               acc_before_recal=round(before, 2) if before is not None else None)
        print(f"  ({time.time() - t0:.0f}s)")

    with open("sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nSaved sweep.csv and pruned/*.pt — copy the numbers into your Sweep table.")


if __name__ == "__main__":
    main()