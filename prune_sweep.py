#!/usr/bin/env python3
"""
prune_sweep.py — structured channel pruning sweep for the Minimalist Edge AI Challenge.

For each prune ratio:
  load distilled MobileNetV2 -> rank filters by importance -> remove channels (Torch-Pruning)
  -> BN recalibration (cheap accuracy recovery, no training)
  -> distilled fine-tune from the ResNet-50 teacher (--ft_epochs, 0 to skip) -> evaluate -> save model
Writes sweep_pass<A|B>_<importance>.csv: the data for your tradeoff curve.

SETUP (once):
  pip install torch-pruning

RUN (from the starter-kit folder: checkpoints/, data/, provisional_split.json):
  python prune_sweep.py                                  # L2 importance, ratios 0.05 0.10 0.20 0.30
  python prune_sweep.py --importance taylor              # gradient-based importance
  python prune_sweep.py --importance bn --ratios 0.1 0.2 # batch-norm gamma importance
  python prune_sweep.py --global_pruning                 # rank all filters together
  python prune_sweep.py --ft_epochs 3 --temperature 4    # longer distilled recovery
  python prune_sweep.py --ft_epochs 0                    # BN recalibration only (old behavior)

PASSES (see ARCHITECTURE.md):
  python prune_sweep.py --prune_pass A --ratios 0.1 0.2 0.3   # all channels (only the classifier is protected)
  python prune_sweep.py --prune_pass B --ratios 0.1 0.2 0.3   # hidden channels only (also protects stem + project convs)

OUTPUT:
  pruned/pruned_pass<A|B>_<importance>_<ratio>.pt   whole model (shape changed, so not a state_dict)
  sweep_pass<A|B>_<importance>.csv                  ratio, accuracy (final, after finetune), worst class, params, MACs, size

LOAD A PRUNED MODEL LATER:
  model = torch.load("pruned/pruned_passA_l2_0.10.pt", map_location="cpu", weights_only=False)
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
from torchvision.models.mobilenetv2 import InvertedResidual

NUM_CLASSES = 10
STUDENT_CKPT = "checkpoints/student_mobilenetv2_distilled.pt"
TEACHER_CKPT = "checkpoints/teacher_resnet50_eurosat.pt"
SPLIT_FILE = "provisional_split.json"  # switch to val_split.json if organizers release it


# ------------------------------------------------------------------ data + model (Section 4 setup)

def get_loaders(batch_size=32, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full = datasets.EuroSAT(root=Path("./data"), download=False, transform=transform)
    split = json.loads(Path(SPLIT_FILE).read_text())
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, persistent_workers=num_workers > 0)
    train_loader = DataLoader(Subset(full, split["train"]), shuffle=True, **loader_args)
    val_loader = DataLoader(Subset(full, split["val"]), **loader_args)
    return train_loader, val_loader


def load_student():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, NUM_CLASSES)
    m.load_state_dict(torch.load(STUDENT_CKPT, map_location="cpu"))
    return m.eval()


def load_teacher():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(TEACHER_CKPT, map_location="cpu"))
    for p in m.parameters():
        p.requires_grad_(False)
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


def protected_layers(model, prune_pass):
    """Layers Torch-Pruning must not shrink.

    Pass A: the classifier only (must keep 10 outputs); every other channel is prunable.
    Pass B: also the stem and each inverted-residual block's project conv, which feeds the skip
    connections, so only the wide hidden (expand / depthwise) channels can be cut.
    """
    ignored = [model.classifier[1]]
    if prune_pass == "B":
        ignored.append(model.features[0][0])  # stem conv
        ignored.extend(block.conv[-2] for block in model.features if isinstance(block, InvertedResidual))
    return ignored


def prune(model, ratio, importance_name, train_loader, global_pruning=False, taylor_batches=10,
          prune_pass="A"):
    model = copy.deepcopy(model)
    example = torch.randn(1, 3, 224, 224)
    importance = make_importance(importance_name)

    pruner = tp.pruner.MetaPruner(
        model,
        example,
        importance=importance,
        pruning_ratio=ratio,
        ignored_layers=protected_layers(model, prune_pass),
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


def distill_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """alpha * KL(teacher || student) at temperature T (scaled by T^2) + (1 - alpha) * CE."""
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * temperature ** 2
    return alpha * soft + (1 - alpha) * F.cross_entropy(student_logits, labels)


def finetune_distilled(model, teacher, train_loader, epochs, lr, temperature, alpha, device="cpu"):
    """Recover accuracy after pruning by distilling from the teacher. Updates model in place.

    Trains on `device` (teacher must already be there) and returns the model back on CPU, so
    saving and evaluation always happen on CPU.
    """
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            loss = distill_loss(model(x), t_logits, y, temperature, alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"  finetune epoch {epoch + 1}/{epochs}  loss {total / len(train_loader):.4f}", flush=True)
    return model.cpu().eval()


# ------------------------------------------------------------------ main

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--prune_pass", choices=["A", "B"], default="A",
                    help="A: all channels; B: hidden channels only (protect stem + project convs)")
    ap.add_argument("--importance", choices=["l2", "bn", "taylor"], default="l2")
    ap.add_argument("--global_pruning", action="store_true")
    ap.add_argument("--recal_batches", type=int, default=100, help="0 to skip BN recalibration")
    ap.add_argument("--ft_epochs", type=int, default=1, help="distilled fine-tune epochs per ratio; 0 to skip")
    ap.add_argument("--ft_lr", type=float, default=0.005)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.7, help="weight on the soft-label (teacher) loss")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto",
                    help="device for the distilled fine-tune only; pruning, BN recal and all evals stay on CPU")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    ap.add_argument("--eval_before_recal", action="store_true",
                    help="also evaluate right after pruning, before BN recal (slow, diagnostic only)")
    args = ap.parse_args()

    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    device = "cpu" if device == "auto" else device
    print(f"fine-tune device: {device}  workers: {args.workers}", flush=True)

    torch.manual_seed(0)
    os.makedirs("pruned", exist_ok=True)
    train_loader, val_loader = get_loaders(num_workers=args.workers)
    base = load_student()
    teacher = load_teacher().to(device) if args.ft_epochs > 0 else None
    example = torch.randn(1, 3, 224, 224)

    tag = f"pass{args.prune_pass}_{args.importance}"
    csv_path = f"sweep_{tag}.csv"
    rows = []

    def record(label, ratio, model, acc_before_recal=None, acc_after_recal=None):
        acc, worst = evaluate(model, val_loader)
        macs, params = tp.utils.count_ops_and_params(model, example)
        path = f"pruned/pruned_pass{args.prune_pass}_{args.importance}_{ratio:.2f}.pt"
        size = size_mb(model, path)
        row = dict(label=label, ratio=ratio, prune_pass=args.prune_pass, importance=args.importance,
                   acc_before_recal=acc_before_recal, acc_after_recal=acc_after_recal, accuracy=round(acc, 2),
                   worst_class=round(worst, 2), params_million=round(params / 1e6, 3),
                   macs_giga=round(macs / 1e9, 3), size_mb=round(size, 2), path=path)
        rows.append(row)
        print(f"  -> acc {acc:.2f}%  worst {worst:.2f}%  params {params/1e6:.2f}M  "
              f"MACs {macs/1e9:.3f}G  size {size:.2f}MB", flush=True)

    print("== Baseline (ratio 0)")
    record("baseline", 0.0, base)

    for ratio in args.ratios:
        t0 = time.time()
        print(f"== Pass {args.prune_pass}: prune {ratio:.0%} ({args.importance}{', global' if args.global_pruning else ''})")
        pruned = prune(base, ratio, args.importance, train_loader, args.global_pruning,
                       prune_pass=args.prune_pass)
        before = None
        if args.recal_batches > 0:
            if args.eval_before_recal:
                before, _ = evaluate(pruned, val_loader)
                print(f"  before BN recal: {before:.2f}%")
            recalibrate_bn(pruned, train_loader, args.recal_batches)
        after_recal = None
        if teacher is not None:
            after_recal, _ = evaluate(pruned, val_loader)
            print(f"  after BN recal, before finetune: {after_recal:.2f}%")
            finetune_distilled(pruned, teacher, train_loader, args.ft_epochs,
                               args.ft_lr, args.temperature, args.alpha, device)
        record(f"pruned_{ratio:.2f}", ratio, pruned,
               acc_before_recal=round(before, 2) if before is not None else None,
               acc_after_recal=round(after_recal, 2) if after_recal is not None else None)
        write_csv(csv_path, rows)
        print(f"  ({time.time() - t0:.0f}s)")

    print(f"\nSaved {csv_path} and pruned/*.pt — copy the numbers into your Sweep table.")


if __name__ == "__main__":
    main()