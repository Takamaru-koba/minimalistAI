# MobileNetV2 Architecture Reference

Oct 13, 2026 · @Jacob Que

## Overview

MobileNetV2 is a small convolutional network built for phones. It looks at an image in stages, from simple details to whole scenes, and ends with a score for each of the 10 EuroSAT classes.

- **Size:** about 2.24 M parameters. Most of them sit in the last few layers.
- **Why it's cheap:** it filters each channel on its own (depthwise), then mixes channels with a cheap 1×1 step, instead of doing both at once.
- **Our model:** the standard architecture with a 10-class output, trained on EuroSAT satellite images by distilling from a ResNet-50.

## The building block

&#91;embedded content: inverted residual block · expand, depthwise, project, skip\]

The expand and depthwise steps work in the wide 6c space; pruning those hidden channels shrinks the block without touching its neighbors, while the thin outputs are shared through the skip.

## What each stage detects

Early layers see small details; each stage after that combines them into bigger, more meaningful patterns, until the last layers recognize whole land-use types.

Each filter in a layer detects one feature, so a layer's channel count is how many features it looks for. These are **typical** patterns for CNNs; the network learns its own features, so individual filters can differ.

| Stage | Layers | Features (channels out) | Image size | Share of params | What it typically detects | In EuroSAT |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Close-up details | features.0–3 | 32 → 24 | 112 → 56 px | under 1% | Edges, colors, simple textures | Green of vegetation, blue of water, gray of concrete, foliage grain |
| 2. Small patterns | features.4–10 | 32 → 64 | 28 → 14 px | about 10% | Repeated textures, lines, small shapes | Crop rows, road lines, field boundaries, rooftop grids |
| 3. Structures | features.11–16 | 96 → 160 | 14 → 7 px | about 49% | Larger shapes and how parts are arranged | A river's curve, a highway through land, a block of buildings, a shoreline |
| 4. Class evidence | features.17–18 | 320 → 1280 | 7 px | about 40% | Combinations that point to one class | "Even rows + green + no buildings" suggests a crop class |
| 5. Decision | classifier | 10 scores | — | under 1% | Weighs the 1,280 features into one score per class | AnnualCrop, Forest, … SeaLake |

**Two ways to check what your model actually uses:**

1. **Look at feature maps.** Run one image (say, a River) through the model and plot a few channels from features.2, features.8 and features.15. Early maps look like edge-traced photos; late maps are blobs that light up on the river.
2. **Per-class sensitivity.** Prune one stage at a time and see which classes drop. If cutting stage 3 mostly hurts PermanentCrop, that stage holds features PermanentCrop depends on.

## What this means for pruning today

Cut where the parameters are, protect where the details start.

- **Biggest savings:** stages 3–4 hold about 89% of the parameters, so pruning there shrinks the model most.
- **Safest channels:** the wide hidden channels inside each block (highlighted in the diagram).
- **Handle with care:** stage 1 holds almost no parameters but feeds everything after it, so pruning it saves little and risks a lot.
- **Never prune:** the classifier, which must keep its 10 outputs.

**Pruning plan: all channels first, then hidden channels only**

1. **Pass A, all channels:** prune every prunable channel (block outputs and hidden channels) at 10%, 20% and 30%. This is the baseline curve.
2. **Pass B, hidden channels only:** protect each block's project conv, the stem and the classifier, so only the wide hidden channels (the highlighted boxes in the diagram) can be cut. Same ratios.
3. **Compare** accuracy and parameter count at each ratio, and keep whichever loses less accuracy per parameter saved.

**Why try both:** block outputs are shared through skip connections, so cutting them disturbs several blocks at once. Hidden channels live inside one block, so cutting only those should cost less accuracy, but fewer channels are eligible, so each ratio saves fewer parameters.

## Sources

- Sandler et al., [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) (2018): the inverted residual block.
- [torchvision mobilenetv2.py](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py): layer sizes, strides and the skip rule; parameter shares are computed from it.
- Stanford CS231n, [Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) lecture notes: filters, depth and channels (one activation map per filter), parameters per conv layer, and early layers responding to edges and color blobs while higher layers respond to larger patterns.
- Zeiler and Fergus, [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) (2013): the technique for seeing what intermediate layers detect. The stage descriptions above are the general early-to-late pattern, with EuroSAT examples as illustrations.
- MIT 6.5940 EfficientML.ai, Lectures 3–4: pruning implications.
