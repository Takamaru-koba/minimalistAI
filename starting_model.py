import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225]),])
full_dataset = datasets.EuroSAT(root=Path("./data"), download=False, transform=transform)

# Use the provided split. Your own split will not be comparable, and judges score on this one.
split_indices = json.loads(Path("provisional_split.json").read_text())
train_loader = DataLoader(Subset(full_dataset, split_indices["train"]), batch_size=32, shuffle=True)
val_loader = DataLoader(Subset(full_dataset, split_indices["val"]), batch_size=32)
student = models.mobilenet_v2(weights=None)
student.classifier[1] = nn.Linear(student.last_channel, 10)
student.load_state_dict(torch.load("checkpoints/student_mobilenetv2_distilled.pt", map_location="cpu"))
student.eval()