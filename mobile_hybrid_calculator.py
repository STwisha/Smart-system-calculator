# mobile_hybrid_calculator_final.py
# Single-file trainer + eval + infer using MobileNetV2.
# Safe for Windows: num_workers=0, pin_memory=False.
# Behavior:
#  - Train head-only for head_epochs, then unfreeze and fine-tune full model.
#  - Stronger augmentations for train.
#  - Saves best checkpoint at ./checkpoints/mobile_hybrid_best.pth

import os, argparse, time
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

# -----------------------
# Helpers
# -----------------------
def device_get(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.3,0.3,0.2,0.05)], p=0.7),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

def load_data(data_root: str, batch_size:int, img_size=224):
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"

    train_tf, val_tf = build_transforms(img_size)
    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder missing: {train_dir}")

    # create small val split if not present
    if not val_dir.exists() or not any(val_dir.iterdir()):
        print("Creating val split (10%) from train...")
        val_dir.mkdir(parents=True, exist_ok=True)
        for cls in [p for p in train_dir.iterdir() if p.is_dir()]:
            dest = val_dir / cls.name
            dest.mkdir(parents=True, exist_ok=True)
            imgs = sorted([p for p in cls.glob("*.*") if p.is_file()])
            n = max(1, int(0.1 * len(imgs)))
            chosen = imgs[-n:]
            for p in chosen:
                p.rename(dest / p.name)

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, train_ds

def build_model(num_classes:int, pretrained=True):
    try:
        from torchvision.models import MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    except Exception:
        model = models.mobilenet_v2(pretrained=pretrained)
    in_f = model.classifier[1].in_features if hasattr(model, 'classifier') else model.last_channel
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, num_classes))
    return model

def save_ckpt(state, path="./checkpoints/mobile_hybrid_best.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print("Saved checkpoint:", path)

# -----------------------
# Training loops
# -----------------------
def evaluate(model, loader, device):
    model.eval()
    total=0; correct=0; loss_sum=0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb,yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            out = model(xb)
            loss_sum += crit(out,yb).item() * xb.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds==yb).sum().item()
            total += xb.size(0)
    return loss_sum / max(1,total), correct / max(1,total)

def train_procedure(model, train_loader, val_loader, device, total_epochs=20, head_epochs=4, lr_head=1e-3, lr_ft=1e-4):
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    best_val = 0.0

    # ----- Phase 1: train head only -----
    print(f"Phase1: training head for {head_epochs} epochs, lr={lr_head}")
    # freeze backbone
    if hasattr(model, "features"):
        for p in model.features.parameters():
            p.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    for epoch in range(1, head_epochs+1):
        model.train()
        t0=time.time()
        running_loss=0.0; total=0; correct=0
        for xb,yb in train_loader:
            xb=xb.to(device); yb=yb.to(device)
            optimizer.zero_grad()
            out=model(xb)
            loss=crit(out,yb)
            loss.backward(); optimizer.step()
            running_loss += loss.item()*xb.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds==yb).sum().item()
            total += xb.size(0)
        train_loss = running_loss/max(1,total); train_acc = correct/max(1,total)
        val_loss, val_acc = evaluate(model, val_loader, device)
        sched.step(val_acc)
        print(f"Epoch {epoch}/{head_epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={time.time()-t0:.1f}s")
        if val_acc > best_val:
            best_val = val_acc
            save_ckpt({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_acc": val_acc})

    # ----- Phase 2: unfreeze and fine-tune -----
    remaining = max(0, total_epochs - head_epochs)
    if remaining <= 0:
        print("No fine-tune epochs requested. Done.")
        return model

    print(f"Phase2: fine-tune full model for {remaining} epochs, lr={lr_ft}")
    for p in model.features.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_ft)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    for e in range(1, remaining+1):
        epoch = head_epochs + e
        model.train()
        t0=time.time()
        running_loss=0.0; total=0; correct=0
        for xb,yb in train_loader:
            xb=xb.to(device); yb=yb.to(device)
            optimizer.zero_grad()
            out=model(xb)
            loss=crit(out,yb)
            loss.backward(); optimizer.step()
            running_loss += loss.item()*xb.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds==yb).sum().item()
            total += xb.size(0)
        train_loss = running_loss/max(1,total); train_acc = correct/max(1,total)
        val_loss, val_acc = evaluate(model, val_loader, device)
        sched.step(val_acc)
        print(f"Epoch {epoch}/{total_epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={time.time()-t0:.1f}s")
        if val_acc > best_val:
            best_val = val_acc
            save_ckpt({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_acc": val_acc})

    return model

# -----------------------
# Eval / Infer
# -----------------------
def evaluate_checkpoint(path, data_root):
    device = device_get()
    _, val_loader, train_ds = load_data(data_root, batch_size=16)
    num_classes = len(train_ds.classes)
    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck.get("model_state", ck))
    model.eval()
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"Eval -> val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

def infer_single(path_ckpt, image_path, data_root, img_size=224):
    device = device_get()
    _, val_tf = build_transforms(img_size)
    train_dir = Path(data_root) / "train"
    train_ds = datasets.ImageFolder(str(train_dir), transform=val_tf)
    num_classes = len(train_ds.classes)
    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    ck = torch.load(path_ckpt, map_location=device)
    model.load_state_dict(ck.get("model_state", ck))
    model.eval()
    img = Image.open(image_path).convert("RGB")
    x = val_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs)); label = train_ds.classes[idx]; conf = float(probs[idx])
    print("Predicted:", label, "confidence:", conf)
    return label, conf

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train","eval","infer"], required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--head_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_ft", type=float, default=1e-4)
    p.add_argument("--checkpoint", type=str, default="./checkpoints/mobile_hybrid_best.pth")
    p.add_argument("--image", type=str, help="path for infer")
    return p.parse_args()

def main():
    args = parse_args()
    dev = device_get()
    print("Device:", dev)
    if args.mode == "train":
        train_loader, val_loader, train_ds = load_data(args.data_root, batch_size=args.batch_size)
        print("Classes:", train_ds.classes)
        model = build_model(num_classes=len(train_ds.classes), pretrained=True)
        train_procedure(model, train_loader, val_loader, dev, total_epochs=args.epochs, head_epochs=args.head_epochs, lr_head=args.lr_head, lr_ft=args.lr_ft)
        print("Training finished.")
    elif args.mode == "eval":
        evaluate_checkpoint(args.checkpoint, args.data_root)
    elif args.mode == "infer":
        if not args.image: raise ValueError("Provide --image for infer")
        infer_single(args.checkpoint, args.image, args.data_root)

if __name__ == "__main__":
    main()
