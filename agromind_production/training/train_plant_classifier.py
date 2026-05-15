"""
training/train_plant_classifier.py

Trains EfficientNet-B0 on PlantVillage dataset for plant disease classification.

Dataset: PlantVillage (Hughes & Salathé, 2015)
  - 54,306 images
  - 38 classes (14 crops × multiple diseases + healthy)
  - Source: https://github.com/spMohanty/PlantVillage-Dataset
  - Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease

Architecture: EfficientNet-B0 (pretrained on ImageNet)
  - Fine-tuned for 38-class plant disease classification
  - Input: 224×224 RGB images
  - Output: 38-class softmax probabilities

Run on Google Colab T4 GPU (~3 hours):
  !pip install torch torchvision timm kaggle
  !kaggle datasets download -d emmarex/plantdisease
  !python train_plant_classifier.py
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)


# ─── CONFIGURATION ────────────────────────────────────────────────────
CONFIG = {
    # Dataset
    "data_dir":        "PlantVillage",        # extracted dataset folder
    "model_save_path": "models/plant_classifier.pth",
    "classes_path":    "models/plant_classes.json",
    "metrics_path":    "models/plant_metrics.json",

    # Architecture
    "model_name":      "efficientnet_b0",     # timm model name
    "num_classes":     38,
    "pretrained":      True,
    "image_size":      224,

    # Training
    "batch_size":      32,
    "num_epochs":      25,
    "learning_rate":   1e-4,
    "weight_decay":    1e-4,
    "dropout":         0.3,

    # Splits (train/val/test)
    "train_ratio":     0.70,
    "val_ratio":       0.15,
    "test_ratio":      0.15,

    # Hardware
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers":     4,
}

# PlantVillage 38 classes (official)
PLANT_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]


# ─── DATA TRANSFORMS ─────────────────────────────────────────────────
def get_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Training: aggressive augmentation to improve generalization
    Validation/Test: no augmentation, only resize + normalize
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.1
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.1),          # occlusion augmentation
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, val_transform


# ─── MODEL DEFINITION ────────────────────────────────────────────────
def build_model(num_classes: int, dropout: float) -> nn.Module:
    """
    EfficientNet-B0 with custom classification head.
    Pretrained on ImageNet, fine-tuned for PlantVillage 38-class task.
    """
    model = timm.create_model(
        CONFIG["model_name"],
        pretrained=CONFIG["pretrained"],
        num_classes=0,          # remove default head
        drop_rate=dropout,
    )

    # Get feature dimension
    in_features = model.num_features

    # Custom classification head
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(dropout * 0.5),
        nn.Linear(512, num_classes),
    )

    return model


# ─── TRAINING LOOP ───────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    scaler: torch.cuda.amp.GradScaler
) -> Tuple[float, float]:
    """Single training epoch with mixed precision."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    class_names: list
) -> Dict:
    """Full evaluation with per-class metrics."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy  = correct / total
    macro_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_pre = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "loss":           total_loss / total,
        "accuracy":       round(accuracy, 4),
        "macro_f1":       round(macro_f1, 4),
        "macro_precision":round(macro_pre, 4),
        "macro_recall":   round(macro_rec, 4),
        "per_class":      report,
        "confusion_matrix": conf_matrix,
        "num_samples":    total,
    }


# ─── MAIN TRAINING FUNCTION ──────────────────────────────────────────
def train():
    print("=" * 60)
    print("AgroMind — Plant Disease Classifier Training")
    print(f"Model: {CONFIG['model_name']} | Dataset: PlantVillage")
    print(f"Device: {CONFIG['device']}")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    device = CONFIG["device"]

    # ── Load dataset ─────────────────────────────────────────────
    print("\n[1/5] Loading PlantVillage dataset...")
    train_transform, val_transform = get_transforms(CONFIG["image_size"])

    full_dataset = datasets.ImageFolder(
        root=CONFIG["data_dir"],
        transform=train_transform
    )

    class_names = full_dataset.classes
    print(f"  Classes found: {len(class_names)}")
    print(f"  Total images:  {len(full_dataset)}")

    # Save class names
    with open(CONFIG["classes_path"], "w") as f:
        json.dump({
            "classes":      class_names,
            "class_to_idx": full_dataset.class_to_idx,
            "num_classes":  len(class_names),
            "dataset":      "PlantVillage",
            "source":       "https://github.com/spMohanty/PlantVillage-Dataset",
        }, f, indent=2)

    # ── Split dataset ─────────────────────────────────────────────
    n = len(full_dataset)
    n_train = int(n * CONFIG["train_ratio"])
    n_val   = int(n * CONFIG["val_ratio"])
    n_test  = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply val transform to val/test sets
    val_set.dataset.transform  = val_transform
    test_set.dataset.transform = val_transform

    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # ── Build model ───────────────────────────────────────────────
    print("\n[2/5] Building EfficientNet-B0 model...")
    model = build_model(len(class_names), CONFIG["dropout"]).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable:,}")

    # ── Loss + Optimizer ──────────────────────────────────────────
    # Class weights for imbalanced classes
    class_counts = np.array([
        len([s for s in train_set.indices if full_dataset.targets[s] == i])
        for i in range(len(class_names))
    ], dtype=np.float32)
    class_weights = torch.tensor(
        1.0 / (class_counts + 1e-6), dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # ── Training loop ─────────────────────────────────────────────
    print("\n[3/5] Training...")
    best_val_f1 = 0.0
    history = []

    for epoch in range(CONFIG["num_epochs"]):
        t0 = time.time()
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_metrics = evaluate(model, val_loader, criterion, device, class_names)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

        history.append({
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["accuracy"],
            "val_f1":     val_metrics["macro_f1"],
        })

        # Save best model by validation F1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_f1":      best_val_f1,
                "val_acc":     val_metrics["accuracy"],
                "config":      CONFIG,
                "classes":     class_names,
            }, CONFIG["model_save_path"])
            print(f"  [OK] Saved best model (F1: {best_val_f1:.4f})")

    # ── Final test evaluation ─────────────────────────────────────
    print("\n[4/5] Final evaluation on held-out test set...")
    checkpoint = torch.load(CONFIG["model_save_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device, class_names)

    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"  Test Precision: {test_metrics['macro_precision']:.4f}")
    print(f"  Test Recall:    {test_metrics['macro_recall']:.4f}")

    # ── Save metrics ──────────────────────────────────────────────
    print("\n[5/5] Saving metrics...")
    metrics = {
        "model":          CONFIG["model_name"],
        "dataset":        "PlantVillage",
        "dataset_source": "https://github.com/spMohanty/PlantVillage-Dataset",
        "num_classes":    len(class_names),
        "num_samples":    n,
        "splits":         {"train": n_train, "val": n_val, "test": n_test},
        "architecture": {
            "base":        "EfficientNet-B0",
            "pretrained":  "ImageNet",
            "image_size":  CONFIG["image_size"],
            "dropout":     CONFIG["dropout"],
        },
        "hyperparameters": {
            "batch_size":    CONFIG["batch_size"],
            "num_epochs":    CONFIG["num_epochs"],
            "learning_rate": CONFIG["learning_rate"],
            "weight_decay":  CONFIG["weight_decay"],
            "optimizer":     "AdamW",
            "scheduler":     "CosineAnnealingLR",
            "loss":          "CrossEntropyLoss (class-weighted)",
            "augmentation":  [
                "RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip",
                "RandomRotation(30°)", "ColorJitter", "RandomErasing"
            ],
        },
        "test_metrics": {
            "accuracy":        test_metrics["accuracy"],
            "macro_f1":        test_metrics["macro_f1"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall":    test_metrics["macro_recall"],
        },
        "training_history": history,
        "per_class_metrics": test_metrics["per_class"],
        "confusion_matrix":  test_metrics["confusion_matrix"],
        "classes":           class_names,
    }

    with open(CONFIG["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Model saved:   {CONFIG['model_save_path']}")
    print(f"  Classes saved: {CONFIG['classes_path']}")
    print(f"  Metrics saved: {CONFIG['metrics_path']}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1:       {test_metrics['macro_f1']:.4f}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    train()
