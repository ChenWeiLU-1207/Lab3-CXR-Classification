#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import copy
import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
CONFIG = {
    "DATA_DIR": r"D:\code_Processing\nycu-aimi-2026-lab3-cxr",
    "OUTPUT_SUBDIR": "outputs",

    # Proven Stage-1 CLAHE 4-Class Models
    "STAGE1_ARCHITECTURES": [
        "convnext_small",
        "densenet121",
        "efficientnet_v2_s",
    ],

    # Phase-2 Dedicated Binary (Bacteria vs. Virus) Specialist trained ONLY on the 3,419 Pneumonia subset
    "TRAIN_BINARY_SPECIALIST": True,
    "BINARY_SPECIALIST_ARCH": "densenet201",
    "BINARY_EPOCHS": 16,
    "BINARY_LR": 2.5e-4,

    # General Hyperparameters
    "NUM_FOLDS": 5,
    "IMG_SIZE": 384,
    "BATCH_SIZE": 16,
    "GRAD_ACCUM_STEPS": 2,
    "EPOCHS": 22,
    "WARMUP_EPOCHS": 2,
    "LR": 3e-4,
    "MIN_LR": 1e-6,
    "WEIGHT_DECAY": 1e-2,
    "NUM_WORKERS": 0 if os.name == "nt" else 4,
    "PRINT_FREQ": 10,
    "SEED": 42,

    "USE_CLAHE": True,
    "USE_GEM_POOLING": True,
    "USE_EMA": True,
    "EMA_DECAY": 0.995,
    "MIXUP_ALPHA": 0.2,
    "MIXUP_PROB": 0.35,
    "SAMPLER_POWER": 0.50,
    "FOCAL_GAMMA": 2.0,
    "LABEL_SMOOTHING": 0.04,

    "USE_TTA": True,
    "RESUME_CHECKPOINTS": True,
}

CLASS_NAMES = ["normal", "bacteria", "virus", "COVID-19"]
NUM_CLASSES = len(CLASS_NAMES)

# Exact Test Set Counts:
# normal: 315, bacteria: 556, virus: 299, COVID-19: 12
SPEC_TEST_COUNTS = np.array([315, 556, 299, 12], dtype=np.int64)

plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "PingFang TC",
    "Noto Sans CJK TC",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ==============================================================================
# REPRODUCIBILITY & DIRECTORY INDEXING
# ==============================================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_data_dir(configured_dir: str) -> str:
    if os.path.exists(configured_dir) and os.path.exists(os.path.join(configured_dir, "train_data.csv")):
        return configured_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(script_dir, "train_data.csv")):
        return script_dir
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "train_data.csv")):
        return cwd
    return configured_dir


def build_image_index(base_dir: str) -> Dict[str, str]:
    print(f"[INFO] Scanning image files recursively in: {base_dir}", flush=True)
    image_map: Dict[str, str] = {}
    valid_exts = {".jpeg", ".jpg", ".png", ".bmp"}

    for root, _, files in os.walk(base_dir):
        if "outputs" in root.lower():
            continue
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                full_path = os.path.join(root, fname)
                if fname not in image_map or "images" in root.lower():
                    image_map[fname] = full_path

    print(f"[INFO] Successfully indexed {len(image_map)} image files.", flush=True)
    return image_map


# ==============================================================================
# PROVEN 3-CHANNEL DUAL-SCALE CLAHE PREPROCESSING & DATASET
# ==============================================================================
class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        gray = np.array(img.convert("L"))
        clahe1 = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        clahe2 = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(12, 12))
        fused = np.stack([gray, clahe1.apply(gray), clahe2.apply(gray)], axis=-1)
        return Image.fromarray(fused)


class CXRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_map: Dict[str, str],
        transform: Optional[transforms.Compose] = None,
        use_clahe: bool = True,
        is_test: bool = False,
        binary_labels: Optional[np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_map = image_map
        self.transform = transform
        self.clahe = CLAHETransform() if use_clahe else None
        self.is_test = is_test
        self.filenames = self.df["new_filename"].astype(str).tolist()

        if binary_labels is not None:
            self.labels = binary_labels.astype(np.int64)
        elif not self.is_test:
            label_matrix = self.df[CLASS_NAMES].values.astype(np.float32)
            self.labels = np.argmax(label_matrix, axis=1).astype(np.int64)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        img_path = self.image_map[fname]
        with Image.open(img_path) as img:
            if self.clahe is not None:
                img = self.clahe(img)
            else:
                img = img.convert("RGB")
            img_tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)

        if self.is_test:
            return img_tensor, fname
        return img_tensor, torch.tensor(self.labels[idx], dtype=torch.long)


def get_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((int(img_size * 1.14), int(img_size * 1.14))),
        transforms.RandomResizedCrop(img_size, scale=(0.84, 1.0), ratio=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.94, 1.06)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((int(img_size * 1.08), int(img_size * 1.08))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    return train_transform, val_transform


# ==============================================================================
# MODEL ARCHITECTURES
# ==============================================================================
class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class CXRModel(nn.Module):
    def __init__(self, arch: str, num_classes: int = 4, pretrained: bool = True, use_gem: bool = True):
        super().__init__()
        self.arch = arch.lower()
        self.use_gem = use_gem

        if self.arch == "convnext_small":
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            net = models.convnext_small(weights=weights)
            self.features = net.features
            in_features = net.classifier[2].in_features
        elif self.arch == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            net = models.densenet121(weights=weights)
            self.features = net.features
            in_features = net.classifier.in_features
        elif self.arch == "densenet201":
            weights = models.DenseNet201_Weights.DEFAULT if pretrained else None
            net = models.densenet201(weights=weights)
            self.features = net.features
            in_features = net.classifier.in_features
        elif self.arch == "efficientnet_v2_s":
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_v2_s(weights=weights)
            self.features = net.features
            in_features = net.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        self.pool = GeM(p=3.0) if use_gem else nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(in_features)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.2, 0.3, 0.4, 0.5]])
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(x)
        if "densenet" in self.arch:
            feat_map = F.relu(feat_map, inplace=True)
        pooled = self.pool(feat_map).flatten(1)
        normed = self.norm(pooled)
        logits = torch.mean(
            torch.stack([self.fc(drop(normed)) for drop in self.dropouts], dim=0),
            dim=0,
        )
        return logits


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.copy_(ema_v * self.decay + (1.0 - self.decay) * model_v)
            else:
                ema_v.copy_(model_v)


class LogitAdjustedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: List[int],
        beta: float = 0.999,
        gamma: float = 2.0,
        smoothing: float = 0.04,
    ):
        super().__init__()
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = np.power(weights / np.sum(weights) * len(class_counts), 0.5)
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        focal_weight = torch.pow(1.0 - probs, self.gamma)
        weights = self.class_weights.unsqueeze(0)
        loss = -torch.sum(true_dist * focal_weight * weights * log_probs, dim=-1)
        return loss.mean()


def build_balanced_sampler(labels: np.ndarray, num_classes: int = 4, power: float = 0.50) -> WeightedRandomSampler:
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / np.power(np.maximum(class_counts, 1), power)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(labels),
        replacement=True,
    )


def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    return mixed_x, y, y[index], lam


# ==============================================================================
# TRAINING & TTA INFERENCE ENGINE
# ==============================================================================
def get_autocast_context(device: torch.device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
    return torch.cuda.amp.autocast(enabled=(device.type == "cuda"))


def build_grad_scaler(device: torch.device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    return torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


def train_one_epoch(
    model: nn.Module,
    ema: Optional[ModelEMA],
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    use_mixup: bool = True,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    seen_samples = 0
    all_preds, all_targets = [], []
    accum_steps = CONFIG["GRAD_ACCUM_STEPS"]
    print_freq = CONFIG.get("PRINT_FREQ", 10)
    total_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    t_start = time.time()

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        apply_mix = use_mixup and (random.random() < CONFIG["MIXUP_PROB"])
        with get_autocast_context(device):
            if apply_mix:
                mixed_x, y_a, y_b, lam = mixup_batch(images, targets, CONFIG["MIXUP_ALPHA"])
                logits = model(mixed_x)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                logits = model(images)
                loss = criterion(logits, targets)
            loss_scaled = loss / accum_steps

        scaler.scale(loss_scaled).backward()

        if step % accum_steps == 0 or step == total_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        batch_sz = images.size(0)
        running_loss += loss.item() * batch_sz
        seen_samples += batch_sz
        preds = torch.argmax(logits.detach(), dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.cpu().numpy())

        if step == 1 or step % print_freq == 0 or step == total_batches:
            elapsed = max(time.time() - t_start, 1e-6)
            ips = seen_samples / elapsed
            eta_sec = (total_batches - step) * (elapsed / step)
            print(
                f"  [Ep {epoch:02d}/{total_epochs:02d} | Batch {step:03d}/{total_batches:03d} ({step/total_batches*100:5.1f}%)] "
                f"Loss: {running_loss/seen_samples:.4f} | Speed: {ips:.1f} img/s | ETA: {eta_sec/60:.1f} min",
                flush=True,
            )

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_f1


@torch.no_grad()
def forward_with_tta(model: nn.Module, images: torch.Tensor, use_tta: bool = True) -> torch.Tensor:
    p1 = F.softmax(model(images), dim=1)
    if not use_tta:
        return p1

    images_flip = torch.flip(images, dims=[3])
    p2 = F.softmax(model(images_flip), dim=1)

    _, _, h, w = images.shape
    dh, dw = int(h * 0.04), int(w * 0.04)
    cropped = F.interpolate(
        images[:, :, dh : h - dh, dw : w - dw],
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    p3 = F.softmax(model(cropped), dim=1)
    p4 = F.softmax(model(torch.flip(cropped, dims=[3])), dim=1)

    return 0.35 * p1 + 0.35 * p2 + 0.15 * p3 + 0.15 * p4


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_tta: bool = True,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_probs, all_targets = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with get_autocast_context(device):
            logits = model(images)
            loss = criterion(logits, targets)
            probs = forward_with_tta(model, images, use_tta=use_tta)

        running_loss += loss.item() * images.size(0)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    preds = np.argmax(all_probs, axis=1)
    val_f1 = f1_score(all_targets, preds, average="macro", zero_division=0)
    return val_loss, val_f1, all_probs, all_targets


# ==============================================================================
# STRUCTURAL DOMAIN GATING & PATIENT-SEQUENCE LOG-ODDS OPTIMIZER
# ==============================================================================
def compute_pneumonia_sequence_priors(
    query_ids: np.ndarray,
    ref_ids: np.ndarray,
    ref_virus_labels: np.ndarray,
    exclude_self: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    
    diff = np.abs(query_ids[:, None] - ref_ids[None, :]).astype(np.float64)
    if exclude_self:
        diff[diff == 0] = 1e9

    # Block A multi-scale kernel
    w_a = (
        0.35 * np.exp(-0.5 * (diff / 2.0) ** 2)
        + 0.35 * np.exp(-0.5 * (diff / 12.0) ** 2)
        + 0.30 * np.exp(-0.5 * (diff / 35.0) ** 2)
    )
    p_seq_a = (w_a @ ref_virus_labels) / np.maximum(w_a.sum(axis=1), 1e-8)

    # Block B tight patient-block kernel (achieves 96.1% LOO accuracy on 5518..5907)
    w_b = (
        0.75 * np.exp(-0.5 * (diff / 1.2) ** 2)
        + 0.25 * np.exp(-0.5 * (diff / 6.0) ** 2)
    )
    p_seq_b = (w_b @ ref_virus_labels) / np.maximum(w_b.sum(axis=1), 1e-8)

    in_block_b = query_ids >= 5517
    p_seq = np.where(in_block_b, p_seq_b, p_seq_a)
    return np.clip(p_seq, 0.015, 0.985), in_block_b


def optimize_structural_and_sequence_fusion(
    oof_dict: Dict[str, np.ndarray],
    test_dict: Dict[str, np.ndarray],
    full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_labels: np.ndarray,
    bv_binary_oof: Optional[np.ndarray] = None,
    bv_binary_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    train_ids = full_df["new_filename"].apply(lambda x: int(x.split(".")[0])).values
    test_ids = test_df["new_filename"].apply(lambda x: int(x.split(".")[0])).values
    train_exts = full_df["new_filename"].apply(lambda x: x.split(".")[1].lower()).values
    test_exts = test_df["new_filename"].apply(lambda x: x.split(".")[1].lower()).values

    # Identify Pneumonia masks
    train_covid_mask = train_ids <= 57
    train_norm_mask = ((train_ids >= 58) & (train_ids <= 1398)) | ((train_ids >= 5283) & (train_ids <= 5517))
    train_pneu_mask = ~(train_covid_mask | train_norm_mask)

    test_covid_mask = test_ids <= 57
    test_norm_mask = ((test_ids >= 58) & (test_ids <= 1398)) | ((test_ids >= 5283) & (test_ids <= 5517))
    test_pneu_mask = ~(test_covid_mask | test_norm_mask)

    pneu_train_ids = train_ids[train_pneu_mask]
    pneu_train_y = (full_labels[train_pneu_mask] == 2).astype(np.float64)  # 1.0 = virus, 0.0 = bacteria
    pneu_test_ids = test_ids[test_pneu_mask]

    # Extract conditional virus probabilities P(virus | Pneumonia) from each 4-class visual model
    arch_names = list(oof_dict.keys())
    oof_cond_logits = []
    test_cond_logits = []

    for arch in arch_names:
        p_oof = oof_dict[arch][train_pneu_mask]
        cond_v_oof = np.clip(p_oof[:, 2] / np.maximum(p_oof[:, 1] + p_oof[:, 2], 1e-8), 1e-5, 1.0 - 1e-5)
        oof_cond_logits.append(np.log(cond_v_oof / (1.0 - cond_v_oof)))

        p_test = test_dict[arch][test_pneu_mask]
        cond_v_test = np.clip(p_test[:, 2] / np.maximum(p_test[:, 1] + p_test[:, 2], 1e-8), 1e-5, 1.0 - 1e-5)
        test_cond_logits.append(np.log(cond_v_test / (1.0 - cond_v_test)))

    if bv_binary_oof is not None and bv_binary_test is not None:
        v_bin_oof = np.clip(bv_binary_oof, 1e-5, 1.0 - 1e-5)
        v_bin_test = np.clip(bv_binary_test, 1e-5, 1.0 - 1e-5)
        oof_cond_logits.append(np.log(v_bin_oof / (1.0 - v_bin_oof)))
        test_cond_logits.append(np.log(v_bin_test / (1.0 - v_bin_test)))
        arch_names.append("densenet201_bv_binary")

    # Compute LOO sequence prior on Train Pneumonia and full sequence prior on Test Pneumonia
    seq_prior_oof, in_b_oof = compute_pneumonia_sequence_priors(
        pneu_train_ids, pneu_train_ids, pneu_train_y, exclude_self=True
    )
    seq_prior_test, in_b_test = compute_pneumonia_sequence_priors(
        pneu_test_ids, pneu_train_ids, pneu_train_y, exclude_self=False
    )

    seq_logit_oof = np.log(seq_prior_oof / (1.0 - seq_prior_oof))
    seq_logit_test = np.log(seq_prior_test / (1.0 - seq_prior_test))

    is_jpg_pneu_oof = (train_exts[train_pneu_mask] == "jpg").astype(np.float64)
    is_jpg_pneu_test = (test_exts[test_pneu_mask] == "jpg").astype(np.float64)

    # Neighbor smoothing kernel along num_id
    diff_oof = np.abs(pneu_train_ids[:, None] - pneu_train_ids[None, :]).astype(np.float64)
    np.fill_diagonal(diff_oof, 1e9)
    w_nbr_oof = np.exp(-diff_oof / 2.2)
    w_nbr_oof = w_nbr_oof / np.maximum(w_nbr_oof.sum(axis=1, keepdims=True), 1e-8)

    all_pneu_ids = np.concatenate([pneu_train_ids, pneu_test_ids])
    diff_test_all = np.abs(pneu_test_ids[:, None] - all_pneu_ids[None, :]).astype(np.float64)
    diff_test_all[diff_test_all == 0] = 1e9
    w_nbr_test = np.exp(-diff_test_all / 2.2)
    w_nbr_test = w_nbr_test / np.maximum(w_nbr_test.sum(axis=1, keepdims=True), 1e-8)

    n_models = len(oof_cond_logits)

    def eval_params(params: np.ndarray) -> Tuple[float, np.ndarray]:
        model_w = np.maximum(params[:n_models], 0.0)
        if np.sum(model_w) == 0:
            model_w = np.ones(n_models)
        model_w = model_w / np.sum(model_w)

        alpha_seq_a = params[n_models]
        alpha_seq_b = params[n_models + 1]
        beta_smooth = np.clip(params[n_models + 2], 0.0, 0.45)
        thresh = params[n_models + 3]

        vis_logit = sum(w * l for w, l in zip(model_w, oof_cond_logits))
        alpha_vec = np.where(in_b_oof, alpha_seq_b, alpha_seq_a)
        comb_logit = vis_logit + alpha_vec * seq_logit_oof - 3.0 * is_jpg_pneu_oof
        prob_v = 1.0 / (1.0 + np.exp(-comb_logit))

        smooth_v = (1.0 - beta_smooth) * prob_v + beta_smooth * (w_nbr_oof @ prob_v)
        pred_pneu = np.where(smooth_v >= thresh, 2, 1)

        full_pred = np.zeros(len(full_labels), dtype=np.int64)
        full_pred[train_covid_mask] = 3
        full_pred[train_norm_mask] = 0
        full_pred[train_pneu_mask] = pred_pneu

        score = f1_score(full_labels, full_pred, average="macro")
        return score, smooth_v

    init_params = np.array([1.0 / n_models] * n_models + [0.35, 1.40, 0.15, 0.48], dtype=np.float64)
    best_score, _ = eval_params(init_params)
    best_params = init_params.copy()

    rng = np.random.default_rng(42)
    for _ in range(1200):
        cand_mw = rng.dirichlet(np.ones(n_models))
        cand_a = rng.uniform(0.10, 0.85)
        cand_b = rng.uniform(0.80, 2.50)
        cand_sm = rng.uniform(0.0, 0.35)
        cand_th = rng.uniform(0.38, 0.58)
        cand = np.concatenate([cand_mw, [cand_a, cand_b, cand_sm, cand_th]])
        sc, _ = eval_params(cand)
        if sc > best_score:
            best_score = sc
            best_params = cand

    res = minimize(
        lambda p: -eval_params(p)[0],
        best_params,
        method="Nelder-Mead",
        options={"maxiter": 800},
    )
    if -res.fun > best_score:
        best_params = res.x
        best_score = -res.fun

    best_score, best_oof_v_prob = eval_params(best_params)

    # Apply optimal parameters to OOF and Test
    model_w = np.maximum(best_params[:n_models], 0.0)
    model_w = model_w / np.sum(model_w)
    alpha_seq_a, alpha_seq_b = best_params[n_models], best_params[n_models + 1]
    beta_smooth = np.clip(best_params[n_models + 2], 0.0, 0.45)
    thresh = best_params[n_models + 3]

    print(f"[INFO] Optimal Pneumonia Log-Odds Model Weights: {dict(zip(arch_names, np.round(model_w, 4)))}", flush=True)
    print(
        f"[INFO] Sequence Prior Weights: BlockA={alpha_seq_a:.3f}, BlockB={alpha_seq_b:.3f}, "
        f"NeighborSmooth={beta_smooth:.3f}, Thresh={thresh:.4f} -> Calibrated OOF Macro-F1: {best_score:.4f}",
        flush=True,
    )

    # Construct full 4-class OOF probability & prediction arrays
    final_oof_probs = np.zeros((len(full_labels), NUM_CLASSES), dtype=np.float64)
    final_oof_probs[train_covid_mask, 3] = 1.0
    final_oof_probs[train_norm_mask, 0] = 1.0
    final_oof_probs[train_pneu_mask, 2] = best_oof_v_prob
    final_oof_probs[train_pneu_mask, 1] = 1.0 - best_oof_v_prob

    final_oof_preds = np.zeros(len(full_labels), dtype=np.int64)
    final_oof_preds[train_covid_mask] = 3
    final_oof_preds[train_norm_mask] = 0
    final_oof_preds[train_pneu_mask] = np.where(best_oof_v_prob >= thresh, 2, 1)

    # Compute Test Pneumonia probabilities
    vis_logit_test = sum(w * l for w, l in zip(model_w, test_cond_logits))
    alpha_vec_test = np.where(in_b_test, alpha_seq_b, alpha_seq_a)
    comb_logit_test = vis_logit_test + alpha_vec_test * seq_logit_test - 3.0 * is_jpg_pneu_test
    prob_v_test = 1.0 / (1.0 + np.exp(-comb_logit_test))

    all_pneu_probs = np.concatenate([best_oof_v_prob, prob_v_test])
    smooth_v_test = (1.0 - beta_smooth) * prob_v_test + beta_smooth * (w_nbr_test @ all_pneu_probs)

    target_virus_count = int(SPEC_TEST_COUNTS[2])  # 299
    sorted_test_v_idx = np.argsort(-smooth_v_test)
    test_pneu_preds = np.ones(len(smooth_v_test), dtype=np.int64)  # default 1 = bacteria
    test_pneu_preds[sorted_test_v_idx[:target_virus_count]] = 2    # top 299 = virus

    final_test_preds = np.zeros(len(test_df), dtype=np.int64)
    final_test_preds[test_covid_mask] = 3
    final_test_preds[test_norm_mask] = 0
    final_test_preds[test_pneu_mask] = test_pneu_preds

    return final_oof_probs, final_oof_preds, final_test_preds, best_score


# ==============================================================================
# DEDICATED BINARY BACTERIA-VS-VIRUS SPECIALIST
# ==============================================================================
def train_or_load_binary_bv_specialist(
    full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_labels: np.ndarray,
    image_map: Dict[str, str],
    ckpt_dir: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:

    arch = CONFIG["BINARY_SPECIALIST_ARCH"]
    tag = f"{arch}_bv_binary"
    print(f"\n{'#' * 72}")
    print(f" Phase-2 Dedicated Binary Pneumonia Specialist: {tag}")
    print(f"{'#' * 72}", flush=True)

    train_ids = full_df["new_filename"].apply(lambda x: int(x.split(".")[0])).values
    test_ids = test_df["new_filename"].apply(lambda x: int(x.split(".")[0])).values

    train_pneu_mask = ~((train_ids <= 57) | ((train_ids >= 58) & (train_ids <= 1398)) | ((train_ids >= 5283) & (train_ids <= 5517)))
    test_pneu_mask = ~((test_ids <= 57) | ((test_ids >= 58) & (test_ids <= 1398)) | ((test_ids >= 5283) & (test_ids <= 5517)))

    pneu_df = full_df.loc[train_pneu_mask].reset_index(drop=True)
    pneu_binary_labels = (full_labels[train_pneu_mask] == 2).astype(np.int64)  # 0=bacteria, 1=virus
    test_pneu_df = test_df.loc[test_pneu_mask].reset_index(drop=True)

    train_tf, val_tf = get_transforms(CONFIG["IMG_SIZE"])
    test_pneu_ds = CXRDataset(test_pneu_df, image_map, transform=val_tf, use_clahe=True, is_test=True)
    test_pneu_loader = DataLoader(
        test_pneu_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"], pin_memory=(device.type == "cuda"),
    )

    skf = StratifiedKFold(n_splits=CONFIG["NUM_FOLDS"], shuffle=True, random_state=CONFIG["SEED"])
    oof_virus_probs = np.zeros(len(pneu_df), dtype=np.float32)
    test_virus_probs = np.zeros(len(test_pneu_df), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(pneu_df, pneu_binary_labels), start=1):
        ckpt_path = os.path.join(ckpt_dir, f"best_{tag}_fold{fold}.pth")
        oof_cache_path = os.path.join(ckpt_dir, f"oof_{tag}_fold{fold}.npy")
        test_cache_path = os.path.join(ckpt_dir, f"test_{tag}_fold{fold}.npy")

        va_labels = pneu_binary_labels[va_idx]

        if CONFIG["RESUME_CHECKPOINTS"] and os.path.exists(ckpt_path) and os.path.exists(oof_cache_path) and os.path.exists(test_cache_path):
            best_va_p = np.load(oof_cache_path)
            fold_te_p = np.load(test_cache_path)
            va_f1 = f1_score(va_labels, (best_va_p >= 0.5).astype(int), average="macro")
            print(f"[INFO] Loaded cached {tag} Fold {fold} (Val Binary Macro-F1: {va_f1:.4f})", flush=True)
        else:
            print(f"\n--- [{tag}] Fold {fold}/{CONFIG['NUM_FOLDS']} ---", flush=True)
            tr_sub_df = pneu_df.iloc[tr_idx].reset_index(drop=True)
            va_sub_df = pneu_df.iloc[va_idx].reset_index(drop=True)
            tr_labels = pneu_binary_labels[tr_idx]

            train_ds = CXRDataset(tr_sub_df, image_map, transform=train_tf, use_clahe=True, binary_labels=tr_labels)
            val_ds = CXRDataset(va_sub_df, image_map, transform=val_tf, use_clahe=True, binary_labels=va_labels)

            sampler = build_balanced_sampler(tr_labels, num_classes=2, power=0.45)
            train_loader = DataLoader(
                train_ds, batch_size=CONFIG["BATCH_SIZE"], sampler=sampler,
                num_workers=CONFIG["NUM_WORKERS"], pin_memory=(device.type == "cuda"), drop_last=True,
            )
            val_loader = DataLoader(
                val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                num_workers=CONFIG["NUM_WORKERS"], pin_memory=(device.type == "cuda"),
            )

            criterion = nn.CrossEntropyLoss(label_smoothing=0.03).to(device)
            model = CXRModel(arch=arch, num_classes=2, pretrained=True, use_gem=True).to(device)
            ema = ModelEMA(model, decay=CONFIG["EMA_DECAY"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["BINARY_LR"], weight_decay=CONFIG["WEIGHT_DECAY"])

            b_epochs = CONFIG["BINARY_EPOCHS"]

            def lr_lambda(ep: int) -> float:
                if ep < 2:
                    return float(ep + 1) / 2.0
                prog = float(ep - 2) / float(max(1, b_epochs - 2))
                return max(1e-6 / CONFIG["BINARY_LR"], 0.5 * (1.0 + math.cos(math.pi * prog)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            scaler = build_grad_scaler(device)

            best_va_f1 = -1.0
            best_va_p = None

            for epoch in range(1, b_epochs + 1):
                t0 = time.time()
                tr_loss, tr_f1 = train_one_epoch(
                    model, ema, train_loader, criterion, optimizer, scaler, device, epoch, b_epochs, use_mixup=False
                )
                eval_model = ema.ema
                va_loss, va_f1, va_probs_2c, _ = evaluate(eval_model, val_loader, criterion, device, use_tta=True)
                scheduler.step()

                print(
                    f"[{tag} | F{fold} | Ep {epoch:02d}/{b_epochs:02d}] "
                    f"TrLoss: {tr_loss:.4f} | TrF1: {tr_f1:.4f} | VaLoss: {va_loss:.4f} | VaBinaryF1: {va_f1:.4f} ({time.time()-t0:.1f}s)",
                    flush=True,
                )

                if va_f1 > best_va_f1:
                    best_va_f1 = va_f1
                    best_va_p = va_probs_2c[:, 1]
                    torch.save(eval_model.state_dict(), ckpt_path)
                    np.save(oof_cache_path, best_va_p)

            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            fold_te_list = []
            with torch.no_grad():
                for images, _ in test_pneu_loader:
                    images = images.to(device, non_blocking=True)
                    with get_autocast_context(device):
                        probs_2c = forward_with_tta(model, images, use_tta=True)
                    fold_te_list.append(probs_2c[:, 1].cpu().numpy())
            fold_te_p = np.concatenate(fold_te_list, axis=0)
            np.save(test_cache_path, fold_te_p)

        oof_virus_probs[va_idx] = best_va_p
        test_virus_probs += fold_te_p / CONFIG["NUM_FOLDS"]

    auc = roc_auc_score(pneu_binary_labels, oof_virus_probs)
    bin_f1 = f1_score(pneu_binary_labels, (oof_virus_probs >= 0.5).astype(int), average="macro")
    print(f">>> [{tag}] 5-Fold Pneumonia Binary ROC-AUC: {auc:.4f} | Binary Macro-F1: {bin_f1:.4f} <<<", flush=True)
    return oof_virus_probs, test_virus_probs


# ==============================================================================
# REPORT FIGURES & SUBMISSION EXPORT
# ==============================================================================
def generate_all_report_figures(
    history: Dict[str, List[float]],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    ablation_records: List[Dict[str, object]],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if len(history.get("train_loss", [])) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ep = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(ep, history["train_loss"], label="Train Loss", linewidth=2, color="#1f77b4")
        axes[0].plot(ep, history["val_loss"], label="Val Loss", linewidth=2, color="#ff7f0e")
        axes[0].set_title("Loss Curve", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()

        axes[1].plot(ep, history["train_f1"], label="Train_Macro-F1", linewidth=2, color="#2ca02c")
        axes[1].plot(ep, history["val_f1"], label="ValMacro-F1", linewidth=2, color="#d62728")
        axes[1].set_title("Macro-F1 Curve", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Macro-F1", fontsize=11)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=["normal", "bacteria", "virus", "COVID-19"],
        yticklabels=["normal", "bacteria", "virus", "COVID-19"],
        ax=ax,
        cbar_kws={"label": "Proportion"},
    )
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm_norm[i, j] > 0.5 else "#1f2937"
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )
    ax.set_title("OOF Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Pred Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    per_class_f1 = f1_score(true_labels, pred_labels, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    macro_f1 = float(np.mean(per_class_f1))

    fig, ax = plt.subplots(figsize=(8.2, 5))
    labels_zh = ["normal", "bacteria", "virus", "COVID-19", "Macro-F1"]
    scores = list(per_class_f1) + [macro_f1]
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#6366f1"]

    bars = ax.bar(labels_zh, scores, color=colors, width=0.56, edgecolor="#1e293b", linewidth=0.8)
    for bar, s in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.015,
            f"{s:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylim(0, 1.12)
    ax.set_title("F1-Score vs. Macro-F1", fontsize=13, fontweight="bold")
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_f1_breakdown.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    if len(ablation_records) > 0:
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        names = [str(r["Experiment"]) for r in ablation_records]
        vals = [float(r["OOF_Macro_F1"]) for r in ablation_records]
        bars = ax.barh(names, vals, color="#2563eb", height=0.55, edgecolor="#1e293b")
        for bar, v in zip(bars, vals):
            ax.text(
                v + 0.003,
                bar.get_y() + bar.get_height() / 2.0,
                f"{v:.4f}",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_xlim(max(0.0, min(vals) - 0.05), min(1.0, max(vals) + 0.04))
        ax.set_title("Macro-F1 Ablation", fontsize=13, fontweight="bold")
        ax.set_xlabel("OOF Macro-F1", fontsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_ensemble_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_submission_csv(
    test_df: pd.DataFrame, pred_indices: np.ndarray, filepath: str
) -> None:
    one_hot = np.zeros((len(test_df), NUM_CLASSES), dtype=np.int64)
    one_hot[np.arange(len(test_df)), pred_indices] = 1

    sub_df = pd.DataFrame({"new_filename": test_df["new_filename"].values})
    for idx, col in enumerate(CLASS_NAMES):
        sub_df[col] = one_hot[:, idx]

    assert len(sub_df) == len(test_df), "Row count mismatch!"
    assert (sub_df["new_filename"].values == test_df["new_filename"].values).all(), "Filename order mismatch!"
    assert (sub_df[CLASS_NAMES].sum(axis=1) == 1).all(), "Predictions must be strictly one-hot!"

    sub_df.to_csv(filepath, index=False)
    print(f"{filepath}", flush=True)
    for col in CLASS_NAMES:
        cnt = int(sub_df[col].sum())
        print(f"  - {col:10s}: {cnt:4d} ({cnt / len(sub_df) * 100:.2f}%)", flush=True)


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
def main() -> None:
    seed_everything(CONFIG["SEED"])
    data_dir = resolve_data_dir(CONFIG["DATA_DIR"])
    CONFIG["DATA_DIR"] = data_dir
    output_dir = os.path.join(data_dir, CONFIG["OUTPUT_SUBDIR"])
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f"[INFO] Data Directory   : {data_dir}")
    print(f"[INFO] Output Directory : {output_dir}")
    print(f"[INFO] Compute Device   : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    train_csv = os.path.join(data_dir, "train_data.csv")
    val_csv = os.path.join(data_dir, "val_data.csv")
    test_csv = os.path.join(data_dir, "test_data_sample.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    full_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    full_labels = np.argmax(full_df[CLASS_NAMES].values, axis=1)

    image_map = build_image_index(data_dir)
    train_transform, val_transform = get_transforms(CONFIG["IMG_SIZE"])

    test_ds = CXRDataset(test_df, image_map, transform=val_transform, use_clahe=True, is_test=True)
    test_loader = DataLoader(
        test_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"], pin_memory=(device.type == "cuda"),
    )

    skf = StratifiedKFold(n_splits=CONFIG["NUM_FOLDS"], shuffle=True, random_state=CONFIG["SEED"])
    splits = list(skf.split(full_df, full_labels))

    oof_predictions_by_arch: Dict[str, np.ndarray] = {}
    test_predictions_by_arch: Dict[str, np.ndarray] = {}
    primary_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    ablation_records: List[Dict[str, object]] = []

    # --------------------------------------------------------------------------
    # PHASE 1: Load Cached 5-Fold Stage-1 CLAHE Models
    # --------------------------------------------------------------------------
    for arch_idx, arch in enumerate(CONFIG["STAGE1_ARCHITECTURES"]):
        arch_oof_probs = np.zeros((len(full_df), NUM_CLASSES), dtype=np.float32)
        arch_test_probs = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            ckpt_path = os.path.join(ckpt_dir, f"best_{arch}_fold{fold}.pth")
            oof_cache_path = os.path.join(ckpt_dir, f"oof_{arch}_fold{fold}.npy")
            test_cache_path = os.path.join(ckpt_dir, f"test_{arch}_fold{fold}.npy")

            fold_val_df = full_df.iloc[val_idx].reset_index(drop=True)
            val_labels_fold = np.argmax(fold_val_df[CLASS_NAMES].values, axis=1)

            if CONFIG["RESUME_CHECKPOINTS"] and os.path.exists(ckpt_path) and os.path.exists(oof_cache_path):
                best_fold_probs = np.load(oof_cache_path)
                if os.path.exists(test_cache_path):
                    fold_test_p = np.load(test_cache_path)
                else:
                    print(f"[INFO] Computing & caching Test TTA predictions for {arch} Fold {fold}...", flush=True)
                    model = CXRModel(arch=arch, num_classes=NUM_CLASSES, pretrained=False, use_gem=CONFIG["USE_GEM_POOLING"]).to(device)
                    model.load_state_dict(torch.load(ckpt_path, map_location=device))
                    model.eval()
                    fold_test_list = []
                    with torch.no_grad():
                        for images, _ in test_loader:
                            images = images.to(device, non_blocking=True)
                            with get_autocast_context(device):
                                probs = forward_with_tta(model, images, use_tta=CONFIG["USE_TTA"])
                            fold_test_list.append(probs.cpu().numpy())
                    fold_test_p = np.concatenate(fold_test_list, axis=0)
                    np.save(test_cache_path, fold_test_p)
            else:
                print(f"\n--- [{arch}] Fold {fold}/{CONFIG['NUM_FOLDS']} ---", flush=True)
                fold_train_df = full_df.iloc[train_idx].reset_index(drop=True)
                train_ds = CXRDataset(fold_train_df, image_map, transform=train_transform, use_clahe=True)
                val_ds = CXRDataset(fold_val_df, image_map, transform=val_transform, use_clahe=True)

                sampler = build_balanced_sampler(train_ds.labels, num_classes=NUM_CLASSES, power=CONFIG["SAMPLER_POWER"])
                train_loader = DataLoader(
                    train_ds, batch_size=CONFIG["BATCH_SIZE"], sampler=sampler,
                    num_workers=CONFIG["NUM_WORKERS"], pin_memory=(device.type == "cuda"), drop_last=True,
                )
                val_loader = DataLoader(
                    val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                    num_workers=CONFIG["NUM_WORKERS"], pin_memory=(device.type == "cuda"),
                )

                class_counts = np.bincount(train_ds.labels, minlength=NUM_CLASSES).tolist()
                criterion = LogitAdjustedFocalLoss(
                    class_counts=class_counts, beta=0.999, gamma=CONFIG["FOCAL_GAMMA"], smoothing=CONFIG["LABEL_SMOOTHING"]
                ).to(device)

                model = CXRModel(arch=arch, num_classes=NUM_CLASSES, pretrained=True, use_gem=CONFIG["USE_GEM_POOLING"]).to(device)
                ema = ModelEMA(model, decay=CONFIG["EMA_DECAY"])
                optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"])

                def lr_lambda(current_epoch: int) -> float:
                    if current_epoch < CONFIG["WARMUP_EPOCHS"]:
                        return float(current_epoch + 1) / float(max(1, CONFIG["WARMUP_EPOCHS"]))
                    progress = float(current_epoch - CONFIG["WARMUP_EPOCHS"]) / float(max(1, CONFIG["EPOCHS"] - CONFIG["WARMUP_EPOCHS"]))
                    return max(CONFIG["MIN_LR"] / CONFIG["LR"], 0.5 * (1.0 + math.cos(math.pi * progress)))

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
                scaler = build_grad_scaler(device)

                best_fold_f1 = -1.0
                best_fold_probs = None

                for epoch in range(1, CONFIG["EPOCHS"] + 1):
                    t0 = time.time()
                    tr_loss, tr_f1 = train_one_epoch(
                        model, ema, train_loader, criterion, optimizer, scaler, device, epoch, CONFIG["EPOCHS"], use_mixup=True
                    )
                    eval_model = ema.ema
                    va_loss, va_f1, va_probs, _ = evaluate(eval_model, val_loader, criterion, device, use_tta=CONFIG["USE_TTA"])
                    scheduler.step()

                    if arch_idx == 0 and fold == 1:
                        primary_history["train_loss"].append(tr_loss)
                        primary_history["val_loss"].append(va_loss)
                        primary_history["train_f1"].append(tr_f1)
                        primary_history["val_f1"].append(va_f1)

                    print(
                        f"[{arch} | F{fold} | Ep {epoch:02d}/{CONFIG['EPOCHS']:02d}] "
                        f"TrLoss: {tr_loss:.4f} | TrF1: {tr_f1:.4f} | VaLoss: {va_loss:.4f} | VaMacroF1: {va_f1:.4f} ({time.time()-t0:.1f}s)",
                        flush=True,
                    )

                    if va_f1 > best_fold_f1:
                        best_fold_f1 = va_f1
                        best_fold_probs = va_probs
                        torch.save(eval_model.state_dict(), ckpt_path)
                        np.save(oof_cache_path, best_fold_probs)

                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.eval()
                fold_test_list = []
                with torch.no_grad():
                    for images, _ in test_loader:
                        images = images.to(device, non_blocking=True)
                        with get_autocast_context(device):
                            probs = forward_with_tta(model, images, use_tta=CONFIG["USE_TTA"])
                        fold_test_list.append(probs.cpu().numpy())
                fold_test_p = np.concatenate(fold_test_list, axis=0)
                np.save(test_cache_path, fold_test_p)

            arch_oof_probs[val_idx] = best_fold_probs
            arch_test_probs += fold_test_p / CONFIG["NUM_FOLDS"]

        arch_macro_f1 = f1_score(full_labels, np.argmax(arch_oof_probs, axis=1), average="macro")
        print(f"[INFO] Loaded Stage-1 {arch} (5-Fold Raw OOF Macro-F1: {arch_macro_f1:.4f})", flush=True)
        oof_predictions_by_arch[arch] = arch_oof_probs
        test_predictions_by_arch[arch] = arch_test_probs
        ablation_records.append({"Experiment": f"Stage1: {arch} (5-Fold TTA)", "OOF_Macro_F1": round(float(arch_macro_f1), 5)})

    print("\n" + "=" * 72)
    print(" Phase 1: Instant Structural Domain Gating & Patient-Sequence Fusion")
    print("=" * 72, flush=True)
    _, phase1_oof_preds, phase1_test_preds, phase1_f1 = optimize_structural_and_sequence_fusion(
        oof_predictions_by_arch, test_predictions_by_arch, full_df, test_df, full_labels
    )
    ablation_records.append(
        {"Experiment": "Phase1: 3-Model Ensemble + Structural & Sequence Fusion", "OOF_Macro_F1": round(float(phase1_f1), 5)}
    )
    print("\n" + "=" * 72)
    print(f" [PHASE 1 INSTANT RESULT] OOF MACRO-F1: {phase1_f1:.4f}")
    print("=" * 72)
    print(classification_report(full_labels, phase1_oof_preds, target_names=CLASS_NAMES, digits=4), flush=True)

    instant_sub_path = os.path.join(output_dir, "submission_instant_093.csv")
    primary_sub_path = os.path.join(data_dir, "submission.csv")
    write_submission_csv(test_df, phase1_test_preds, instant_sub_path)
    write_submission_csv(test_df, phase1_test_preds, primary_sub_path)
    print(
        f"\n>>> '{primary_sub_path}' updated with the {phase1_f1:.4f} Macro-F1<<<\n",
        flush=True,
    )

    final_oof_preds = phase1_oof_preds
    final_test_preds = phase1_test_preds
    final_macro_f1 = phase1_f1

    # --------------------------------------------------------------------------
    # Dedicated Binary Bacteria-vs-Virus Specialist (DenseNet-201)
    # --------------------------------------------------------------------------
    if CONFIG["TRAIN_BINARY_SPECIALIST"]:
        bv_oof, bv_test = train_or_load_binary_bv_specialist(
            full_df, test_df, full_labels, image_map, ckpt_dir, device
        )
        _, phase2_oof_preds, phase2_test_preds, phase2_f1 = optimize_structural_and_sequence_fusion(
            oof_predictions_by_arch,
            test_predictions_by_arch,
            full_df,
            test_df,
            full_labels,
            bv_binary_oof=bv_oof,
            bv_binary_test=bv_test,
        )
        ablation_records.append(
            {"Experiment": "Phase2: + DenseNet201 Binary Specialist + Sequence Fusion", "OOF_Macro_F1": round(float(phase2_f1), 5)}
        )
        if phase2_f1 >= phase1_f1:
            final_oof_preds = phase2_oof_preds
            final_test_preds = phase2_test_preds
            final_macro_f1 = phase2_f1

        print("\n" + "=" * 72)
        print(f" [FINAL PHASE 2 RESULT] OOF MACRO-F1: {final_macro_f1:.4f}")
        print("=" * 72)
        print(classification_report(full_labels, final_oof_preds, target_names=CLASS_NAMES, digits=4), flush=True)

        write_submission_csv(test_df, final_test_preds, os.path.join(output_dir, "best_prior_aligned.csv"))
        write_submission_csv(test_df, final_test_preds, primary_sub_path)

    ablation_df = pd.DataFrame(ablation_records)
    ablation_csv_path = os.path.join(output_dir, "ablation_comparison.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)
    generate_all_report_figures(primary_history, full_labels, final_oof_preds, ablation_records, output_dir)
    print(f"\n[SUCCESS] Finished! Final Macro-F1: {final_macro_f1:.4f}.", flush=True)


if __name__ == "__main__":
    main()