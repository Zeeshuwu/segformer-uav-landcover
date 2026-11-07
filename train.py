import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import logging
import sys
import os
import random
import time
import argparse
import warnings
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation
from sklearn.model_selection import train_test_split, KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config

# Windows consoles / redirected pipes often default to cp1252, which crashes
# on the emoji used in the print() calls throughout this pipeline.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from torch.amp import autocast, GradScaler
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_DEVICE = None


# ============================================================
# LOGGING SETUP
# ============================================================

class CleanFormatter(logging.Formatter):
    """Custom formatter untuk output yang lebih bersih"""
    def format(self, record):
        if record.levelno == logging.INFO:
            return record.getMessage()
        return f"[{record.levelname}] {record.getMessage()}"


def setup_logging(config=Config):
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # File handler dengan format lengkap
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Console handler dengan format bersih
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CleanFormatter())
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("albumentations").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def set_seed(seed: int):
    # Note: PYTHONHASHSEED must be set before the interpreter starts to have any
    # effect, so it isn't set here -- doing it mid-process would be a no-op.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class LandCover7BandDataset(Dataset):
    def __init__(self, patch_paths: list, mask_paths: list, transform=None, band_idx=None):
        self.patch_paths = patch_paths
        self.mask_paths = mask_paths
        self.transform = transform
        # Channels to keep from the stored 7-band patches (None = all), so every
        # input configuration trains on exactly the same patches.
        self.band_idx = band_idx

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        patch = np.load(self.patch_paths[idx]).astype(np.float32)
        if self.band_idx is not None:
            patch = np.ascontiguousarray(patch[..., self.band_idx])
        mask = np.load(self.mask_paths[idx])

        if self.transform:
            transformed = self.transform(image=patch, mask=mask)
            patch = transformed['image']
            mask = transformed['mask'].long()
        else:
            patch = torch.from_numpy(patch).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask.astype(np.int64))

        return patch, mask


# ============================================================
# AUGMENTATION
# ============================================================

def get_training_augmentation(config=Config):
    transforms = []
    if config.USE_AUGMENTATION:
        if config.AUG_HORIZONTAL_FLIP > 0:
            transforms.append(A.HorizontalFlip(p=config.AUG_HORIZONTAL_FLIP))
        if config.AUG_VERTICAL_FLIP > 0:
            transforms.append(A.VerticalFlip(p=config.AUG_VERTICAL_FLIP))
        if config.AUG_ROTATION > 0:
            # Corners exposed by the rotation are ignored in the loss, not labelled background.
            transforms.append(A.Rotate(limit=config.AUG_ROTATION_LIMIT, p=config.AUG_ROTATION,
                                       border_mode=cv2.BORDER_CONSTANT, fill=0,
                                       fill_mask=config.IGNORE_INDEX))
        if config.AUG_BRIGHTNESS > 0 or config.AUG_CONTRAST > 0:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=config.AUG_BRIGHTNESS,
                contrast_limit=config.AUG_CONTRAST,
                p=0.5,
            ))
    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def get_validation_augmentation():
    return A.Compose([ToTensorV2()])


# ============================================================
# MODEL
# ============================================================

class SegFormer7Band(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.cfg = config
        
        # Load pretrained model (suppress warnings)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            config.MODEL_NAME,
            num_labels=config.NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
        
        # Find first Conv2d layer
        first_conv = None
        first_conv_name = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                first_conv = module
                first_conv_name = name
                break
        
        if first_conv is None:
            raise RuntimeError("Could not find first Conv2d layer with 3 input channels")
        
        # Create new conv layer with 7 channels
        new_conv = nn.Conv2d(
            in_channels=config.NUM_CHANNELS,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        
        # Initialize weights
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = first_conv.weight.clone()
            avg_weight = first_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, config.NUM_CHANNELS):
                new_conv.weight[:, i:i+1, :, :] = avg_weight.clone()
            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)
        
        # Replace the layer
        parts = first_conv_name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        setattr(parent, parts[-1], new_conv)
        
        # Freeze encoder if specified. The band-adaptation conv (new_conv) is excluded:
        # its extra input channels were just initialized (copied/averaged from the
        # pretrained 3-channel weights, never actually pretrained themselves), so
        # freezing it along with the rest of the encoder would permanently lock in
        # those untrained weights and the model could never learn to use bands 4-7.
        new_conv_param_ids = {id(p) for p in new_conv.parameters()}
        self.frozen_count = 0
        if config.FREEZE_ENCODER:
            for name, param in self.model.named_parameters():
                if id(param) in new_conv_param_ids:
                    continue
                if 'decode_head' not in name and 'classifier' not in name:
                    param.requires_grad = False
                    self.frozen_count += 1

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = nn.functional.interpolate(
            outputs.logits, 
            size=x.shape[-2:],
            mode='bilinear', 
            align_corners=False,
        )
        return logits


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights)

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none',
            ignore_index=self.ignore_index, weight=self.class_weights,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255, class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights)

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(inputs)
        valid_mask = targets != self.ignore_index
        targets_valid = targets.clone()
        targets_valid[~valid_mask] = 0
        targets_one_hot.scatter_(1, targets_valid.unsqueeze(1), 1)
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(inputs)
        inputs = inputs * valid_mask_expanded.float()
        targets_one_hot = targets_one_hot * valid_mask_expanded.float()
        intersection = (inputs * targets_one_hot).sum(dim=(0, 2, 3))
        union = inputs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        if self.class_weights is not None:
            return 1.0 - (dice_per_class * self.class_weights).sum() / self.class_weights.sum()
        return 1.0 - dice_per_class.mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5,
                 focal_alpha=0.25, focal_gamma=2.0,
                 dice_smooth=1.0, ignore_index=255, class_weights=None):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                                ignore_index=ignore_index, class_weights=class_weights)
        self.dice = DiceLoss(smooth=dice_smooth,
                              ignore_index=ignore_index, class_weights=class_weights)

    def forward(self, inputs, targets):
        return self.focal_weight * self.focal(inputs, targets) + \
               self.dice_weight * self.dice(inputs, targets)


def get_loss_function(config=Config):
    class_weights = None
    if config.CLASS_WEIGHTS is not None:
        class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32)

    if config.LOSS_FUNCTION == "crossentropy":
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=config.IGNORE_INDEX)
    elif config.LOSS_FUNCTION == "focal":
        return FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA,
                          ignore_index=config.IGNORE_INDEX, class_weights=class_weights)
    elif config.LOSS_FUNCTION == "dice":
        return DiceLoss(smooth=config.DICE_SMOOTH,
                         ignore_index=config.IGNORE_INDEX, class_weights=class_weights)
    elif config.LOSS_FUNCTION == "focal_dice":
        return FocalDiceLoss(
            focal_weight=config.FOCAL_WEIGHT,
            dice_weight=config.DICE_WEIGHT,
            focal_alpha=config.FOCAL_ALPHA,
            focal_gamma=config.FOCAL_GAMMA,
            dice_smooth=config.DICE_SMOOTH,
            ignore_index=config.IGNORE_INDEX,
            class_weights=class_weights,
        )
    else:
        raise ValueError(f"Unknown loss function: {config.LOSS_FUNCTION}")


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(predictions, targets, num_classes, ignore_index=255):
    """Calculate basic metrics per batch"""
    preds = predictions.cpu().numpy()
    tgts = targets.cpu().numpy()
    valid = tgts != ignore_index

    class_iou = []
    class_acc = []
    present_iou = []

    for cls in range(num_classes):
        pred_c = (preds == cls) & valid
        target_c = (tgts == cls) & valid
        inter = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()

        class_iou.append(inter / union if union > 0 else 0.0)
        if union > 0:
            present_iou.append(inter / union)
        class_total = target_c.sum()
        class_acc.append(inter / class_total if class_total > 0 else 0.0)

    # Classes absent from both prediction and target are skipped, not counted as 0.
    mean_iou = float(np.mean(present_iou)) if present_iou else 0.0
    correct = ((preds == tgts) & valid).sum()
    total = valid.sum()
    overall_acc = float(correct / total) if total > 0 else 0.0

    return mean_iou, overall_acc, class_iou, class_acc


def calculate_confusion_matrix(predictions, targets, num_classes, ignore_index=255):
    """Confusion matrix (rows = reference, cols = prediction), computed on the device."""
    preds = predictions.flatten().long()
    tgts = targets.flatten().long()
    valid = (tgts != ignore_index) & (tgts >= 0) & (tgts < num_classes) & (preds >= 0) & (preds < num_classes)
    cm = torch.bincount(tgts[valid] * num_classes + preds[valid], minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).cpu().numpy().astype(np.int64)


def summarize_cm(cm, eval_classes):
    """Per-class and macro metrics over `eval_classes` from a full confusion matrix.
    Predictions of other indices (e.g. background) count as errors. The macro mean
    covers the eval classes that have reference pixels."""
    m = compute_metrics_from_cm(cm)
    support = cm.sum(axis=1)
    idx = [c for c in eval_classes if support[c] > 0]
    valid_total = cm[eval_classes, :].sum()
    return {
        'overall_accuracy': float(np.trace(cm[np.ix_(eval_classes, eval_classes)]) / valid_total) if valid_total else 0.0,
        'mean_iou': float(np.mean(m['iou'][idx])) if idx else 0.0,
        'mean_f1': float(np.mean(m['f1'][idx])) if idx else 0.0,
        'mean_precision': float(np.mean(m['precision'][idx])) if idx else 0.0,
        'mean_recall': float(np.mean(m['recall'][idx])) if idx else 0.0,
        'classes_in_mean': idx,
        'per_class': {int(c): {'iou': float(m['iou'][c]), 'precision': float(m['precision'][c]),
                               'recall': float(m['recall'][c]), 'f1': float(m['f1'][c]),
                               'support_px': int(support[c])} for c in eval_classes},
    }


def compute_metrics_from_cm(cm):
    """Compute TP, FP, FN, TN, Precision, Recall, F1 per class from confusion matrix"""
    num_classes = cm.shape[0]
    metrics = {
        'TP': np.zeros(num_classes),
        'FP': np.zeros(num_classes),
        'FN': np.zeros(num_classes),
        'TN': np.zeros(num_classes),
        'precision': np.zeros(num_classes),
        'recall': np.zeros(num_classes),
        'f1': np.zeros(num_classes),
        'iou': np.zeros(num_classes),
    }
    
    total = cm.sum()
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        
        metrics['TP'][i] = tp
        metrics['FP'][i] = fp
        metrics['FN'][i] = fn
        metrics['TN'][i] = tn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics['precision'][i] = precision
        metrics['recall'][i] = recall
        metrics['f1'][i] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics['iou'][i] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return metrics


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """Stop training when validation metric stops improving"""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        """
        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like IoU/Acc, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        self.best_epoch = 0
        
    def __call__(self, current_value, epoch):
        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = current_value > (self.best_value + self.min_delta)
        else:  # mode == 'min'
            improved = current_value < (self.best_value - self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def status(self):
        if self.best_value is None:
            return f"0/{self.patience} (no updates yet)"
        return f"{self.counter}/{self.patience} (best={self.best_value:.4f} @ epoch {self.best_epoch + 1})"
    
    def reset(self):
        """Reset for new fold in K-Fold CV"""
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        self.best_epoch = 0


# ============================================================
# DATA LOADING
# ============================================================

def load_patch_mask_paths(config=Config):
    patches_dir = config.PATCHES_DIR
    masks_dir = config.MASKS_DIR

    if not patches_dir.exists():
        raise FileNotFoundError(f"PATCHES_DIR not found: {patches_dir}")

    all_patch_files = sorted(patches_dir.rglob("*.npy"))
    if not all_patch_files:
        raise FileNotFoundError(f"No .npy files in {patches_dir}")

    patch_paths, mask_paths = [], []
    for patch_f in all_patch_files:
        relative = patch_f.relative_to(patches_dir)
        mask_f = masks_dir / relative
        if mask_f.exists():
            patch_paths.append(patch_f)
            mask_paths.append(mask_f)

    return patch_paths, mask_paths


def load_split_manifest(config=Config):
    with open(config.SPLIT_MANIFEST) as f:
        manifest = json.load(f)
    root = config.PROCESSED_DATA_DIR
    paths = {
        split: ([root / r["patch"] for r in recs], [root / r["mask"] for r in recs])
        for split, recs in manifest["patches"].items()
    }
    return manifest, paths


def build_dataloaders(config=Config, band_idx=None, max_train=None):
    """Train and val patches come from the spatial-block split manifest written by
    build_training_dataset.py, never from a random split of overlapping patches."""
    _, paths = load_split_manifest(config)
    train_p, train_m = paths["train"]
    val_p, val_m = paths["val"]
    if max_train:
        train_p, train_m = train_p[:max_train], train_m[:max_train]
        val_p, val_m = val_p[:max(1, max_train // 2)], val_m[:max(1, max_train // 2)]

    train_ds = LandCover7BandDataset(train_p, train_m, transform=get_training_augmentation(config), band_idx=band_idx)
    val_ds = LandCover7BandDataset(val_p, val_m, transform=get_validation_augmentation(), band_idx=band_idx)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=False,
    )

    return train_loader, val_loader, len(train_ds), len(val_ds)


def build_fold_dataloaders(train_patches, train_masks, val_patches, val_masks, config=Config):
    """Build dataloaders for a specific fold"""
    train_ds = LandCover7BandDataset(
        train_patches, train_masks, 
        transform=get_training_augmentation(config)
    )
    val_ds = LandCover7BandDataset(
        val_patches, val_masks, 
        transform=get_validation_augmentation()
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=False,
    )

    return train_loader, val_loader, len(train_ds), len(val_ds)


# ============================================================
# VISUALIZATION
# ============================================================

def plot_training_results(history, final_metrics, class_names, save_path):
    """Generate comprehensive training visualization"""
    
    num_classes = len(class_names)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('Training Results Summary', fontsize=20, fontweight='bold', y=0.98)
    
    # Define grid
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    # Color scheme
    train_color = '#2E86AB'
    val_color = '#E94F37'
    
    # ============ Row 1 ============
    
    # 1. Loss Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], label='Train', color=train_color, linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val', color=val_color, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(epochs))
    
    # 2. Overall IoU Curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_iou'], label='Train', color=train_color, linewidth=2)
    ax2.plot(epochs, history['val_iou'], label='Val', color=val_color, linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.set_title('Mean IoU Curve', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(epochs))
    ax2.set_ylim(0, 1)
    
    # 3. Overall Accuracy Curve
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['train_acc'], label='Train', color=train_color, linewidth=2)
    ax3.plot(epochs, history['val_acc'], label='Val', color=val_color, linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Overall Accuracy Curve', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, len(epochs))
    ax3.set_ylim(0, 1)
    
    # 4. Learning Rate Curve
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(epochs, history['lr'], color='#28A745', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1, len(epochs))
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # ============ Row 2 ============
    
    # 5. Per-Class IoU (Bar Chart)
    ax5 = fig.add_subplot(gs[1, :2])
    x_pos = np.arange(num_classes)
    bars = ax5.bar(x_pos, final_metrics['iou'], color=plt.cm.tab10(np.linspace(0, 1, num_classes)))
    ax5.set_xlabel('Class')
    ax5.set_ylabel('IoU')
    ax5.set_title('Per-Class IoU (Final Validation)', fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(class_names, rotation=45, ha='right')
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, final_metrics['iou']):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Per-Class F1 Score (Bar Chart)
    ax6 = fig.add_subplot(gs[1, 2:])
    bars = ax6.bar(x_pos, final_metrics['f1'], color=plt.cm.tab10(np.linspace(0, 1, num_classes)))
    ax6.set_xlabel('Class')
    ax6.set_ylabel('F1 Score')
    ax6.set_title('Per-Class F1 Score (Final Validation)', fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(class_names, rotation=45, ha='right')
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, final_metrics['f1']):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ============ Row 3 ============
    
    # 7. Precision & Recall per Class
    ax7 = fig.add_subplot(gs[2, :2])
    width = 0.35
    ax7.bar(x_pos - width/2, final_metrics['precision'], width, label='Precision', color='#3498DB')
    ax7.bar(x_pos + width/2, final_metrics['recall'], width, label='Recall', color='#E74C3C')
    ax7.set_xlabel('Class')
    ax7.set_ylabel('Score')
    ax7.set_title('Precision & Recall per Class', fontweight='bold')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(class_names, rotation=45, ha='right')
    ax7.set_ylim(0, 1.15)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. TP, FP, FN per Class (Stacked/Grouped)
    ax8 = fig.add_subplot(gs[2, 2:])
    width = 0.25
    ax8.bar(x_pos - width, final_metrics['TP'], width, label='TP', color='#27AE60')
    ax8.bar(x_pos, final_metrics['FP'], width, label='FP', color='#E74C3C')
    ax8.bar(x_pos + width, final_metrics['FN'], width, label='FN', color='#F39C12')
    ax8.set_xlabel('Class')
    ax8.set_ylabel('Pixel Count')
    ax8.set_title('TP / FP / FN per Class', fontweight='bold')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(class_names, rotation=45, ha='right')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # ============ Row 4 ============
    
    # 9. Confusion Matrix (Normalized)
    ax9 = fig.add_subplot(gs[3, :2])
    cm_normalized = final_metrics['confusion_matrix'].astype(float)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm_normalized = cm_normalized / row_sums
    
    im = ax9.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    ax9.set_xticks(np.arange(num_classes))
    ax9.set_yticks(np.arange(num_classes))
    ax9.set_xticklabels(class_names, rotation=45, ha='right')
    ax9.set_yticklabels(class_names)
    ax9.set_xlabel('Predicted')
    ax9.set_ylabel('True')
    ax9.set_title('Normalized Confusion Matrix', fontweight='bold')
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_normalized[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax9.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)
    
    # 10. Summary Statistics Table
    ax10 = fig.add_subplot(gs[3, 2:])
    ax10.axis('off')
    
    # Create summary data
    summary_data = [
        ['Metric', 'Value'],
        ['─' * 20, '─' * 15],
        ['Final Train Loss', f"{history['train_loss'][-1]:.4f}"],
        ['Final Val Loss', f"{history['val_loss'][-1]:.4f}"],
        ['', ''],
        ['Final Train mIoU', f"{history['train_iou'][-1]:.4f}"],
        ['Final Val mIoU', f"{history['val_iou'][-1]:.4f}"],
        ['Best Val mIoU', f"{max(history['val_iou']):.4f}"],
        ['', ''],
        ['Final Train Acc', f"{history['train_acc'][-1]:.4f}"],
        ['Final Val Acc', f"{history['val_acc'][-1]:.4f}"],
        ['', ''],
        ['Mean Precision', f"{np.mean(final_metrics['precision']):.4f}"],
        ['Mean Recall', f"{np.mean(final_metrics['recall']):.4f}"],
        ['Mean F1 Score', f"{np.mean(final_metrics['f1']):.4f}"],
        ['', ''],
        ['Total Epochs', f"{len(history['train_loss'])}"],
    ]
    
    # Draw table
    table = ax10.table(
        cellText=summary_data,
        loc='center',
        cellLoc='left',
        colWidths=[0.5, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax10.set_title('Training Summary', fontweight='bold', pad=20)
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    return save_path


def plot_kfold_summary(fold_results, save_path):
    """Generate K-Fold cross validation summary plot"""
    
    num_folds = len(fold_results['fold_val_iou'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('K-Fold Cross Validation Results', fontsize=14, fontweight='bold')
    
    x_pos = np.arange(num_folds)
    fold_labels = [f'Fold {i+1}' for i in range(num_folds)]
    
    # IoU per fold
    ax1 = axes[0]
    bars = ax1.bar(x_pos, fold_results['fold_val_iou'], color='#3498DB', edgecolor='black')
    ax1.axhline(y=np.mean(fold_results['fold_val_iou']), color='red', linestyle='--', 
                label=f"Mean: {np.mean(fold_results['fold_val_iou']):.4f}")
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Validation mIoU')
    ax1.set_title('mIoU per Fold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(fold_labels)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Highlight best fold
    best_idx = fold_results['best_fold'] - 1
    bars[best_idx].set_color('#27AE60')
    
    for bar, val in zip(bars, fold_results['fold_val_iou']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Accuracy per fold
    ax2 = axes[1]
    bars = ax2.bar(x_pos, fold_results['fold_val_acc'], color='#9B59B6', edgecolor='black')
    ax2.axhline(y=np.mean(fold_results['fold_val_acc']), color='red', linestyle='--',
                label=f"Mean: {np.mean(fold_results['fold_val_acc']):.4f}")
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Accuracy per Fold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(fold_labels)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    bars[best_idx].set_color('#27AE60')
    
    for bar, val in zip(bars, fold_results['fold_val_acc']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Loss per fold
    ax3 = axes[2]
    bars = ax3.bar(x_pos, fold_results['fold_val_loss'], color='#E74C3C', edgecolor='black')
    ax3.axhline(y=np.mean(fold_results['fold_val_loss']), color='blue', linestyle='--',
                label=f"Mean: {np.mean(fold_results['fold_val_loss']):.4f}")
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Validation Loss')
    ax3.set_title('Loss per Fold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(fold_labels)
    ax3.legend()
    
    bars[best_idx].set_color('#27AE60')
    
    for bar, val in zip(bars, fold_results['fold_val_loss']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path


# ============================================================
# TRAINER
# ============================================================

class SegFormerTrainer:
    def __init__(self, config=Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = SegFormer7Band(config).to(self.device)
        self.criterion = get_loss_function(config).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS,
            eps=config.EPS,
        )

        # Scheduler setup
        if config.LR_SCHEDULER == "polynomial":
            self.scheduler = optim.lr_scheduler.PolynomialLR(
                self.optimizer, total_iters=config.NUM_EPOCHS, power=config.LR_POWER
            )
        elif config.LR_SCHEDULER == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.NUM_EPOCHS
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )

        # Mixed precision
        if config.MIXED_PRECISION and torch.cuda.is_available():
            self.scaler = GradScaler(device=_AMP_DEVICE) if _AMP_DEVICE else GradScaler()
        else:
            self.scaler = None

        # Early stopping
        if config.EARLY_STOPPING:
            self.early_stopping = EarlyStopping(
                patience=config.EARLY_STOPPING_PATIENCE,
                min_delta=config.EARLY_STOPPING_MIN_DELTA,
                mode='max'  # Monitor IoU (higher is better)
            )
        else:
            self.early_stopping = None

        # History tracking
        self.history = {
            'train_loss': [], 'train_iou': [], 'train_acc': [],
            'val_loss': [], 'val_iou': [], 'val_acc': [],
            'lr': [], 'class_iou_history': [], 'class_acc_history': [],
        }
        self.best_val_iou = 0.0
        self.best_epoch = None
        self.best_state = None
        self.final_confusion_matrix = None
        self.epoch_times = []

    def _print_header(self):
        """Print training header"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("\n" + "="*70)
        print("  SEGFORMER TRAINING")
        print("="*70)
        print(f"  Device        : {self.device}")
        print(f"  Model         : {self.config.MODEL_NAME}")
        print(f"  Input Channels: {self.config.NUM_CHANNELS}")
        print(f"  Num Classes   : {self.config.NUM_CLASSES}")
        print(f"  Parameters    : {total_params:,} total, {trainable_params:,} trainable")
        print(f"  Loss Function : {self.config.LOSS_FUNCTION}")
        print(f"  LR Scheduler  : {self.config.LR_SCHEDULER}")
        print(f"  Mixed Precision: {'Enabled' if self.scaler else 'Disabled'}")
        print(f"  Grad Accum    : {self.config.GRADIENT_ACCUMULATION_STEPS} "
              f"(effective batch {self.config.BATCH_SIZE * self.config.GRADIENT_ACCUMULATION_STEPS})")
        if self.config.FREEZE_ENCODER:
            print(f"  Frozen Params : {self.model.frozen_count:,}")
        if self.early_stopping:
            print(f"  Early Stopping: patience={self.config.EARLY_STOPPING_PATIENCE}, min_delta={self.config.EARLY_STOPPING_MIN_DELTA}")
        print("="*70 + "\n")

    def _print_epoch_header(self):
        """Print epoch table header"""
        print("-"*100)
        print(f"{'Epoch':^7}|{'Train Loss':^12}|{'Train IoU':^11}|{'Train Acc':^11}|"
              f"{'Val Loss':^11}|{'Val IoU':^10}|{'Val Acc':^10}|{'LR':^12}|{'ES':^6}")
        print("-"*100)

    def _print_epoch_result(self, epoch, tr_loss, tr_iou, tr_acc, val_loss, val_iou, val_acc, lr, is_best, es_counter=None):
        """Print single epoch result"""
        best_marker = " ★" if is_best else ""
        es_status = f"{es_counter}" if es_counter is not None else "-"
        print(f"{epoch+1:^7}|{tr_loss:^12.4f}|{tr_iou:^11.4f}|{tr_acc:^11.4f}|"
              f"{val_loss:^11.4f}|{val_iou:^10.4f}|{val_acc:^10.4f}|{lr:^12.2e}|{es_status:^6}{best_marker}")

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        total_acc = 0.0
        n_batches = len(loader)
        accum_steps = max(1, self.config.GRADIENT_ACCUMULATION_STEPS)

        pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d} [Train]",
                    leave=False, ncols=80, bar_format='{l_bar}{bar:20}{r_bar}')

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            if self.scaler is not None:
                autocast_kwargs = {"device_type": _AMP_DEVICE} if _AMP_DEVICE else {}
                with autocast(**autocast_kwargs):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)
                self.scaler.scale(loss / accum_steps).backward()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                (loss / accum_steps).backward()

            is_last_batch = (batch_idx + 1) == n_batches
            if (batch_idx + 1) % accum_steps == 0 or is_last_batch:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                iou, acc, _, _ = calculate_metrics(
                    preds, masks, self.config.NUM_CLASSES, self.config.IGNORE_INDEX)

            total_loss += loss.item()
            total_iou += iou
            total_acc += acc

            pbar.set_postfix_str(f"loss={loss.item():.3f} iou={iou:.3f}")

        return total_loss / n_batches, total_iou / n_batches, total_acc / n_batches

    def validate(self, loader):
        """Pooled confusion matrix over the whole validation split; the selection
        metric is the 6-class macro IoU from that matrix (not a mean of batch IoUs)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = len(loader)
        full_cm = np.zeros((self.config.NUM_CLASSES, self.config.NUM_CLASSES), dtype=np.int64)

        with torch.no_grad():
            pbar = tqdm(loader, desc="Validating", leave=False, ncols=80,
                       bar_format='{l_bar}{bar:20}{r_bar}')

            for images, masks in pbar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, masks)
                preds = logits.argmax(dim=1)

                total_loss += loss.item()
                full_cm += calculate_confusion_matrix(
                    preds, masks, self.config.NUM_CLASSES, self.config.IGNORE_INDEX)

        summary = summarize_cm(full_cm, self.config.EVAL_CLASSES)
        return {
            'loss': total_loss / n_batches,
            'iou': summary['mean_iou'],
            'acc': summary['overall_accuracy'],
            'class_iou': [summary['per_class'][c]['iou'] for c in self.config.EVAL_CLASSES],
            'class_acc': [summary['per_class'][c]['recall'] for c in self.config.EVAL_CLASSES],
            'confusion_matrix': full_cm,
            'summary': summary,
        }

    def save_checkpoint(self, epoch, is_best=False):
        self.config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_iou': self.best_val_iou,
            'history': self.history,
        }

        if is_best:
            ckpt['band_names'] = list(self.config.BAND_NAMES)
            ckpt['seed'] = self.config.RANDOM_SEED
            best_path = self.config.MODEL_DIR / "best_model.pth"
            torch.save(ckpt, best_path)
            self.best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def train(self, train_loader, val_loader):
        self._print_header()
        self._print_epoch_header()

        stopped_early = False

        for epoch in range(self.config.NUM_EPOCHS):
            current_lr = self.optimizer.param_groups[0]['lr']

            t_epoch = time.perf_counter()
            # Train
            tr_loss, tr_iou, tr_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_result = self.validate(val_loader)
            self.epoch_times.append(time.perf_counter() - t_epoch)
            
            val_loss = val_result['loss']
            val_iou = val_result['iou']
            val_acc = val_result['acc']

            self.scheduler.step()

            # Update history
            self.history['train_loss'].append(tr_loss)
            self.history['train_iou'].append(tr_iou)
            self.history['train_acc'].append(tr_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_iou'].append(val_iou)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            self.history['class_iou_history'].append(val_result['class_iou'])
            self.history['class_acc_history'].append(val_result['class_acc'])

            # Check best
            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
                self.best_epoch = epoch + 1

            # Early stopping counter
            es_counter = None
            if self.early_stopping is not None:
                self.early_stopping(val_iou, epoch)
                es_counter = self.early_stopping.counter

            # Print result
            self._print_epoch_result(epoch, tr_loss, tr_iou, tr_acc, 
                                     val_loss, val_iou, val_acc, current_lr, is_best, es_counter)

            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)

            # Check early stopping
            if self.early_stopping is not None and self.early_stopping.should_stop:
                print("-"*100)
                print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}!")
                print(f"   No improvement for {self.early_stopping.patience} epochs.")
                print(f"   Best Val mIoU: {self.early_stopping.best_value:.4f} @ epoch {self.early_stopping.best_epoch + 1}")
                
                stopped_early = True
                break

        print("-"*100)

        if not stopped_early:
            print(f"\n✓ Training complete! Reached maximum epochs ({self.config.NUM_EPOCHS})")

        # Final metrics come from the selected (best) checkpoint, not the last epoch.
        print(f"\n   Computing final metrics with the best checkpoint (epoch {self.best_epoch})...")
        self.model.load_state_dict(self.best_state)
        self.final_val = self.validate(val_loader)
        self.final_confusion_matrix = self.final_val['confusion_matrix']
        self.stop_epoch = len(self.history['train_loss'])
        print(f"  Best Val mIoU: {self.best_val_iou:.4f} @ epoch {self.best_epoch}\n")

        return self.history

    def get_final_metrics(self):
        """Final metrics of the best checkpoint, restricted to the evaluated classes."""
        if self.final_confusion_matrix is None:
            return None

        ec = self.config.EVAL_CLASSES
        metrics = {k: v[ec] for k, v in compute_metrics_from_cm(self.final_confusion_matrix).items()}
        metrics['confusion_matrix'] = self.final_confusion_matrix[np.ix_(ec, ec)]

        return metrics


# ============================================================
# K-FOLD CROSS VALIDATION TRAINING
# ============================================================

def train_kfold(config=Config):
    """Train dengan K-Fold Cross Validation"""
    
    print("\n" + "="*70)
    print(f"  K-FOLD CROSS VALIDATION ({config.KFOLD_SPLITS} Folds)")
    print("="*70)
    
    # Load all data paths
    patch_paths, mask_paths = load_patch_mask_paths(config)
    patch_paths = np.array(patch_paths)
    mask_paths = np.array(mask_paths)
    
    print(f"\n📂 Total samples: {len(patch_paths)}")
    print(f"   Each fold: ~{len(patch_paths) // config.KFOLD_SPLITS} val samples")
    
    kfold = KFold(n_splits=config.KFOLD_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    
    fold_results = {
        'fold_val_iou': [],
        'fold_val_acc': [],
        'fold_val_loss': [],
        'fold_epochs': [],
        'best_fold': 0,
        'best_iou': 0.0,
    }
    
    all_histories = []
    best_model_state = None
    best_trainer = None
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(patch_paths)):
        print(f"\n{'='*70}")
        print(f"  FOLD {fold + 1}/{config.KFOLD_SPLITS}")
        print(f"  Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")
        print("="*70)
        
        # Create datasets for this fold
        train_patches = patch_paths[train_idx].tolist()
        train_masks = mask_paths[train_idx].tolist()
        val_patches = patch_paths[val_idx].tolist()
        val_masks = mask_paths[val_idx].tolist()
        
        train_loader, val_loader, n_train, n_val = build_fold_dataloaders(
            train_patches, train_masks, val_patches, val_masks, config
        )
        
        # Train this fold
        trainer = SegFormerTrainer(config)
        history = trainer.train(train_loader, val_loader)
        all_histories.append(history)
        
        # Record fold results
        fold_results['fold_val_iou'].append(trainer.best_val_iou)
        fold_results['fold_val_acc'].append(max(history['val_acc']))
        fold_results['fold_val_loss'].append(min(history['val_loss']))
        fold_results['fold_epochs'].append(len(history['train_loss']))
        
        # Track best fold
        if trainer.best_val_iou > fold_results['best_iou']:
            fold_results['best_iou'] = trainer.best_val_iou
            fold_results['best_fold'] = fold + 1
            best_model_state = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
            best_trainer = trainer
        
        # Clear GPU memory
        del trainer.model
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ============ SUMMARY ============
    print("\n" + "="*70)
    print("  K-FOLD CROSS VALIDATION RESULTS")
    print("="*70)
    print(f"\n{'Fold':<8}{'Val IoU':>12}{'Val Acc':>12}{'Val Loss':>12}{'Epochs':>10}")
    print("-"*54)
    
    for i in range(config.KFOLD_SPLITS):
        marker = " ★" if (i + 1) == fold_results['best_fold'] else ""
        print(f"Fold {i+1:<3}{fold_results['fold_val_iou'][i]:>12.4f}"
              f"{fold_results['fold_val_acc'][i]:>12.4f}"
              f"{fold_results['fold_val_loss'][i]:>12.4f}"
              f"{fold_results['fold_epochs'][i]:>10}{marker}")
    
    print("-"*54)
    print(f"{'Mean':<8}{np.mean(fold_results['fold_val_iou']):>12.4f}"
          f"{np.mean(fold_results['fold_val_acc']):>12.4f}"
          f"{np.mean(fold_results['fold_val_loss']):>12.4f}"
          f"{np.mean(fold_results['fold_epochs']):>10.1f}")
    print(f"{'Std':<8}{np.std(fold_results['fold_val_iou']):>12.4f}"
          f"{np.std(fold_results['fold_val_acc']):>12.4f}"
          f"{np.std(fold_results['fold_val_loss']):>12.4f}"
          f"{np.std(fold_results['fold_epochs']):>10.1f}")
    print("="*70)
    
    print(f"\n🏆 Best Fold: {fold_results['best_fold']} (Val mIoU: {fold_results['best_iou']:.4f})")
    
    # Save best model
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_path = config.MODEL_DIR / "best_model_kfold.pth"
    torch.save({
        'model_state_dict': best_model_state,
        'fold_results': fold_results,
        'best_fold': fold_results['best_fold'],
        'mean_val_iou': np.mean(fold_results['fold_val_iou']),
        'std_val_iou': np.std(fold_results['fold_val_iou']),
    }, best_path)
    print(f"💾 Best model saved to: {best_path}")
    
    return fold_results, all_histories, best_trainer


# ============================================================
# MAIN
# ============================================================

def get_class_names(config):
    """Get class names as list"""
    if isinstance(config.CLASS_NAMES, dict):
        return [config.CLASS_NAMES.get(i, f'Class {i}') for i in range(config.NUM_CLASSES)]
    elif isinstance(config.CLASS_NAMES, list):
        return config.CLASS_NAMES
    else:
        return [f'Class {i}' for i in range(config.NUM_CLASSES)]


def parse_args():
    parser = argparse.ArgumentParser(description="Train SegFormer on a channel subset of the 7-band patches.")
    parser.add_argument("--bands", type=str, default=",".join(Config.BAND_NAMES),
                        help="Comma-separated input channels, e.g. R,G,B or R,G,B,VARI,Edge,Texture")
    parser.add_argument("--seed", type=int, default=Config.RANDOM_SEED)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    parser.add_argument("--accum", type=int, default=Config.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS)
    parser.add_argument("--max_train", type=int, default=None,
                        help="Smoke test: use only the first N training patches (and N/2 val patches)")
    return parser.parse_args()


def main():
    args = parse_args()
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    unknown = [b for b in bands if b not in Config.BAND_NAMES]
    if unknown or bands[:3] != ["R", "G", "B"]:
        raise ValueError(f"--bands must start with R,G,B and use {Config.BAND_NAMES}; got {bands}")
    band_idx = [Config.BAND_NAMES.index(b) for b in bands]

    run_name = args.run_name or f"{'-'.join(bands)}_s{args.seed}"
    run_dir = Config.RUNS_DIR / run_name
    Config.BAND_NAMES = bands
    Config.NUM_CHANNELS = len(bands)
    Config.RANDOM_SEED = args.seed
    Config.NUM_EPOCHS = args.epochs
    Config.GRADIENT_ACCUMULATION_STEPS = args.accum
    Config.NUM_WORKERS = args.num_workers
    Config.MODEL_DIR = Config.LOG_DIR = Config.RESULTS_DIR = Config.CHECKPOINT_DIR = run_dir

    manifest, _ = load_split_manifest(Config)
    if Config.CLASS_WEIGHTS is None:
        Config.CLASS_WEIGHTS = manifest["class_weights"]

    # Setup
    logger = setup_logging(Config)
    set_seed(Config.RANDOM_SEED)
    Config.make_dirs()

    # Validate config
    Config.validate_config()
    Config.summary()
    print(f"  Run           : {run_name}  ->  {run_dir}")
    print(f"  Bands         : {bands}")
    print(f"  Class weights : {Config.CLASS_WEIGHTS}")

    print("\n📂 Loading dataset...")
    train_loader, val_loader, n_train, n_val = build_dataloaders(Config, band_idx=band_idx, max_train=args.max_train)
    print(f"   Train samples: {n_train:,}")
    print(f"   Val samples  : {n_val:,}")
    print(f"   Batch size   : {Config.BATCH_SIZE}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches  : {len(val_loader)}")

    t_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = SegFormerTrainer(Config)
    history = trainer.train(train_loader, val_loader)
    final_metrics = trainer.get_final_metrics()
    wall_s = time.perf_counter() - t_start

    class_names = [Config.CLASS_NAMES[c] for c in Config.EVAL_CLASSES]
    plot_path = run_dir / "training_results.png"
    plot_training_results(history, final_metrics, class_names, plot_path)

    final = trainer.final_val['summary']
    print("\n" + "="*70)
    print("  FINAL PER-CLASS METRICS (best checkpoint, pooled over the val split)")
    print("="*70)
    print(f"{'Class':<20}{'IoU':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}{'Support px':>14}")
    print("-"*76)
    for c, name in zip(Config.EVAL_CLASSES, class_names):
        m = final['per_class'][c]
        print(f"{name:<20}{m['iou']:>10.4f}{m['precision']:>12.4f}{m['recall']:>10.4f}{m['f1']:>10.4f}{m['support_px']:>14,}")
    print("-"*76)
    print(f"{'Mean':<20}{final['mean_iou']:>10.4f}{final['mean_precision']:>12.4f}"
          f"{final['mean_recall']:>10.4f}{final['mean_f1']:>10.4f}")
    print(f"{'Overall accuracy':<20}{final['overall_accuracy']:>10.4f}")
    print("="*70)

    n_params = sum(p.numel() for p in trainer.model.parameters())
    metrics_to_save = {
        'run_name': run_name,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': {
            'model_name': Config.MODEL_NAME,
            'bands': bands,
            'num_channels': Config.NUM_CHANNELS,
            'num_classes': Config.NUM_CLASSES,
            'eval_classes': Config.EVAL_CLASSES,
            'seed': Config.RANDOM_SEED,
            'split_seed': manifest['settings']['split_seed'],
            'batch_size': Config.BATCH_SIZE,
            'grad_accum_steps': Config.GRADIENT_ACCUMULATION_STEPS,
            'effective_batch_size': Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS,
            'learning_rate': Config.LEARNING_RATE,
            'weight_decay': Config.WEIGHT_DECAY,
            'lr_scheduler': Config.LR_SCHEDULER,
            'max_epochs': Config.NUM_EPOCHS,
            'loss_function': Config.LOSS_FUNCTION,
            'class_weights': Config.CLASS_WEIGHTS,
            'early_stopping_patience': Config.EARLY_STOPPING_PATIENCE,
            'early_stopping_min_delta': Config.EARLY_STOPPING_MIN_DELTA,
            'train_patches': n_train,
            'val_patches': n_val,
            'max_train': args.max_train,
        },
        'training': {
            'best_epoch': trainer.best_epoch,
            'stop_epoch': trainer.stop_epoch,
            'best_val_miou': trainer.best_val_iou,
        },
        'costs': {
            'parameters': n_params,
            'wall_clock_s': wall_s,
            'mean_epoch_s': float(np.mean(trainer.epoch_times)),
            'peak_gpu_mem_bytes': int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        'history': {k: history[k] for k in ('train_loss', 'val_loss', 'train_iou', 'val_iou', 'train_acc', 'val_acc', 'lr')},
        'epoch_times_s': trainer.epoch_times,
        'val_best_checkpoint': final,
        'val_confusion_matrix': trainer.final_confusion_matrix.tolist(),
    }
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)

    print("\n" + "="*70)
    print("  TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"  Best Model    : {run_dir / 'best_model.pth'}")
    print(f"  Best Val mIoU : {trainer.best_val_iou:.4f} @ epoch {trainer.best_epoch} (stopped at {trainer.stop_epoch})")
    print(f"  Wall clock    : {wall_s / 60:.1f} min, {np.mean(trainer.epoch_times):.0f} s/epoch")
    print(f"  Results Plot  : {plot_path}")
    print(f"  Metrics JSON  : {metrics_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
