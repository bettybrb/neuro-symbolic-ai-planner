import random
import time
import contextlib
import requests
from io import BytesIO
from collections import defaultdict
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

import unittest
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch
from torch.utils.data import TensorDataset


# =============================================================================
# DATASET & MODEL
# =============================================================================

class CIFAR100Filtered(Dataset):
    """
    CIFAR-100 dataset wrapper with preprocessing and train/val split support.
    
    Args:
        root (str): Directory to store/load CIFAR-100 data
        split (str): Either "train" or "val" to specify which split to use
        transform (callable, optional): Transform to apply to images. If None, uses default.
    
    Attributes:
        dataset: The underlying torchvision CIFAR100 dataset
    """
    
    def __init__(self, root="./data", split="train", transform=None, augment=False, image_size=224):
        # TODO: Validate that split is either "train" or "val"
        # Use assert to check this condition
        assert split in {"train", "val"}, "split must be 'train' or 'val'"
        
        # TODO: If transform is None, create a default transform that:
        # 1. Resizes images to 224x224 (use transforms.Resize)
        # 2. Converts to tensor (use transforms.ToTensor)
        # 3. Normalizes with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # Use transforms.Compose to chain these together
        if transform is None:
            resize_size = max(image_size, int(round(image_size * 1.14)))
            if augment and split == "train":
                transform = transforms.Compose([
                    transforms.Resize(resize_size),
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
        
        # TODO: Load CIFAR-100 using datasets.CIFAR100
        # Set train=True for split="train", train=False for split="val"
        # Remember to set download=True and pass the transform
        def find_cifar_root(start: str, max_levels: int = 3):
            cur = os.path.abspath(start)
            for _ in range(max_levels + 1):
                if os.path.exists(os.path.join(cur, "cifar-100-python")):
                    return cur
                parent = os.path.dirname(cur)
                if parent == cur:
                    break
                cur = parent
            return None

        env_root = os.environ.get("CIFAR100_ROOT")
        local_root = find_cifar_root(os.getcwd())
        resolved_root = root
        download = True
        for candidate in (root, env_root, local_root):
            if candidate and os.path.exists(os.path.join(candidate, "cifar-100-python")):
                resolved_root = candidate
                download = False
                break

        self.dataset = datasets.CIFAR100(
            root=resolved_root,
            train=(split == "train"),
            download=download,
            transform=transform,
        )
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        # TODO: Return the length of the underlying dataset
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
        
        Returns:
            tuple: (image, label) where image is a transformed tensor and label is an integer
        """
        # TODO: Index into self.dataset and return the (image, label) tuple
        return self.dataset[idx]



class ImageEncoder(nn.Module):
    """
    MobileNetV3-based image encoder with trainable projection head.
    
    This model consists of two parts:
    1. Frozen pretrained MobileNetV3 backbone (feature extractor)
    2. Trainable projection head (maps features to embedding space)
    
    Args:
        proj_dim (int): Dimension of the output projection embeddings
        device (str): Device to place the model on ("cuda" or "cpu")
    
    Attributes:
        backbone: Frozen MobileNetV3 feature extractor (output: 576-dim)
        projection: Trainable MLP that projects to proj_dim
    """
    
    def __init__(self, proj_dim=258, device="cuda", input_size=None):
        super().__init__()
        self.device = device
        self.backbone_trainable = False
        if input_size is None:
            input_size = 224 if str(device) != "cpu" else 160
        self.input_size = int(input_size)
        
        # TODO: Load pretrained MobileNetV3-Small
        # Use models.mobilenet_v3_small with DEFAULT weights
        # Extract all layers except the final classifier using list(base.children())[:-1]
        # Wrap in nn.Sequential, move to device, and set to eval mode
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        try:
            base = models.mobilenet_v3_small(weights=weights)
        except Exception:
            cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
            cache_path = cache_dir / Path(weights.url).name
            if cache_path.exists():
                base = models.mobilenet_v3_small(weights=None)
                state_dict = torch.load(cache_path, map_location=device)
                base.load_state_dict(state_dict)
                warnings.warn("Loaded MobileNetV3 weights from local cache (offline).")
            else:
                base = models.mobilenet_v3_small(weights=None)
                warnings.warn(
                    "MobileNetV3 pretrained weights unavailable; using random init. "
                    "Download weights or place them in the torch hub cache for better results."
                )
        self.backbone = nn.Sequential(*list(base.children())[:-1]).to(device)
        self.backbone.eval()
        
        # TODO: Freeze the backbone parameters
        # Loop through self.backbone.parameters() and set requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # TODO: Create trainable projection head
        # Architecture: Linear(576 -> 512) -> BatchNorm1d(512) -> ReLU -> Linear(512 -> proj_dim)
        # Use nn.Sequential to chain the layers
        # Move to device using .to(device)
        self.projection = nn.Sequential(
            nn.Linear(576, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        ).to(device)

    def train(self, mode=True):
        super().train(mode)
        # Keep backbone frozen unless explicitly unfrozen.
        if self.backbone_trainable:
            self.backbone.train(mode)
        else:
            self.backbone.eval()
        return self

    def unfreeze_backbone(self, last_n: int = 2, train_bn: bool = False) -> None:
        """Unfreeze the last N blocks of the backbone for fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = False

        features = None
        if isinstance(self.backbone, nn.Sequential) and len(self.backbone) > 0:
            if isinstance(self.backbone[0], nn.Sequential):
                features = self.backbone[0]
            else:
                features = self.backbone

        if features is not None:
            blocks = list(features.children())
            if blocks:
                for block in blocks[-max(1, last_n):]:
                    for p in block.parameters():
                        p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True

        if train_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
                    for p in m.parameters():
                        p.requires_grad = True

        self.backbone_trainable = True

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load checkpoints saved with older module naming conventions."""
        if not isinstance(state_dict, dict):
            return super().load_state_dict(state_dict, strict=strict)

        model_keys = set(super().state_dict().keys())

        def strip_prefix(sd, prefix):
            if sd and all(k.startswith(prefix) for k in sd):
                return {k[len(prefix):]: v for k, v in sd.items()}
            return None

        def rename_keys(sd):
            remapped = {}
            for k, v in sd.items():
                nk = k
                if nk.startswith("projection_head."):
                    nk = "projection." + nk[len("projection_head."):]
                if nk.startswith("proj."):
                    nk = "projection." + nk[len("proj."):]
                if nk.startswith("backbone.features."):
                    nk = "backbone.0." + nk[len("backbone.features."):]
                if nk.startswith("features."):
                    nk = "backbone.0." + nk[len("features."):]
                remapped[nk] = v
            return remapped

        def insert_backbone_level(sd):
            remapped = {}
            for k, v in sd.items():
                if k.startswith("backbone.0.") and not k.startswith("backbone.0.0."):
                    k = "backbone.0.0." + k[len("backbone.0."):]
                remapped[k] = v
            return remapped

        def remove_backbone_level(sd):
            remapped = {}
            for k, v in sd.items():
                if k.startswith("backbone.0.0."):
                    k = "backbone.0." + k[len("backbone.0.0."):]
                remapped[k] = v
            return remapped

        candidates = [state_dict]
        for prefix in ("module.", "model.", "image_encoder.", "encoder."):
            stripped = strip_prefix(state_dict, prefix)
            if stripped is not None:
                candidates.append(stripped)

        expanded = []
        for cand in candidates:
            expanded.append(cand)
            renamed = rename_keys(cand)
            expanded.append(renamed)
            expanded.append(insert_backbone_level(cand))
            expanded.append(remove_backbone_level(cand))
            expanded.append(insert_backbone_level(renamed))
            expanded.append(remove_backbone_level(renamed))

        def score(sd):
            return len(model_keys & set(sd.keys()))

        best = max(expanded, key=score)
        return super().load_state_dict(best, strict=strict)
    
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
        
        Returns:
            tuple: (backbone_features, projected_embeddings)
                - backbone_features: Raw features from MobileNet (batch_size, 576)
                - projected_embeddings: Projected embeddings (batch_size, proj_dim)
        """
        # TODO: Extract features using the frozen backbone
        # Use torch.no_grad() context to save memory
        # Flatten the output to shape (batch_size, 576) using .flatten(1)
        
        # TODO: Project features through the trainable projection head
        # Pass the flattened features through self.projection
        
        # Return both the backbone features and projected embeddings as a tuple
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if not x.is_floating_point():
            x = x.float()
        target_size = self.input_size
        if x.shape[-2:] != (target_size, target_size):
            x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)

        x_min = float(x.amin().item())
        x_max = float(x.amax().item())
        if x_max > 1.5 and x_min >= 0.0:
            x = x / 255.0
            x_min = x_min / 255.0
            x_max = x_max / 255.0
        if x_min >= 0.0 and x_max <= 1.0 + 1e-3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = (x - mean) / std
        if self.backbone_trainable:
            feats = self.backbone(x).flatten(1)
        else:
            with torch.no_grad():
                feats = self.backbone(x).flatten(1)
        proj = self.projection(feats)
        return feats, proj

# =============================================================================
# DATA & TRAINING UTILITIES
# =============================================================================

def filter_dataset_indices(dataset, valid_labels):
    """Return indices of samples with labels in valid_labels set."""
    return [i for i, label in enumerate(dataset.dataset.targets) if label in valid_labels]

def create_data_splits(indices, val_ratio=0.2, seed=42):
    """Split indices into train/val sets."""
    np.random.seed(seed)
    indices = np.array(indices)
    np.random.shuffle(indices)
    split_idx = int((1 - val_ratio) * len(indices))
    return indices[:split_idx].tolist(), indices[split_idx:].tolist()

def create_dataloaders(train_idx, val_idx, test_idx, batch_sizes,
                       num_workers=2, pin_memory=False, persistent_workers=False,
                       prefetch_factor=2, image_size=224):
    """Create train, val, and test dataloaders."""
    datasets = {
        'train': Subset(CIFAR100Filtered(split="train", augment=True, image_size=image_size), train_idx),
        'val': Subset(CIFAR100Filtered(split="train", augment=False, image_size=image_size), val_idx),
        'test': Subset(CIFAR100Filtered(split="val", augment=False, image_size=image_size), test_idx)
    }
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return {
        k: DataLoader(
            v,
            batch_size=batch_sizes['train' if k == 'train' else 'eval'],
            shuffle=(k == 'train'),
            **loader_kwargs,
        )
        for k, v in datasets.items()
    }



def compute_contrastive_loss(visual_proj, text_emb, temperature):
    """
    Compute symmetric InfoNCE (contrastive) loss for vision-language alignment.
    
    This loss encourages matching pairs (image, correct_text) to have high similarity
    while pushing apart non-matching pairs (image, wrong_text).
    
    Args:
        visual_proj (torch.Tensor): Projected visual embeddings, shape (batch_size, proj_dim)
        text_emb (torch.Tensor): Text embeddings for the batch, shape (batch_size, proj_dim)
        temperature (float): Temperature parameter to scale logits (typically 0.07)
    
    Returns:
        torch.Tensor: Scalar loss value (symmetric InfoNCE loss)
    
    Mathematical formulation:
        1. Normalize both embeddings to unit vectors
        2. Compute similarity matrix: S = (visual @ text.T) / temperature
        3. Apply cross-entropy loss treating diagonal as correct matches
        4. Average image-to-text and text-to-image losses for symmetry
    """
    
    # TODO: Normalize visual_proj to unit vectors (L2 normalization)
    # Use F.normalize with p=2 and dim=1
    # Shape: (batch_size, proj_dim)
    
    # TODO: Normalize text_emb to unit vectors (L2 normalization)
    # Use F.normalize with p=2 and dim=1
    # Shape: (batch_size, proj_dim)
    
    # TODO: Compute similarity matrix (logits)
    # Matrix multiply: normalized_visual @ normalized_text.T
    # Divide by temperature to scale the logits
    # Use torch.matmul for matrix multiplication
    # Shape: (batch_size, batch_size)
    
    # TODO: Create ground truth labels
    # For a batch of N samples, correct matches are on the diagonal
    # Labels should be [0, 1, 2, ..., N-1]
    # Use torch.arange to create labels on the same device as visual_proj
    
    # TODO: Compute image-to-text loss
    # Treat each row as logits for which text matches this image
    # Use F.cross_entropy(logits, labels)
    
    # TODO: Compute text-to-image loss
    # Treat each column as logits for which image matches this text
    # Transpose the logits matrix and use F.cross_entropy(logits.T, labels)
    
    # TODO: Return symmetric loss
    # Average the two losses: (i2t_loss + t2i_loss) / 2
    v = F.normalize(visual_proj, p=2, dim=1)
    t = F.normalize(text_emb, p=2, dim=1)
    logits = torch.matmul(v, t.T) / temperature
    labels = torch.arange(logits.shape[0], device=visual_proj.device)
    i2t_loss = F.cross_entropy(logits, labels)
    t2i_loss = F.cross_entropy(logits.T, labels)
    return (i2t_loss + t2i_loss) / 2
    

def build_soft_full_vocab_targets(class_words, full_vocab_words, full_text_emb,
                                  top_k=8, tau=0.07, exclude_words=None):
    """Build soft target distributions over the full vocabulary for each class."""
    if not class_words or not full_vocab_words:
        return None

    exclude_words = set(exclude_words or [])
    vocab_index = {w: i for i, w in enumerate(full_vocab_words)}
    class_indices = [vocab_index[w] for w in class_words if w in vocab_index]
    if not class_indices:
        return None

    emb = torch.as_tensor(full_text_emb, dtype=torch.float32)
    emb = F.normalize(emb, p=2, dim=1)
    class_vecs = emb[class_indices]
    sims = torch.matmul(class_vecs, emb.T)

    if exclude_words:
        for w in exclude_words:
            idx = vocab_index.get(w)
            if idx is not None and w not in class_words:
                sims[:, idx] = -1e9

    for row, w in enumerate(class_words):
        idx = vocab_index.get(w)
        if idx is not None:
            sims[row, idx] = sims[row, idx].max() + 1e-4

    k = max(1, min(int(top_k), sims.shape[1]))
    topk = torch.topk(sims, k=k, dim=1)
    weights = F.softmax(topk.values / float(tau), dim=1)
    targets = torch.zeros_like(sims)
    targets.scatter_(1, topk.indices, weights)
    return targets


def run_epoch(model, dataloader, text_emb, class_words, label_to_word, optimizer,
              temperature, device, mode='train', label_to_class_idx=None,
              full_text_emb=None, label_to_full_idx=None, full_vocab_weight=0.0,
              full_vocab_soft_targets=None, full_vocab_soft_weight=0.0,
              use_amp=False, scaler=None):
    """
    Run one epoch of training or evaluation for contrastive vision-language learning.
    
    Handles a single pass over the dataset, computes loss and mean similarity,
    and updates the model if in training mode.
    
    Args:
        model (nn.Module): Image encoder model.
        dataloader (DataLoader): DataLoader providing (image, label) batches.
        text_emb (torch.Tensor): Embeddings for all class words, shape (num_classes, proj_dim).
        class_words (list): List of word strings for each class label index.
        label_to_word (dict): Maps integer label to the corresponding word string.
        optimizer (torch.optim.Optimizer or None): Optimizer for model parameters (set to None in eval mode).
        temperature (float): Contrastive loss temperature parameter.
        device (str or torch.device): Device for computation.
        mode (str): 'train' or 'eval' (evaluation).
    
    Returns:
        tuple: (mean_loss, mean_similarity)
            mean_loss: Average loss over the epoch.
            mean_similarity: Average cosine similarity between visual and aligned text embeddings.
    """

    # TODO: Set model mode (train or eval)
    # Use model.train() for training, model.eval() for evaluation
    model.train() if mode == 'train' else model.eval()

    if not torch.is_tensor(text_emb):
        text_emb = torch.tensor(text_emb, dtype=torch.float32, device=device)
    else:
        text_emb = text_emb.to(device)

    if full_text_emb is not None:
        if not torch.is_tensor(full_text_emb):
            full_text_emb = torch.tensor(full_text_emb, dtype=torch.float32, device=device)
        else:
            full_text_emb = full_text_emb.to(device)

    if label_to_class_idx is None:
        label_to_class_idx = {
            k: class_words.index(v)
            for k, v in label_to_word.items()
            if v in class_words
        }

    total_loss = 0
    total_sim = 0
    count = 0

    # TODO: Choose correct context manager:
    # Use torch.no_grad() for eval, torch.enable_grad() for training

    # Loop over dataloader
    # for images, labels in tqdm(dataloader):
    #     - Move images and labels to device
    #     - Forward model to get visual features and projected embeddings
    #     - Build batch_text_idx: index for each label to get its word embedding
    #     - batch_text_emb: Index text_emb with batch_text_idx (must be in correct device and dtype)
    #     - Compute loss using compute_contrastive_loss
    #     - If training:
    #         - Zero gradients, backward, and step optimizer
    #     - Accumulate loss and similarity (mean per batch), weighted by batch size
    #     - For similarity, use F.normalize for both visual and text features, multiply and sum per row

    # Return average loss and average similarity over all examples in the epoch

    context = torch.no_grad() if mode == 'eval' else torch.enable_grad()
    amp_ctx = contextlib.nullcontext()
    if use_amp and device.type == 'cuda':
        amp_ctx = torch.cuda.amp.autocast()
    with context:
        for images, labels in tqdm(dataloader, desc=f"[{mode.upper()}]"):
            images, labels = images.to(device), labels.to(device)
            with amp_ctx:
                _, visual_proj = model(images)

                batch_text_idx = [label_to_class_idx[l.item()] for l in labels]
                batch_text_emb = text_emb[batch_text_idx]

                loss = compute_contrastive_loss(visual_proj, batch_text_emb, temperature)
                if full_text_emb is not None and (full_vocab_weight or full_vocab_soft_weight):
                    v_norm = F.normalize(visual_proj, p=2, dim=1)
                    t_full = F.normalize(full_text_emb, p=2, dim=1)
                    full_logits = torch.matmul(v_norm, t_full.T) / temperature
                    extra_loss = 0.0
                    if label_to_full_idx is not None and full_vocab_weight:
                        full_labels = torch.tensor([label_to_full_idx[l.item()] for l in labels], device=device)
                        hard_loss = F.cross_entropy(full_logits, full_labels)
                        extra_loss = extra_loss + (full_vocab_weight * hard_loss)
                    if full_vocab_soft_targets is not None and full_vocab_soft_weight:
                        target_probs = full_vocab_soft_targets[batch_text_idx].to(device)
                        log_probs = F.log_softmax(full_logits, dim=1)
                        soft_loss = -(target_probs * log_probs).sum(dim=1).mean()
                        extra_loss = extra_loss + (full_vocab_soft_weight * soft_loss)
                    if extra_loss:
                        loss = loss + extra_loss

            if mode == 'train':
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * len(images)

            V = F.normalize(visual_proj, p=2, dim=1)
            S = F.normalize(batch_text_emb, p=2, dim=1)

            total_sim += (V * S).sum(dim=1).sum().item()
            count += len(images)
    return total_loss / max(count, 1), total_sim / max(count, 1)
            
            

def train_with_early_stopping(model, dataloaders, text_emb, class_words, label_to_word,
                              config, device, full_text_emb=None, label_to_full_idx=None,
                              full_vocab_words=None, full_vocab_exclude_words=None):
    """
    Train model with early stopping based on validation similarity.
    
    Trains the model for multiple epochs, monitoring validation performance and stopping
    early if no improvement is seen for a specified number of epochs (patience).
    
    Args:
        model (nn.Module): Image encoder model to train.
        dataloaders (dict): Dictionary with keys 'train' and 'val', each containing a DataLoader.
        text_emb (torch.Tensor): Text embeddings for all classes, shape (num_classes, proj_dim).
        class_words (list): List of class word strings.
        label_to_word (dict): Maps integer labels to word strings.
        config (dict): Training configuration containing:
            - 'lr': Learning rate
            - 'weight_decay': Weight decay for optimizer
            - 'epochs': Maximum number of epochs
            - 'temperature': Temperature for contrastive loss
            - 'patience': Early stopping patience (epochs without improvement)
            - 'save_path': Path to save best model checkpoint
        device (str or torch.device): Device for computation.
        full_text_emb (torch.Tensor or None): Full vocab text embeddings for optional extra loss.
        label_to_full_idx (dict or None): Mapping from CIFAR label to full-vocab index.
        full_vocab_soft_targets (torch.Tensor or None): Soft targets over full vocab.
        full_vocab_soft_weight (float): Weight for soft full-vocab loss.
    
    Returns:
        tuple: (history, best_epoch, best_val_sim, best_val_loss)
            history: Dictionary tracking 'train_loss', 'val_loss', 'val_similarity', 'learning_rate'
            best_epoch: Epoch number with best validation similarity
            best_val_sim: Best validation similarity achieved
            best_val_loss: Validation loss at best epoch
    """
    
    # TODO: Create optimizer for trainable parameters (model.projection.parameters())
    # Use torch.optim.AdamW with lr and weight_decay from config
    
    # TODO: Create learning rate scheduler
    # Use torch.optim.lr_scheduler.CosineAnnealingLR with T_max=config['epochs']
    
    # TODO: Initialize tracking variables
    # best_val_sim = -inf (we want to maximize similarity)
    # patience_counter = 0
    # best_epoch = 0
    # history = defaultdict(list) to track metrics
    
    # Print training header
    
    # TODO: Training loop
    # try:
    #     for epoch in range(1, config['epochs'] + 1):
    #         - Run training epoch using run_epoch (mode='train')
    #         - Run validation epoch using run_epoch (mode='eval', optimizer=None)
    #         - Step the scheduler
    #         - Get current learning rate using scheduler.get_last_lr()[0]
    #         - Append metrics to history: train_loss, val_loss, val_similarity, learning_rate
    #         - Print epoch summary
    #         
    #         - If val_sim > best_val_sim:
    #             - Update best_val_sim, best_val_loss, best_epoch
    #             - Reset patience_counter to 0
    #             - Save checkpoint using torch.save with:
    #                 epoch, model_state_dict, val_loss, val_similarity,
    #                 class_words, text_embeddings (cpu), history, projection_head state
    #             - Print success message
    #         - Else:
    #             - Increment patience_counter
    #             - Print no improvement message
    #         
    #         - If patience_counter >= config['patience']:
    #             - Print early stopping message
    #             - Break
    # 
    # except Exception as e:
    #     - Print error message
    #     - Print message about continuing with best saved model
    
    # TODO: Return history (as dict), best_epoch, best_val_sim, best_val_loss

    optimizer = torch.optim.AdamW(
        model.projection.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    best_val_sim, patience_counter, best_epoch = -float('inf'), 0, 0
    history = defaultdict(list)
    label_to_class_idx = {
        k: class_words.index(v)
        for k, v in label_to_word.items()
        if v in class_words
    }
    full_vocab_weight = float(config.get('full_vocab_weight', 0.0))
    full_vocab_soft_weight = float(config.get('full_vocab_soft_weight', 0.0))
    full_vocab_soft_targets = None
    if full_vocab_soft_weight and full_text_emb is not None and full_vocab_words:
        full_vocab_soft_targets = build_soft_full_vocab_targets(
            class_words,
            full_vocab_words,
            full_text_emb,
            top_k=int(config.get("full_vocab_top_k", 8)),
            tau=float(config.get("full_vocab_tau", 0.07)),
            exclude_words=full_vocab_exclude_words,
        )
    use_amp = bool(config.get('use_amp', False)) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    unfreeze_epoch = int(config.get("unfreeze_backbone_at", 0) or 0)
    backbone_last_n = int(config.get("backbone_last_n", 2))
    backbone_lr = float(config.get("backbone_lr", config['lr'] * 0.1))
    train_backbone_bn = bool(config.get("train_backbone_bn", False))
    
    max_minutes = float(config.get('max_minutes', 0.0))
    start_time = time.time()
    print(f"\n{'='*70}\nTraining (max {config['epochs']} epochs, patience={config['patience']})\n{'='*70}")
    
    try:
        for epoch in range(1, config['epochs'] + 1):
            if unfreeze_epoch and epoch == unfreeze_epoch:
                model.unfreeze_backbone(last_n=backbone_last_n, train_bn=train_backbone_bn)
                backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
                if backbone_params:
                    optimizer = torch.optim.AdamW(
                        [
                            {'params': model.projection.parameters(), 'lr': config['lr']},
                            {'params': backbone_params, 'lr': backbone_lr},
                        ],
                        weight_decay=config['weight_decay'],
                    )
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(1, config['epochs'] - epoch + 1)
                    )
                    if use_amp:
                        scaler = torch.cuda.amp.GradScaler()
                    print(f"  → Unfroze backbone (last {backbone_last_n} blocks)")
            train_loss, _ = run_epoch(
                model, dataloaders['train'], text_emb, class_words, label_to_word,
                optimizer, config['temperature'], device, 'train',
                label_to_class_idx=label_to_class_idx,
                full_text_emb=full_text_emb,
                label_to_full_idx=label_to_full_idx,
                full_vocab_weight=full_vocab_weight,
                full_vocab_soft_targets=full_vocab_soft_targets,
                full_vocab_soft_weight=full_vocab_soft_weight,
                use_amp=use_amp,
                scaler=scaler,
            )
            val_loss, val_sim = run_epoch(
                model, dataloaders['val'], text_emb, class_words, label_to_word,
                None, config['temperature'], device, 'eval',
                label_to_class_idx=label_to_class_idx,
                full_text_emb=full_text_emb,
                label_to_full_idx=label_to_full_idx,
                full_vocab_weight=full_vocab_weight,
                full_vocab_soft_targets=full_vocab_soft_targets,
                full_vocab_soft_weight=full_vocab_soft_weight,
                use_amp=use_amp,
            )
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Update history
            for metric, value in zip(['train_loss', 'val_loss', 'val_similarity', 'learning_rate'], 
                                    [train_loss, val_loss, val_sim, current_lr]):
                history[metric].append(value)
            
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Sim: {val_sim:.4f} | LR: {current_lr:.6f}")
            
            if val_sim > best_val_sim:
                best_val_sim, best_val_loss, best_epoch, patience_counter = val_sim, val_loss, epoch, 0
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_loss': val_loss,
                        'val_similarity': val_sim,
                        'class_words': class_words,
                        'text_embeddings': text_emb.cpu(),
                        'history': dict(history),
                        'projection_head': model.projection.state_dict(),
                    },
                    config['save_path']
                )
                print(f"  ✓ New best model saved! (Val Sim: {val_sim:.4f})")
            #if ur training loss goes down but validation one doesnt ur model isnt working/learning well.
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{config['patience']})")
            
            if patience_counter >= config['patience']:
                print(f"\n{'='*70}\nEarly stopping at epoch {epoch}\nBest: {best_epoch} (Val Sim: {best_val_sim:.4f})\n{'='*70}")
                break
            if max_minutes > 0 and (time.time() - start_time) / 60.0 >= max_minutes:
                print(f"\n{'='*70}\nTime budget reached ({max_minutes:.0f} min). Stopping early.\n{'='*70}")
                break
    except Exception as e:
        print(f"\n⚠️  Training crashed with error: {str(e)}")
        print(f"Attempting to continue with best saved model...")
    
    return dict(history), best_epoch, best_val_sim, best_val_loss
    
#add regularisation, account for synonyms, 

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def collect_embeddings(model, dataloader, device):
    """Collect all embeddings and labels from dataset."""
    model.eval()
    all_visual, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Collecting embeddings"):
            images = images.to(device)
            _, visual_proj = model(images)
            all_visual.append(F.normalize(visual_proj, p=2, dim=1).cpu())
            all_labels.extend(labels.tolist())
    return torch.cat(all_visual, dim=0).numpy(), all_labels


def compute_alignment_metrics(visual_emb, labels, text_emb, class_words, label_to_word):
    """Compute comprehensive alignment metrics in one pass."""
    # Per-class statistics
    class_sims = defaultdict(list)
    for i, label in enumerate(labels):
        if (word := label_to_word[label]) in class_words:
            sim = np.dot(visual_emb[i], text_emb[class_words.index(word)])
            class_sims[word].append(sim)
    
    stats = sorted([{
        'word': word, 'mean': np.mean(sims), 'std': np.std(sims), 
        'min': np.min(sims), 'max': np.max(sims), 'count': len(sims)
    } for word, sims in class_sims.items()], key=lambda x: x['mean'], reverse=True)
    
    # Retrieval metrics
    sim_matrix = cosine_similarity(visual_emb, text_emb)
    i2t_recalls = {k: 0 for k in [1, 5, 10]}
    t2i_recalls = {k: 0 for k in [1, 5, 10]}
    
    # Image-to-text retrieval
    for i, label in enumerate(labels):
        if (word := label_to_word[label]) in class_words:
            correct_idx = class_words.index(word)
            ranking = np.argsort(-sim_matrix[i])
            for k in i2t_recalls:
                if correct_idx in ranking[:k]: i2t_recalls[k] += 1
    
    # Text-to-image retrieval  
    for class_idx, word in enumerate(class_words):
        class_img_idx = [i for i, l in enumerate(labels) if label_to_word[l] == word]
        if class_img_idx:
            ranking = np.argsort(-sim_matrix[:, class_idx])
            for k in t2i_recalls:
                if any(idx in ranking[:k] for idx in class_img_idx): t2i_recalls[k] += 1
    
    return stats, i2t_recalls, t2i_recalls, sim_matrix


def print_analysis_results(stats, i2t_recalls, t2i_recalls, n_samples, n_classes):
    """Print comprehensive analysis results."""
    print("\n📊 Per-Class Similarity Analysis:")
    print("-" * 70)
    for title, data in [("Top 10 Best Aligned Classes:", stats[:10]), 
                        ("Bottom 10 Worst Aligned Classes:", stats[-10:])]:
        print(f"\n{title}")
        for i, s in enumerate(data, 1):
            print(f"{i:2d}. {s['word']:15s} | Mean: {s['mean']:.4f} ± {s['std']:.4f}")
    
    print("\n📊 Retrieval Performance:")
    print("-" * 70)
    for name, recalls, total in [("Image-to-Text", i2t_recalls, n_samples), 
                                 ("Text-to-Image", t2i_recalls, n_classes)]:
        print(f"\n{name} Retrieval (Recall@K):")
        for k, count in recalls.items():
            print(f"  Recall@{k:2d}: {count/total*100:.2f}% ({count}/{total})")

def print_example_retrievals(sim_matrix, labels, class_words, label_to_word, n_examples=5):
    """Print text-based retrieval examples."""
    print("\n📸 Example Image-to-Text Retrievals:")
    print("-" * 70)
    
    display_idx = np.random.choice(len(labels), size=n_examples, replace=False)
    
    for idx in display_idx:
        label = labels[idx]
        true_word = label_to_word[label]
        
        sims = sim_matrix[idx]
        top_5_idx = np.argsort(-sims)[:5]
        top_5_words = [class_words[i] for i in top_5_idx]
        top_5_sims = [sims[i] for i in top_5_idx]
        
        correct_sim = sims[class_words.index(true_word)]
        correct_rank = np.where(np.argsort(-sims) == class_words.index(true_word))[0][0] + 1
        
        print(f"\nTest Image #{idx}:")
        print(f"  True class: '{true_word}' (similarity: {correct_sim:.4f}, rank: {correct_rank})")
        print(f"  Top 5 predictions:")
        for rank, (word, sim) in enumerate(zip(top_5_words, top_5_sims), 1):
            marker = "✓" if word == true_word else " "
            print(f"    {rank}. {marker} {word:15s} (similarity: {sim:.4f})")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_visualizations(sim_matrix, labels, class_words, label_to_word, test_indices, images=None, names=None, predictions=None):
    """Create all visualizations in one coordinated function."""
    # OOD analysis if provided
    if images and names and predictions:
        print(f"\n📸 Creating OOD visualization for {len(images)} images...")
        n_imgs = len(images)
        n_cols = min(4, n_imgs)
        n_rows = (n_imgs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 7*n_rows))
        axes = [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
        
        for i, (img, name, pred) in enumerate(zip(images, names, predictions)):
            axes[i].imshow(img); axes[i].axis('off')
            pred_text = f"{name.upper()}\n\nTop matches:\n"
            for rank, (word, sim) in enumerate(zip(pred['words'][:5], pred['sims'][:5]), 1):
                pred_text += f"{rank}. {word} ({sim:.3f})\n"
            axes[i].set_title(pred_text, fontsize=11, ha='center', color='darkblue', fontweight='bold', pad=12)
        
        for j in range(i+1, len(axes)): 
            axes[j].axis('off'); axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('ood_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    else:
        print("\n📊 Creating confusion matrix...")
        n_classes = len(class_words)
        conf_matrix = np.zeros((n_classes, n_classes))
        
        for i, label in enumerate(labels):
            if (word := label_to_word[label]) in class_words:
                true_idx = class_words.index(word)
                pred_idx = np.argmax(sim_matrix[i])
                conf_matrix[true_idx, pred_idx] += 1
        
        conf_matrix = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(conf_matrix, xticklabels=class_words, yticklabels=class_words,
                    cmap='Blues', ax=ax, cbar_kws={'label': 'Probability'}, square=True)
        ax.set_xlabel('Predicted Class'); ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix (All Classes)', fontsize=14, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Retrieval examples
        print("\n📸 Creating retrieval examples...")
        test_raw = CIFAR100Filtered(split="val", transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for plot_idx, ax in enumerate(axes.flatten()):
            if plot_idx >= 12: break
            ex_idx = random.randint(0, len(labels)-1)
            original_idx = test_indices[ex_idx]
            img, label = test_raw[original_idx]
            
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis('off')
            
            true_word = label_to_word[label]
            sims = sim_matrix[ex_idx]
            top_5_idx = np.argsort(-sims)[:5]
            top_5_words = [class_words[i] for i in top_5_idx]
            top_5_sims = [sims[i] for i in top_5_idx]

            # Build prediction text
            pred_text = f"GT: {true_word}\n"
            for rank, (word, sim) in enumerate(zip(top_5_words, top_5_sims), 1):
                marker = "✓" if word == true_word else "✗"
                pred_text += f"{rank}. {marker} {word}: {sim:.2f}\n"

            # --- COLOR LOGIC CHANGE ---
            if top_5_words[0] == true_word:
                title_color = "green"          # correct top-1
            elif true_word in top_5_words:
                title_color = "#CC8A00"        # amber
            else:
                title_color = "red"            # incorrect
            # --------------------------

            ax.set_title(
                pred_text, fontsize=9, ha='left',
                fontfamily='monospace',
                color=title_color, fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig('retrieval_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
        

# =============================================================================
# OOD PROCESSING
# =============================================================================

BASIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "on", "in", "to", "for", "from", "with",
    "at", "by", "is", "are", "was", "were", "be", "been", "this", "that", "these",
    "those", "it", "its", "as", "into", "over", "under", "above", "below", "near",
    "behind", "front", "left", "right", "up", "down", "off", "out", "across"
}

GENERIC_TOKENS = {
    "white", "black", "blue", "red", "green", "yellow", "brown", "pink", "purple",
    "gray", "grey", "large", "small", "tall", "short", "long", "standing", "sitting",
    "walking", "riding", "holding", "wearing", "wood", "wooden", "metal", "plastic",
    "glass", "top", "bottom", "side", "middle", "center", "centre", "background",
    "foreground", "left", "right", "front", "back"
}


def filter_ood_vocabulary(vocab_words, text_emb, keep_words=None, stopwords=None,
                          extra_exclude=None, top_k_exclude=40, min_len=3,
                          reject_suffixes=("ing", "ed", "ous", "ive", "al", "ful", "less", "able", "ish"),
                          reject_underscore=True):
    """Filter generic tokens from the OOD vocabulary while preserving keep_words."""
    keep_set = set(keep_words or [])
    exclude = set(stopwords or set()) | set(extra_exclude or set())
    if top_k_exclude:
        exclude.update(vocab_words[:top_k_exclude])

    emb_array = text_emb
    if torch.is_tensor(emb_array):
        emb_array = emb_array.detach().cpu().numpy()
    emb_array = np.asarray(emb_array)

    kept_words = []
    kept_indices = []
    for i, word in enumerate(vocab_words):
        if word in keep_set:
            kept_words.append(word)
            kept_indices.append(i)
            continue
        if len(word) < min_len:
            continue
        if reject_underscore and "_" in word:
            continue
        if any(ch.isdigit() for ch in word):
            continue
        if any(word.endswith(suf) for suf in reject_suffixes):
            continue
        if word in exclude:
            continue
        kept_words.append(word)
        kept_indices.append(i)

    return kept_words, emb_array[kept_indices]

def process_ood_images(model, image_urls, text_emb, class_words, device, image_size=224):
    """Download and process OOD images in one function."""
    print(f"\nDownloading {len(image_urls)} OOD test images...")
    images, names, headers = [], [], {'User-Agent': 'Mozilla/5.0', 'Accept': 'image/*'}
    
    for desc, url in image_urls.items():
        try:
            response = requests.get(url, timeout=30, headers=headers)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                images.append(img.resize((image_size, image_size), Image.BILINEAR))
                names.append(desc)
                print(f"  ✓ Downloaded: {desc}")
        except Exception as e:
            print(f"  ✗ Error downloading {desc}: {str(e)[:50]}")
    
    if not images: return [], [], []
    
    print(f"\n🔬 Processing {len(images)} OOD images...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    model.eval()
    with torch.no_grad():
        ood_emb = []
        for img in images:
            img_tensor = normalize(to_tensor(img).unsqueeze(0).to(device))
            _, visual_proj = model(img_tensor)
            ood_emb.append(F.normalize(visual_proj, p=2, dim=1).cpu().numpy()[0])
    
    ood_emb = np.array(ood_emb)
    predictions = []
    for emb in ood_emb:
        sims = cosine_similarity(emb.reshape(1, -1), text_emb)[0]
        top_5_idx = np.argsort(-sims)[:5]
        predictions.append({
            'words': [class_words[j] for j in top_5_idx],
            'sims': [sims[j] for j in top_5_idx]
        })
    
    return images, names, predictions

# =============================================================================
# FINAL REPORTING
# =============================================================================

def print_final_report(config, test_loss, test_sim, i2t_recalls, t2i_recalls, class_stats,
                      n_train, n_val, n_test, n_classes, batch_size, has_ood, history=None, 
                      best_epoch=None, best_val_sim=None, best_val_loss=None, n_vocab_total=None):
    """Print comprehensive final summary report."""
    n_samples = n_test
    
    print(f"""
📋 Training Configuration:
   ├─ Model: MobileNetV3-Small with projection head
   ├─ Embedding dimension: {config['proj_dim']}
   ├─ Training samples: {n_train:,}, Validation samples: {n_val:,}, Test samples: {n_test:,}
   ├─ Number of training classes: {n_classes}
   {f'├─ Total vocabulary size: {n_vocab_total} words' if n_vocab_total else ''}
   {f'├─ Total epochs trained: {len(history["train_loss"]) if history else 0}, Best epoch: {best_epoch if best_epoch else "N/A"}' if history else ''}
   └─ Early stopping patience: {config['patience']}

🎯 Performance Metrics:
   {f"├─ Best Val Similarity: {best_val_sim:.4f}" if best_val_sim else ""}
   ├─ Test Similarity: {test_sim:.4f}, Test Loss: {test_loss:.4f}
   ├─ Random baseline loss: ~{np.log(batch_size):.2f}
   │
   ├─ Image→Text Recall@1: {i2t_recalls[1]/n_samples*100:.2f}%
   ├─ Image→Text Recall@5: {i2t_recalls[5]/n_samples*100:.2f}%
   ├─ Image→Text Recall@10: {i2t_recalls[10]/n_samples*100:.2f}%
   │
   ├─ Text→Image Recall@1: {t2i_recalls[1]/n_classes*100:.2f}%
   ├─ Text→Image Recall@5: {t2i_recalls[5]/n_classes*100:.2f}%
   └─ Text→Image Recall@10: {t2i_recalls[10]/n_classes*100:.2f}%

📊 Embedding Space Alignment:
   ├─ Mean per-class similarity: {np.mean([s['mean'] for s in class_stats]):.4f} ± {np.std([s['mean'] for s in class_stats]):.4f}
   ├─ Best aligned class: '{class_stats[0]['word']}' ({class_stats[0]['mean']:.4f})
   └─ Worst aligned class: '{class_stats[-1]['word']}' ({class_stats[-1]['mean']:.4f})

💡 Key Insights:
   • The model {'successfully learns' if test_sim > 0.5 else 'attempts to learn'} visual-text alignment
   • {'High' if test_sim > 0.7 else 'Moderate' if test_sim > 0.5 else 'Low'} overall alignment (similarity: {test_sim:.4f})
   • Loss: {test_loss:.2f} vs random baseline ~{np.log(batch_size):.2f}
   • Retrieval performance: {'Good' if i2t_recalls[1]/n_samples > 0.5 else 'Moderate'}
   • Class performance varies (range: {class_stats[-1]['mean']:.4f} to {class_stats[0]['mean']:.4f})
   {f'• OOD predictions use full vocabulary of {n_vocab_total} words' if n_vocab_total else ''}

✅ Model saved to: '{config['save_path']}'
✅ Confusion matrix & retrieval examples saved
{'✅ OOD analysis saved' if has_ood else ''}
""")
