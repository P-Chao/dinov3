#!/usr/bin/env python3
"""
Training script for UPerNet head with frozen DINOv3 ViT backbone

This script implements training of a UPerNet segmentation head
with a frozen DINOv3 ViT backbone.
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Add the DINOv3 repository to the path
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_DIR)

from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
from dinov3.hub.backbones import dinov3_vit7b16

# Setup logging
logger = logging.getLogger("train_upernet_head")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SegmentationDataset(torch.utils.data.Dataset):
    """Custom dataset for segmentation training"""
    
    def __init__(self, image_paths, mask_paths, transform=None, target_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        # Load mask
        mask = Image.open(self.mask_paths[idx])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # Convert mask to tensor
            mask = torch.from_numpy(np.array(mask)).long()
            
        return image, mask


def build_model(cfg):
    """Build the segmentation model with frozen DINOv3 backbone"""
    # Load pretrained DINOv3 backbone
    backbone_model = dinov3_vit7b16(pretrained=True)
    
    # Freeze the backbone
    backbone_model.requires_grad_(False)
    
    # Create adapter for the backbone
    backbone_model = DINOv3_Adapter(
        backbone_model,
        interaction_indexes=[9, 19, 29, 39],
        pretrain_size=512,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        drop_path_rate=0.3,
        init_values=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=True,
    )
    
    # Build segmentation decoder (UPerNet head)
    segmentation_model = build_segmentation_decoder(
        backbone_model=backbone_model,
        backbone_name="dinov3_vit7b16",
        decoder_type="upernet",
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.num_classes,
        autocast_dtype=torch.bfloat16,
    )
    
    return segmentation_model


def build_optimizer(cfg, model):
    """Build optimizer for training"""
    # Only optimize parameters of the segmentation head, not the frozen backbone
    params_groups = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_groups.append(param)
            
    optimizer = torch.optim.AdamW(
        params_groups,
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay
    )
    
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    """Build learning rate scheduler"""
    # Linear warmup + cosine decay
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return float(epoch) / float(max(1, cfg.warmup_epochs))
        progress = float(epoch - cfg.warmup_epochs) / float(max(1, cfg.epochs - cfg.warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def build_data_loader(cfg):
    """Build data loader for training"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
    ])
    
    # Create dataset
    dataset = SegmentationDataset(
        cfg.image_paths,
        cfg.mask_paths,
        transform=transform,
        target_transform=target_transform
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    return data_loader


def compute_loss(pred_masks, target_masks, num_classes):
    """Compute segmentation loss"""
    # Cross-entropy loss for segmentation
    loss = F.cross_entropy(pred_masks, target_masks, ignore_index=255)
    return loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, cfg):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    num_batches = len(data_loader)
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(images)
            # For UperNet, outputs is directly the segmentation map
            pred_masks = outputs
            
            # Compute loss
            loss = compute_loss(pred_masks, targets, cfg.num_classes)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        
        # Log progress
        if batch_idx % cfg.log_interval == 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                f"Loss: {loss.item():.4f}"
            )
    
    # Average losses
    avg_loss = total_loss / num_batches
    
    logger.info(
        f"Epoch {epoch} completed - "
        f"Avg Loss: {avg_loss:.4f}"
    )
    
    return avg_loss


def validate(model, data_loader, device, cfg):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(images)
                pred_masks = outputs
                
                # Compute loss
                loss = compute_loss(pred_masks, targets, cfg.num_classes)
            
            # Accumulate losses
            total_loss += loss.item()
    
    # Average losses
    avg_loss = total_loss / num_batches
    
    logger.info(
        f"Validation - "
        f"Avg Loss: {avg_loss:.4f}"
    )
    
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, cfg, checkpoint_dir):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'cfg': cfg
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"upernet_checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
    return epoch, loss


def main(cfg):
    """Main training function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model
    logger.info("Building model...")
    model = build_model(cfg)
    model.to(device)
    
    # Build optimizer and scheduler
    logger.info("Building optimizer and scheduler...")
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    
    # Build data loaders
    logger.info("Building data loaders...")
    train_loader = build_data_loader(cfg)
    
    # Load checkpoint if specified
    start_epoch = 0
    if cfg.resume_checkpoint:
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, cfg.resume_checkpoint)
        start_epoch += 1
    
    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, cfg.epochs):
        # Train for one epoch
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, cfg
        )
        
        # Step scheduler
        scheduler.step()
        
        # Validate (if validation data is available)
        # For simplicity, we're using the same data for validation
        # In practice, you should have separate validation data
        if cfg.validate:
            val_loss = validate(
                model, train_loader, device, cfg
            )
        else:
            val_loss = train_loss
        
        # Save checkpoint
        if epoch % cfg.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, cfg, cfg.checkpoint_dir
            )
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, cfg, 
                os.path.join(cfg.checkpoint_dir, "upernet_best_model")
            )
    
    logger.info("Training completed!")


def get_args_parser():
    """Get argument parser"""
    parser = argparse.ArgumentParser("Train UPerNet head with frozen DINOv3 backbone")
    
    # Model parameters
    parser.add_argument("--hidden-dim", type=int, default=2048, help="Hidden dimension for UPerNet")
    parser.add_argument("--num-classes", type=int, default=150, help="Number of segmentation classes")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Number of warmup epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--image-size", type=int, default=512, help="Image size for training")
    
    # Data parameters
    parser.add_argument("--image-paths", nargs="+", required=True, help="Paths to training images")
    parser.add_argument("--mask-paths", nargs="+", required=True, help="Paths to training masks")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    
    # Logging and checkpointing
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save-interval", type=int, default=10, help="Checkpoint saving interval")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Path to checkpoint to resume from")
    
    # Validation
    parser.add_argument("--validate", action="store_true", help="Perform validation during training")
    
    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = get_args_parser()
    cfg = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    # Run training
    main(cfg)
