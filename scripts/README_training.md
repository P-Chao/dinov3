# Training Segmentation Heads with Frozen DINOv3 ViT Backbone

This guide explains how to train segmentation heads (Mask2Former and UPerNet) with a frozen DINOv3 ViT backbone.

## Overview

This repository provides training scripts for two different segmentation heads:
1. **Mask2Former**: A transformer-based segmentation head
2. **UPerNet**: A pyramid-based segmentation head

Both approaches keep the DINOv3 backbone frozen and only train the segmentation head, allowing you to leverage the powerful pre-trained features from DINOv3 while training only the segmentation head for your specific task.

## Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher
- CUDA-enabled GPU (recommended)
- DINOv3 repository cloned and set up

## Installation

1. Clone the DINOv3 repository:
   ```bash
   git clone https://github.com/facebookresearch/dinov3.git
   cd dinov3
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Prepare your dataset with the following structure:
```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── mask1.png
    ├── mask2.png
    └── ...
```

The masks should be single-channel images where each pixel value represents the class ID. Unlabeled pixels should have a value of 255.

## Training

### Mask2Former Training

#### Basic Usage

Use the provided example script to start training:
```bash
bash demo/run_train_m2f_example.sh
```

#### Custom Training

You can also run the training script directly with custom parameters:
```bash
python demo/train_m2f_head.py \
    --hidden-dim 2048 \
    --num-classes 150 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --epochs 100 \
    --warmup-epochs 10 \
    --batch-size 8 \
    --image-size 512 \
    --image-paths "/path/to/your/images/*.jpg" \
    --mask-paths "/path/to/your/masks/*.png" \
    --num-workers 4 \
    --log-interval 10 \
    --save-interval 10 \
    --checkpoint-dir ./checkpoints \
    --validate
```

### UPerNet Training

#### Basic Usage

Use the provided example script to start training:
```bash
bash demo/run_train_upernet_example.sh
```

#### Custom Training

You can also run the UPerNet training script directly with custom parameters:
```bash
python demo/train_upernet_head.py \
    --hidden-dim 2048 \
    --num-classes 150 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --epochs 100 \
    --warmup-epochs 10 \
    --batch-size 8 \
    --image-size 512 \
    --image-paths "/path/to/your/images/*.jpg" \
    --mask-paths "/path/to/your/masks/*.png" \
    --num-workers 4 \
    --log-interval 10 \
    --save-interval 10 \
    --checkpoint-dir ./checkpoints \
    --validate
```

### Training Parameters

- `--hidden-dim`: Hidden dimension for the segmentation head (default: 2048)
- `--num-classes`: Number of segmentation classes (default: 150)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay (default: 1e-4)
- `--epochs`: Number of training epochs (default: 100)
- `--warmup-epochs`: Number of warmup epochs (default: 10)
- `--batch-size`: Batch size (default: 8)
- `--image-size`: Image size for training (default: 512)
- `--image-paths`: Paths to training images (required)
- `--mask-paths`: Paths to training masks (required)
- `--num-workers`: Number of data loading workers (default: 4)
- `--log-interval`: Logging interval (default: 10)
- `--save-interval`: Checkpoint saving interval (default: 10)
- `--checkpoint-dir`: Checkpoint directory (default: ./checkpoints)
- `--resume-checkpoint`: Path to checkpoint to resume from (default: "")
- `--validate`: Perform validation during training (flag)

## Model Architecture

### Mask2Former

The Mask2Former training script uses the following architecture:

1. **Backbone**: DINOv3 ViT-7B with frozen weights
2. **Adapter**: DINOv3_Adapter to interface between the backbone and decoder
3. **Decoder**: Mask2Former head for segmentation

The backbone is kept frozen during training, and only the adapter and decoder weights are updated.

### UPerNet

The UPerNet training script uses the following architecture:

1. **Backbone**: DINOv3 ViT-7B with frozen weights
2. **Adapter**: DINOv3_Adapter to interface between the backbone and decoder
3. **Decoder**: UPerNet head for segmentation

The UPerNet head consists of:
- Feature Pyramid Network (FPN) for multi-scale feature fusion
- Pyramid Pooling Module (PPM) for global context aggregation
- Segmentation head for final prediction

The backbone is kept frozen during training, and only the adapter and decoder weights are updated.

## Training Process

The training process follows these steps:

1. Load the pre-trained DINOv3 backbone and freeze its weights
2. Create the DINOv3 adapter and Mask2Former decoder
3. Set up the data loader with image and mask preprocessing
4. Initialize the optimizer and learning rate scheduler
5. Train the model for the specified number of epochs
6. Save checkpoints at regular intervals
7. Validate the model if validation is enabled

## Loss Function

### Mask2Former

The Mask2Former training uses a combination of:
1. Cross-entropy loss for mask classification
2. Dice loss for mask prediction

The combined loss is:
```
loss = cls_loss + dice_loss
```

### UPerNet

The UPerNet training uses:
1. Cross-entropy loss for segmentation

The loss function is:
```
loss = cross_entropy_loss
```

## Checkpoints

Checkpoints are saved in the specified checkpoint directory with the following structure:
```
checkpoints/
├── checkpoint_epoch_0.pth
├── checkpoint_epoch_10.pth
├── ...
├── best_model/
│   └── checkpoint_epoch_X.pth
└── upernet_best_model/
    └── upernet_checkpoint_epoch_X.pth
```

The best model is saved separately based on validation loss. For UPerNet, checkpoints are prefixed with "upernet_".

## Inference

After training, you can use the saved checkpoints for inference with the DINOv3 segmentation pipeline.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce the batch size or image size
2. **Data Loading Errors**: Check that your image and mask paths are correct and the files exist
3. **Training Not Converging**: Try adjusting the learning rate or increasing the number of epochs

### Tips

1. Start with a smaller batch size and increase it if memory allows
2. Use the validation flag to monitor training progress
3. Adjust the learning rate based on your dataset size and complexity
4. Monitor the logs to ensure the model is learning properly

## Customization

You can customize the training script by modifying:

1. **Model Architecture**: Adjust the adapter and decoder parameters in `build_model`
2. **Data Loading**: Modify the `SegmentationDataset` class and transforms
3. **Loss Function**: Update the `compute_loss` function
4. **Optimizer**: Change the optimizer parameters in `build_optimizer`
5. **Scheduler**: Modify the learning rate scheduler in `build_lr_scheduler`

## References

- [DINOv3: Scaling Vision Transformers to 1 Billion Images](https://arxiv.org/abs/2304.07193)
- [Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)
