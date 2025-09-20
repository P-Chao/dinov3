#!/usr/bin/env python3
"""
Foreground Segmentation Script

This script trains a linear foreground segmentation model using DINOv3 features.
"""

import io
import os
import pickle
import tarfile
import urllib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Setup
DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"

if os.getenv("DINOV3_LOCATION") is not None:
    DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
else:
    DINOV3_LOCATION = DINOV3_GITHUB_LOCATION

print(f"DINOv3 location set to {DINOV3_LOCATION}")

# Model
# examples of available DINOv3 models:
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_NAME = MODEL_DINOV3_VITL

model = torch.hub.load(
    repo_or_dir=DINOV3_LOCATION,
    model=MODEL_NAME,
    source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
)
model.cuda()

# Data
IMAGES_URI = "https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/foreground_segmentation_images.tar.gz"
LABELS_URI = "https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/foreground_segmentation_labels.tar.gz"

def load_images_from_remote_tar(tar_uri: str) -> list[Image.Image]:
    """Load images from a remote tar file."""
    images = []
    with urllib.request.urlopen(tar_uri) as f:
        tar = tarfile.open(fileobj=io.BytesIO(f.read()))
        for member in tar.getmembers():
            image_data = tar.extractfile(member)
            image = Image.open(image_data)
            images.append(image)
    return images

def load_image_from_url(url: str) -> Image:
    """Load an image from a URL."""
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")

def visualize_image_mask_pair(image, mask, index=0):
    """Visualize an image and its mask."""
    foreground = Image.composite(image, mask, mask)
    mask_bg_np = np.copy(np.array(mask))
    mask_bg_np[:, :, 3] = 255 - mask_bg_np[:, :, 3]
    mask_bg = Image.fromarray(mask_bg_np)
    background = Image.composite(image, mask_bg, mask_bg)
    
    data_to_show = [image, mask, foreground, background]
    data_labels = ["Image", "Mask", "Foreground", "Background"]
    
    plt.figure(figsize=(16, 4), dpi=300)
    for i in range(len(data_to_show)):
        plt.subplot(1, len(data_to_show), i + 1)
        plt.imshow(data_to_show[i])
        plt.axis('off')
        plt.title(data_labels[i], fontsize=12)
    plt.show()

# Building Per-Patch Label Map
PATCH_SIZE = 16
IMAGE_SIZE = 768

# quantization filter for the given patch size
patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

# image resize transform to dimensions divisible by patch size
def resize_transform(
    mask_image: Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """Resize an image to dimensions divisible by patch size."""
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))

def visualize_mask_quantization(mask):
    """Visualize mask before and after quantization."""
    mask_resized = resize_transform(mask)
    with torch.no_grad():
        mask_quantized = patch_quant_filter(mask_resized).squeeze().detach().cpu()
    
    plt.figure(figsize=(4, 2), dpi=300)
    plt.subplot(1, 2, 1)
    plt.imshow(mask)
    plt.axis('off')
    plt.title(f"Original Mask, Size {mask.size}", fontsize=5)
    plt.subplot(1, 2, 2)
    plt.imshow(mask_quantized)
    plt.axis('off')
    plt.title(f"Quantized Mask, Size {tuple(mask_quantized.shape)}", fontsize=5)
    plt.show()

# Extracting Features and Labels for All the Images
def extract_features_and_labels(images, labels):
    """Extract features and labels for all images."""
    xs = []
    ys = []
    image_index = []
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    MODEL_TO_NUM_LAYERS = {
        MODEL_DINOV3_VITS: 12,
        MODEL_DINOV3_VITSP: 12,
        MODEL_DINOV3_VITB: 12,
        MODEL_DINOV3_VITL: 24,
        MODEL_DINOV3_VITHP: 32,
        MODEL_DINOV3_VIT7B: 40,
    }
    
    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]
    n_images = len(images)
    
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i in tqdm(range(n_images), desc="Processing images"):
                # Loading the ground truth
                mask_i = labels[i].split()[-1]
                mask_i_resized = resize_transform(mask_i)
                mask_i_quantized = patch_quant_filter(mask_i_resized).squeeze().view(-1).detach().cpu()
                ys.append(mask_i_quantized)
                
                # Loading the image data 
                image_i = images[i].convert('RGB')
                image_i_resized = resize_transform(image_i)
                image_i_resized = TF.normalize(image_i_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                image_i_resized = image_i_resized.unsqueeze(0).cuda()
                
                feats = model.get_intermediate_layers(image_i_resized, n=range(n_layers), reshape=True, norm=True)
                dim = feats[-1].shape[1]
                xs.append(feats[-1].squeeze().view(dim, -1).permute(1,0).detach().cpu())
                
                image_index.append(i * torch.ones(ys[-1].shape))
    
    # Concatenate all lists into torch tensors 
    xs = torch.cat(xs)
    ys = torch.cat(ys)
    image_index = torch.cat(image_index)
    
    # keeping only the patches that have clear positive or negative label
    idx = (ys < 0.01) | (ys > 0.99)
    xs = xs[idx]
    ys = ys[idx]
    image_index = image_index[idx]
    
    print("Design matrix of size : ", xs.shape)
    print("Label matrix of size : ", ys.shape)
    
    return xs, ys, image_index

# Training a Classifier and Model Selection
def train_classifier_with_validation(xs, ys, image_index):
    """Train a classifier with leave-one-out validation."""
    n_images = len(torch.unique(image_index))
    cs = np.logspace(-7, 0, 8)
    scores = np.zeros((n_images, len(cs)))
    
    for i in range(n_images):
        # We use leave-one-out so train will be all but image i, val will be image i
        print('validation using image_{:02d}.jpg'.format(i+1))
        train_selection = image_index != float(i)
        fold_x = xs[train_selection].numpy()
        fold_y = (ys[train_selection] > 0).long().numpy()
        val_x = xs[~train_selection].numpy()
        val_y = (ys[~train_selection] > 0).long().numpy()
        
        plt.figure()
        for j, c in enumerate(cs):
            print("training logistic regression with C={:.2e}".format(c))
            clf = LogisticRegression(random_state=0, C=c, max_iter=10000).fit(fold_x, fold_y)
            output = clf.predict_proba(val_x)
            precision, recall, thresholds = precision_recall_curve(val_y, output[:, 1])
            s = average_precision_score(val_y, output[:, 1])
            scores[i, j] = s
            plt.plot(recall, precision, label='C={:.1e} AP={:.1f}'.format(c, s*100))
        
        plt.grid()
        plt.xlabel('recall')
        plt.title('image_{:02d}.jpg'.format(i+1))
        plt.ylabel('precision')
        plt.axis([0, 1, 0, 1])
        plt.legend()
        plt.show()
    
    return scores, cs

def plot_average_map(scores, cs):
    """Plot the average mAP across all validation images."""
    plt.figure(figsize=(3, 2), dpi=300)
    plt.rcParams.update({
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "axes.labelsize": 5,
    })
    plt.plot(scores.mean(axis=0))
    plt.xticks(np.arange(len(cs)), ["{:.0e}".format(c) for c in cs])
    plt.xlabel('data fit C')
    plt.ylabel('average AP')
    plt.grid()
    plt.show()

# Retraining with the optimal regularization
def train_final_classifier(xs, ys, C=0.1):
    """Train the final classifier with the optimal regularization."""
    clf = LogisticRegression(random_state=0, C=C, max_iter=100000, verbose=2).fit(xs.numpy(), (ys > 0).long().numpy())
    return clf

# Test Images and Inference
def inference_on_test_image(clf, test_image_fpath):
    """Perform inference on a test image."""
    test_image = load_image_from_url(test_image_fpath)
    test_image_resized = resize_transform(test_image)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    test_image_normalized = TF.normalize(test_image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    MODEL_TO_NUM_LAYERS = {
        MODEL_DINOV3_VITS: 12,
        MODEL_DINOV3_VITSP: 12,
        MODEL_DINOV3_VITB: 12,
        MODEL_DINOV3_VITL: 24,
        MODEL_DINOV3_VITHP: 32,
        MODEL_DINOV3_VIT7B: 40,
    }
    
    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]
    
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            feats = model.get_intermediate_layers(test_image_normalized.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True)
            x = feats[-1].squeeze().detach().cpu()
            dim = x.shape[0]
            x = x.view(dim, -1).permute(1, 0)
    
    h_patches, w_patches = [int(d / PATCH_SIZE) for d in test_image_resized.shape[1:]]
    
    fg_score = clf.predict_proba(x)[:, 1].reshape(h_patches, w_patches)
    fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))
    
    plt.figure(figsize=(9, 3), dpi=300)
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(test_image_resized.permute(1, 2, 0))
    plt.title('input image')
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(fg_score)
    plt.title('foreground score')
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(fg_score_mf)
    plt.title('+ median filter')
    plt.show()
    
    return fg_score, fg_score_mf

# Saving the Model for Future Use
def save_model(clf, model_path="fg_classifier.pkl"):
    """Save the trained model to a file."""
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")

# Main execution
if __name__ == "__main__":
    # Load data
    images = load_images_from_remote_tar(IMAGES_URI)
    labels = load_images_from_remote_tar(LABELS_URI)
    n_images = len(images)
    assert n_images == len(labels), f"{len(images)=}, {len(labels)=}"
    print(f"Loaded {n_images} images and labels")
    
    # Visualize first image/mask pair
    data_index = 0
    print(f"Showing image / mask at index {data_index}:")
    visualize_image_mask_pair(images[data_index], labels[data_index], data_index)
    
    # Visualize mask quantization
    mask_0 = labels[0].split()[-1]
    visualize_mask_quantization(mask_0)
    
    # Extract features and labels
    xs, ys, image_index = extract_features_and_labels(images, labels)
    
    # Train classifier with validation
    scores, cs = train_classifier_with_validation(xs, ys, image_index)
    
    # Plot average mAP
    plot_average_map(scores, cs)
    
    # Train final classifier
    clf = train_final_classifier(xs, ys, C=0.1)
    
    # Test on a new image
    test_image_fpath = "https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/test_image.jpg"
    fg_score, fg_score_mf = inference_on_test_image(clf, test_image_fpath)
    
    # Save the model
    save_model(clf)
