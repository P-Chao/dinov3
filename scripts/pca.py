#!/usr/bin/env python3
"""
PCA Visualization Script

This script computes the PCA of a foreground object to create colorful visualizations
as shown in the DINOv3 paper.
"""

import pickle
import os
import urllib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from scipy import signal

# Setup
DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"

if os.getenv("DINOV3_LOCATION") is not None:
    DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
else:
    DINOV3_LOCATION = DINOV3_GITHUB_LOCATION

print(f"DINOv3 location set to {DINOV3_LOCATION}")

# Model Loading
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

def load_image_from_url(url: str) -> Image:
    """Load an image from a URL."""
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")

# Loading the Foreground Classifier from the Other Tutorial
def load_foreground_classifier(model_path="fg_classifier.pkl"):
    """Load the foreground classifier trained in the foreground_segmentation tutorial."""
    with open(model_path, 'rb') as file:
        clf = pickle.load(file)
    return clf

# Loading an Image and Applying the Right Transform
PATCH_SIZE = 16
IMAGE_SIZE = 768

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

# Model Forward
MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

def extract_features(image_resized_norm):
    """Extract features from the resized and normalized image."""
    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]
    
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            feats = model.get_intermediate_layers(image_resized_norm.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True)
            x = feats[-1].squeeze().detach().cpu()
            dim = x.shape[0]
            x = x.view(dim, -1).permute(1, 0)
    return x

# Computing Foreground Probability
def compute_foreground_probability(x, clf, h_patches, w_patches):
    """Compute foreground probability using the classifier."""
    fg_score = clf.predict_proba(x)[:, 1].reshape(h_patches, w_patches)
    fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))
    return fg_score, fg_score_mf

def visualize_foreground_probability(image, fg_score_mf):
    """Visualize the image and its foreground probability map."""
    plt.rcParams.update({
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "axes.labelsize": 5,
        "axes.titlesize": 4,
    })
    
    plt.figure(figsize=(4, 2), dpi=300)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Image, Size {image.size}")
    plt.subplot(1, 2, 2)
    plt.imshow(fg_score_mf)
    plt.title(f"Foreground Score, Size {tuple(fg_score_mf.shape)}")
    plt.colorbar()
    plt.axis('off')
    plt.show()

# Extracting Foreground Patches
def extract_foreground_patches(x, fg_score_mf):
    """Extract foreground patches based on classifier output."""
    foreground_selection = fg_score_mf.view(-1) > 0.5
    fg_patches = x[foreground_selection]
    return fg_patches

# Fitting the PCA
def fit_pca(fg_patches, n_components=3):
    """Fit PCA on the foreground patches."""
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(fg_patches)
    return pca

# Applying the PCA, and Masking Background
def apply_pca_and_visualize(x, pca, fg_score_mf, h_patches, w_patches):
    """Apply PCA and visualize the result with background masking."""
    # apply the PCA, and then reshape
    projected_image = torch.from_numpy(pca.transform(x.numpy())).view(h_patches, w_patches, 3)
    
    # multiply by 2.0 and pass through a sigmoid to get vibrant colors 
    projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)
    
    # mask the background using the fg_score_mf
    projected_image *= (fg_score_mf.unsqueeze(0) > 0.5)
    
    # visualize
    plt.figure(dpi=300)
    plt.imshow(projected_image.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    
    return projected_image

# Main execution
if __name__ == "__main__":
    # Load the foreground classifier
    clf = load_foreground_classifier()
    
    # Load an image
    image_uri = "https://dl.fbaipublicfiles.com/dinov3/notebooks/pca/test_image.jpg"
    image = load_image_from_url(image_uri)
    image_resized = resize_transform(image)
    image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    # Extract features
    x = extract_features(image_resized_norm)
    
    # Compute dimensions
    h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized.shape[1:]]
    
    # Compute foreground probability
    fg_score, fg_score_mf = compute_foreground_probability(x, clf, h_patches, w_patches)
    
    # Visualize foreground probability
    visualize_foreground_probability(image, fg_score_mf)
    
    # Extract foreground patches
    fg_patches = extract_foreground_patches(x, fg_score_mf)
    
    # Fit PCA
    pca = fit_pca(fg_patches)
    
    # Apply PCA and visualize
    projected_image = apply_pca_and_visualize(x, pca, fg_score_mf, h_patches, w_patches)
