#!/usr/bin/env python3
"""
Dense And Sparse Correspondence Script

This script demonstrates how DINOv3 features can be used to establish 
correspondences between two objects.
"""

import pickle
import os
import urllib
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

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

# we take DINOv3 ViT-L
MODEL_NAME = MODEL_DINOV3_VITL

model = torch.hub.load(
    repo_or_dir=DINOV3_LOCATION,
    model=MODEL_NAME,
    source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
)
model.cuda()

# Data Loading
image_left_uri = "https://dl.fbaipublicfiles.com/dinov3/notebooks/dense_sparse_matching/image_left.jpg"
mask_left_uri = "https://dl.fbaipublicfiles.com/dinov3/notebooks/dense_sparse_matching/image_left_fg.png"
image_right_uri = "https://dl.fbaipublicfiles.com/dinov3/notebooks/dense_sparse_matching/image_right.jpg"
mask_right_uri = "https://dl.fbaipublicfiles.com/dinov3/notebooks/dense_sparse_matching/image_right_fg.png"

def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f)

image_left = load_image_from_url(image_left_uri)
mask_left = load_image_from_url(mask_left_uri)
image_right = load_image_from_url(image_right_uri)
mask_right = load_image_from_url(mask_right_uri)

# Visualize the two images and the corresponding masks
def visualize_images_and_masks():
    plt.figure(figsize=(16, 8), dpi=300)
    
    for j, (image, mask) in enumerate([(image_left, mask_left), (image_right, mask_right)]):
        foreground = Image.composite(image, mask, mask)
        mask_bg_np = np.copy(np.array(mask))
        mask_bg_np[:, :, 3] = 255 - mask_bg_np[:, :, 3]
        mask_bg = Image.fromarray(mask_bg_np)
        background = Image.composite(image, mask_bg, mask_bg)
        
        data_to_show = [image, mask, foreground, background]
        data_labels = ["Image", "Mask", "Foreground", "Background"]
        
        for i in range(len(data_to_show)):
            plt.subplot(2, len(data_to_show), 4 * j + i + 1)
            plt.imshow(data_to_show[i])
            plt.axis('off')
            plt.title(data_labels[i], fontsize=12)
            
    plt.show()

# Data Transforms
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
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))

# Extracting Features
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

patch_mask_values = []
patch_features = []

def extract_features():
    global patch_mask_values, patch_features
    
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for image, mask in tqdm([(image_left, mask_left), (image_right, mask_right)], desc="Processing images"):
                # processing mask
                mask = mask.split()[-1]
                mask_resized = resize_transform(mask)
                mask_quantized = patch_quant_filter(mask_resized).squeeze().detach().cpu()
                patch_mask_values.append(mask_quantized)
                
                # processing image
                image = image.convert('RGB')
                image_resized = resize_transform(image)
                image_resized = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                image_resized = image_resized.unsqueeze(0).cuda()
                
                feats = model.get_intermediate_layers(image_resized, n=range(n_layers), reshape=True, norm=True)
                dim = feats[-1].shape[1]
                patch_features.append(feats[-1].squeeze().detach().cpu())

# Matching Patches
MASK_FG_THRESHOLD = 0.5

def match_patches():
    global patch_features, locs_2d_left, locs_2d_right, patches_fg_selection
    
    patch_features[0] = F.normalize(patch_features[0], p=2, dim=0)
    patch_features[1] = F.normalize(patch_features[1], p=2, dim=0)
    
    heatmaps = torch.einsum(
        "k f, f h w -> k h w",
        patch_features[0].view(dim, -1).permute(1, 0),
        patch_features[1],
    )
    
    # compute 2D patch locations in the left image
    n_patches_left = patch_features[0].shape[1] * patch_features[0].shape[2]
    patch_indices_left = torch.arange(n_patches_left)
    locs_2d_left = (
        torch.stack(
            (
                patch_indices_left // patch_features[0].shape[2],  # row
                patch_indices_left % patch_features[0].shape[2]    # column
            ),
            dim=-1
        ) + 0.5
    ) * PATCH_SIZE
    
    # compute the corresponding 2D patch locations in the right image
    patch_indices_right = torch.flatten(heatmaps, start_dim=-2).argmax(dim=-1)
    locs_2d_right = (
        torch.stack(
            (
                patch_indices_right // patch_features[1].shape[2],  # row
                patch_indices_right % patch_features[1].shape[2]    # column
            ),
            dim=-1
        ) + 0.5
    ) * PATCH_SIZE
    
    # foreground patches mask in the left image
    patches_left_fg_selection = (patch_mask_values[0].view(-1) > MASK_FG_THRESHOLD)
    # left image patches mask for patches that map to a foreground patch in the right image
    patches_right_fg_selection = (patch_mask_values[1].view(-1)[patch_indices_right] > MASK_FG_THRESHOLD)
    # select foreground left image patches that map to foreground right image patches
    patches_fg_selection = (patches_left_fg_selection * patches_right_fg_selection)
    
    # select (row, col) coordinates for the matched patches
    locs_2d_left_fg = locs_2d_left[patches_fg_selection, :]
    locs_2d_right_fg = locs_2d_right[patches_fg_selection, :]
    
    return locs_2d_left_fg, locs_2d_right_fg

# Dense Correspondence
def dense_correspondence():
    pca = PCA(n_components=3, whiten=True)
    fg_patches_left = patch_features[0].view(dim, -1).permute(1, 0)[patches_fg_selection]
    pca.fit(fg_patches_left)
    
    # get colors for the left image
    h_patches_left = patch_features[0].shape[1]
    w_patches_left = patch_features[0].shape[2]
    x_left = patch_features[0].view(dim, -1).permute(1, 0)
    projected_image_left = torch.from_numpy(
        pca.transform(x_left.numpy())
    ).view(h_patches_left, w_patches_left, 3)
    # multiply by 2.0 and pass through a sigmoid to get vibrant colors 
    projected_image_left = torch.nn.functional.sigmoid(projected_image_left.mul(2.0)).permute(2, 0, 1)
    
    # get colors for the right image
    h_patches_right = patch_features[1].shape[1]
    w_patches_right = patch_features[1].shape[2]
    x_right = patch_features[1].view(dim, -1).permute(1, 0)
    projected_image_right = torch.from_numpy(
        pca.transform(x_right.numpy())
    ).view(h_patches_right, w_patches_right, 3)
    projected_image_right = torch.nn.functional.sigmoid(projected_image_right.mul(2.0)).permute(2, 0, 1)
    
    # apply foreground masks to both visualizations
    projected_image_left *= (patch_mask_values[0] > MASK_FG_THRESHOLD)[None, :, :]
    projected_image_right *= (patch_mask_values[1] > MASK_FG_THRESHOLD)[None, :, :]
    
    plt.figure(figsize=(4, 2), dpi=300)
    plt.subplot(1, 2, 1)
    plt.imshow(projected_image_left.permute(1, 2, 0))
    plt.title("Left Image, Dense Correspondences", fontsize=5)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(projected_image_right.permute(1, 2, 0))
    plt.title("Right Image, Dense Correspondences", fontsize=5)
    plt.axis('off')
    plt.show()
    
    return projected_image_left, projected_image_right

# Sparse Correspondence
# image scale to go from patches to the original image coordinates
scale_left = image_left.height / IMAGE_SIZE
scale_right = image_right.height / IMAGE_SIZE

STRATIFY_DISTANCE_THRESHOLD = 100.0

def compute_distances_l2(X, Y, X_squared_norm, Y_squared_norm):
    distances = -2 * X @ Y.T
    distances.add_(X_squared_norm[:, None]).add_(Y_squared_norm[None, :])
    return distances

def stratify_points(pts_2d: torch.Tensor, threshold: float = 100.0) -> torch.Tensor:
    # pts_2d: [N, 2]
    n = len(pts_2d)
    max_value = threshold + 1
    pts_2d_sq_norms = torch.linalg.vector_norm(pts_2d, dim=1)
    pts_2d_sq_norms.square_()
    distances = compute_distances_l2(pts_2d, pts_2d, pts_2d_sq_norms, pts_2d_sq_norms)
    distances.fill_diagonal_(max_value)
    distances_mask = torch.empty((n, n), dtype=pts_2d.dtype, device=pts_2d.device)
    torch.le(distances, threshold, out=distances_mask)
    ones_vec = torch.ones(n, device=pts_2d.device, dtype=pts_2d.dtype)
    counts_vec = torch.mv(distances_mask, ones_vec)
    indices_mask = np.ones(n)
    
    while torch.any(counts_vec).item():
        index_max = torch.argmax(counts_vec).item()
        indices_mask[index_max] = 0
        distances[index_max, :] = max_value
        distances[:, index_max] = max_value
        torch.le(distances, threshold, out=distances_mask)
        torch.mv(distances_mask, ones_vec, out=counts_vec)
    
    indices_to_exclude = np.nonzero(indices_mask == 0)[0]
    indices_to_keep = np.nonzero(indices_mask > 0)[0]
    return indices_to_exclude, indices_to_keep

def sparse_correspondence(locs_2d_left_fg, locs_2d_right_fg, projected_image_left):
    print(f"Non-stratified points: {tuple(locs_2d_left_fg.shape)}")
    
    indices_to_exclude, indices_to_keep = stratify_points(locs_2d_left_fg * scale_left, STRATIFY_DISTANCE_THRESHOLD**2)
    
    sparse_points_left_yx = locs_2d_left_fg[indices_to_keep, :].cpu().numpy()
    sparse_points_right_yx = locs_2d_right_fg[indices_to_keep, :].cpu().numpy()
    
    print(f"Stratified points: {sparse_points_left_yx.shape}")
    
    # show original left and right images
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    ax1.imshow(image_left)
    ax1.set_axis_off()
    ax2 = fig.add_subplot(122)
    ax2.imshow(image_right)
    ax2.set_axis_off()
    
    for i, (row_left, col_left), (row_right, col_right) in zip(
        indices_to_keep, sparse_points_left_yx, sparse_points_right_yx
    ):
        row_left_orig, col_left_orig = locs_2d_left_fg[i]
        # use the color used for PCA visualization
        color = projected_image_left[
            :,
            int(row_left_orig / PATCH_SIZE),
            int(col_left_orig / PATCH_SIZE)
        ].cpu().numpy()
        con = ConnectionPatch(
            xyA=(col_left * scale_left, row_left * scale_left),
            xyB=(col_right * scale_right, row_right * scale_right),
            coordsA="data",
            coordsB="data",
            axesA=ax1,
            axesB=ax2,
            color=color,
        )
        ax2.add_artist(con)
    
    plt.show()

# Main execution
if __name__ == "__main__":
    # Visualize input images and masks
    visualize_images_and_masks()
    
    # Extract features
    extract_features()
    
    # Match patches
    locs_2d_left_fg, locs_2d_right_fg = match_patches()
    
    # Show dense correspondence
    projected_image_left, projected_image_right = dense_correspondence()
    
    # Show sparse correspondence
    sparse_correspondence(locs_2d_left_fg, locs_2d_right_fg, projected_image_left)
