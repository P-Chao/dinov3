#!/usr/bin/env python3
"""
Segmentation Tracking Script

This script demonstrates using DINOv3 for video segmentation tracking
using a non-parametric method similar to
"Space-time correspondence as a contrastive random walk" (Jabri et al. 2020).

Given:
- RGB video frames
- Instance segmentation masks for the first frame

We will extract patch features from each frame and use patch similarity
to propagate the ground-truth labels to all frames.
"""

import datetime
import functools
import io
import logging
import math
import os
from pathlib import Path
import tarfile
import time
import urllib

import lovely_tensors
import matplotlib.pyplot as plt
import mediapy as mp
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from torch import Tensor, nn
from tqdm import tqdm

# Setup
DISPLAY_HEIGHT = 200
lovely_tensors.monkey_patch()
torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

# we take DINOv3 ViT-L
MODEL_NAME = MODEL_DINOV3_VITL

model = torch.hub.load(
    repo_or_dir=DINOV3_LOCATION,
    model=MODEL_NAME,
    source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
)
model.to("cuda")
model.eval()

patch_size = model.patch_size
embed_dim = model.embed_dim
print(f"Patch size: {patch_size}")
print(f"Embedding dimension: {embed_dim}")
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

@torch.compile(disable=True)
def forward(
    model: nn.Module,
    img: Tensor,  # [3, H, W] already normalized for the model
) -> Tensor:
    """Forward pass to extract L2-normalized features from a single image."""
    feats = model.get_intermediate_layers(img.unsqueeze(0), n=1, reshape=True)[0]  # [1, D, h, w]
    feats = feats.movedim(-3, -1)  # [1, h, w, D]
    feats = F.normalize(feats, dim=-1, p=2)
    return feats.squeeze(0)  # [h, w, D]

# Data
VIDEO_FRAMES_URI = "https://dl.fbaipublicfiles.com/dinov3/notebooks/segmentation_tracking/video_frames.tar.gz"

def load_video_frames_from_remote_tar(tar_uri: str) -> list[Image.Image]:
    """Load video frames from a remote tar file."""
    images = []
    indices = []
    with urllib.request.urlopen(tar_uri) as f:
        tar = tarfile.open(fileobj=io.BytesIO(f.read()))
        for member in tar.getmembers():
            index_str, _ = os.path.splitext(member.name)
            image_data = tar.extractfile(member)
            image = Image.open(image_data).convert("RGB")
            images.append(image)
            indices.append(int(index_str))
    order = np.argsort(indices)
    return [images[i] for i in order]

def load_image_from_url(url: str) -> Image:
    """Load an image from a URL."""
    with urllib.request.urlopen(url) as f:
        return Image.open(f)

def mask_to_rgb(mask: np.ndarray | Tensor, num_masks: int) -> np.ndarray:
    """Convert a mask to an RGB image."""
    if isinstance(mask, Tensor):
        mask = mask.cpu().numpy()

    # Exclude background
    background = mask == 0
    mask = mask - 1
    num_masks = num_masks - 1

    # Choose palette
    if num_masks <= 10:
        mask_rgb = plt.get_cmap("tab10")(mask)[..., :3]
    elif num_masks <= 20:
        mask_rgb = plt.get_cmap("tab20")(mask)[..., :3]
    else:
        mask_rgb = plt.get_cmap("gist_rainbow")(mask / (num_masks - 1))[..., :3]

    mask_rgb = (mask_rgb * 255).astype(np.uint8)
    mask_rgb[background, :] = 0
    return mask_rgb

def show_sample_frames(frames, num_frames):
    """Show sample frames from the video."""
    num_selected_frames = 4
    selected_frames = np.linspace(0, num_frames - 1, num_selected_frames, dtype=int)
    
    mp.show_images(
        [frames[int(i)] for i in selected_frames],
        titles=[f"Frame {i}" for i in selected_frames],
        height=DISPLAY_HEIGHT,
    )

class ResizeToMultiple(nn.Module):
    """Resize an image to a multiple of a given number."""
    def __init__(self, short_side: int, multiple: int):
        super().__init__()
        self.short_side = short_side
        self.multiple = multiple

    def _round_up(self, side: float) -> int:
        return math.ceil(side / self.multiple) * self.multiple

    def forward(self, img):
        old_width, old_height = TVTF.get_image_size(img)
        if old_width > old_height:
            new_height = self._round_up(self.short_side)
            new_width = self._round_up(old_width * new_height / old_height)
        else:
            new_width = self._round_up(self.short_side)
            new_height = self._round_up(old_height * new_width / old_width)
        return TVTF.resize(img, [new_height, new_width], interpolation=TVT.InterpolationMode.BICUBIC)

# Hyperparameters
SHORT_SIDE = 960
MAX_CONTEXT_LENGTH = 7
NEIGHBORHOOD_SIZE = 12
NEIGHBORHOOD_SHAPE = "circle"
TOPK = 5
TEMPERATURE = 0.2

# Label propagation functions
@torch.compile(disable=True)
def propagate(
    current_features: Tensor,  # [h", w", D], where h=h", w=w", and " stands for current
    context_features: Tensor,  # [t, h, w, D]
    context_probs: Tensor,  # [t, h, w, M]
    neighborhood_mask: Tensor,  # [h", w", h, w]
    topk: int,
    temperature: float,
) -> Tensor:
    """Propagate labels from context frames to the current frame."""
    t, h, w, M = context_probs.shape

    # Compute similarity current -> context
    dot = torch.einsum(
        "ijd, tuvd -> ijtuv",
        current_features,  # [h", w", D]
        context_features,  # [t, h, w, D]
    )  # [h", w", t, h, w]

    # Restrict focus to local neighborhood
    dot = torch.where(
        neighborhood_mask[:, :, None, :, :],  # [h", w", 1, h, w]
        dot,  # [h", w", t, h, w]
        -torch.inf,
    )

    # Select top-k patches inside the neighborhood
    dot = dot.flatten(2, -1).flatten(0, 1)  # [h"w", thw]
    k_th_largest = torch.topk(dot, dim=1, k=topk).values  # [h"w", k]
    dot = torch.where(
        dot >= k_th_largest[:, -1:],  # [h"w", thw]
        dot,  # [h"w", thw]
        -torch.inf,
    )

    # Propagate probabilities from context to current frame
    weights = F.softmax(dot / temperature, dim=1)  # [h"w", thw]
    current_probs = torch.mm(
        weights,  # [h"w", thw]
        context_probs.flatten(0, 2),  # [thw, M]
    )  # [h"w", M]

    # Propagated probs should already sum to 1, but just in case
    current_probs = current_probs / current_probs.sum(dim=1, keepdim=True)  # [h"w", M]

    return current_probs.unflatten(0, (h, w))  # [h", w", M]

@functools.lru_cache()
def make_neighborhood_mask(h: int, w: int, size: float, shape: str) -> Tensor:
    """Create a neighborhood mask for label propagation."""
    ij = torch.stack(
        torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device="cuda"),
            torch.arange(w, dtype=torch.float32, device="cuda"),
            indexing="ij",
        ),
        dim=-1,
    )  # [h, w, 2]
    if shape == "circle":
        ord = 2
    elif shape == "square":
        ord = torch.inf
    else:
        raise ValueError(f"Invalid {shape=}")
    norm = torch.linalg.vector_norm(
        ij[:, :, None, None, :] - ij[None, None, :, :, :],  # [h", w", h, w, 2]
        ord=ord,
        dim=-1,
    )  # [h", w", h, w]
    mask = norm <= size  # [h", w", h, w] bool, True inside, False outside
    return mask

def postprocess_probs(
    probs: Tensor,  # [B, M, H', W']
) -> Tensor:
    """Postprocess probabilities."""
    vmin = probs.flatten(2, 3).min(dim=2).values  # [B, M]
    vmax = probs.flatten(2, 3).max(dim=2).values  # [B, M]
    probs = (probs - vmin[:, :, None, None]) / (vmax[:, :, None, None] - vmin[:, :, None, None])
    probs = torch.nan_to_num(probs, nan=0)
    return probs  # [B, M, H', W']

def process_video(frames, first_mask_np, num_masks, model, transform):
    """Process the video to track segmentation masks."""
    num_frames = len(frames)
    original_width, original_height = frames[0].size
    print(f"Original size: width={original_width}, height={original_height}")

    first_frame = transform(frames[0]).to("cuda")
    print(f"First frame: {first_frame}")

    _, frame_height, frame_width = first_frame.shape  # Abbreviated as [H, W]
    feats_height, feats_width = frame_height // patch_size, frame_width // patch_size  # Abbreviated as [h, w]

    # Label propagation happens at the output resolution of the model,
    # so we downsample the ground-truth masks of the first frame and turn them into a one-hot probability map.
    first_mask = torch.from_numpy(first_mask_np).to("cuda", dtype=torch.long)  # [H', W']
    first_mask = F.interpolate(
        first_mask[None, None, :, :].float(),  # [1, 1, H', W']
        (feats_height, feats_width),
        mode="nearest-exact",
    )[0, 0].long()  # [h, w]
    print(f"First mask:  {first_mask}")

    first_probs = F.one_hot(first_mask, num_masks).float()  # [h, w, M]
    print(f"First probs: {first_probs}")

    mask_height, mask_width = first_mask_np.shape  # Abbreviated at [H', W']
    mask_predictions = torch.zeros([num_frames, mask_height, mask_width], dtype=torch.uint8)  # [T, H', W']
    mask_predictions[0, :, :] = torch.from_numpy(first_mask_np)

    mask_probabilities = torch.zeros([num_frames, num_masks, mask_height, mask_width])  # [T, M, H', W']
    mask_probabilities[0, :, :, :] = F.one_hot(torch.from_numpy(first_mask_np).long(), num_masks).movedim(-1, -3)

    features_queue: list[Tensor] = []
    probs_queue: list[Tensor] = []

    neighborhood_mask = make_neighborhood_mask(
        feats_height,
        feats_width,
        size=NEIGHBORHOOD_SIZE,
        shape=NEIGHBORHOOD_SHAPE,
    )  # [h", w", h, w]

    first_feats = forward(model, first_frame)  # [h, w, D]
    print(f"First feats:   {first_feats.shape}")

    start = time.perf_counter()
    for frame_idx in tqdm(range(1, num_frames), desc="Processing"):
        # Extract features for the current frame
        current_frame_pil = frames[frame_idx]
        current_frame = transform(current_frame_pil).to("cuda")  # [3, H, W]
        torch._dynamo.maybe_mark_dynamic(current_frame, (1, 2))
        current_feats = forward(model, current_frame)  # [h", w", D]

        # Prepare the context, marking the time and mask dimensions as dynamic for torch compile
        context_feats = torch.stack([first_feats, *features_queue], dim=0)  # [1+len(queue), h, w, D]
        context_probs = torch.stack([first_probs, *probs_queue], dim=0)  # [1+len(queue), h, w, M]
        torch._dynamo.maybe_mark_dynamic(context_feats, 0)
        torch._dynamo.maybe_mark_dynamic(context_probs, (0, 3))

        # Propagate segmentation probs from context frames
        current_probs = propagate(
            current_feats,
            context_feats,
            context_probs,
            neighborhood_mask,
            TOPK,
            TEMPERATURE,
        )  # [h", w", M]

        # Update queues with current features and probs
        features_queue.append(current_feats)
        probs_queue.append(current_probs)
        if len(features_queue) > MAX_CONTEXT_LENGTH:
            features_queue.pop(0)
        if len(probs_queue) > MAX_CONTEXT_LENGTH:
            probs_queue.pop(0)

        # Upsample and postprocess segmentation probs, argmax to obtain a prediction
        current_probs = F.interpolate(
            current_probs.movedim(-1, -3)[None, :, :, :],
            size=(mask_height, mask_width),
            mode="nearest",
        )  # [1, M, H', W']
        current_probs = postprocess_probs(current_probs)  # [1, M, H', W']
        current_probs = current_probs.squeeze(0)
        mask_probabilities[frame_idx, :, :, :] = current_probs
        pred = torch.argmax(current_probs, dim=0).to(dtype=torch.uint8)  # [H', W']
        mask_predictions[frame_idx, :, :] = pred  # [H', W']

    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Processing time:    {datetime.timedelta(seconds=round(end - start))}")
    print(f"Mask probabilities: {mask_probabilities}")
    print(f"Mask predictions:   {mask_predictions}")
    
    return mask_predictions, mask_probabilities, selected_frames

def visualize_results(frames, mask_predictions, mask_probabilities, selected_frames, num_masks):
    """Visualize the results."""
    mp.show_images(
        [frames[i].convert("RGB") for i in selected_frames]
        + [mask_to_rgb(mask_predictions[i], num_masks) for i in selected_frames],
        titles=[f"Frame {i}" for i in selected_frames] + [""] * len(selected_frames),
        columns=len(selected_frames),
        height=DISPLAY_HEIGHT,
    )

    mp.show_videos(
        {
            "Input": [np.array(frame) for frame in frames],
            "Pred": mask_to_rgb(mask_predictions, num_masks),
        },
        height=DISPLAY_HEIGHT,
        fps=24,
    )
    
    mp.show_videos(
        {f"Prob {i}": mask_probabilities[:, i].numpy() for i in range(num_masks)},
        height=DISPLAY_HEIGHT,
        fps=24,
    )

# Main execution
if __name__ == "__main__":
    # Load video frames
    frames = load_video_frames_from_remote_tar(VIDEO_FRAMES_URI)
    num_frames = len(frames)
    print(f"Number of frames: {num_frames}")
    
    # Show sample frames
    show_sample_frames(frames, num_frames)
    
    # Load first frame mask
    first_mask_np = np.array(
        load_image_from_url(
            "https://dl.fbaipublicfiles.com/dinov3/notebooks/segmentation_tracking/first_video_frame_mask.png"
        )
    )

    mask_height, mask_width = first_mask_np.shape  # Abbreviated at [H', W']
    print(f"Mask size: {[mask_height, mask_width]}")

    num_masks = int(first_mask_np.max() + 1)  # Abbreviated as M
    print(f"Number of masks: {num_masks}")

    mp.show_images(
        [frames[0], mask_to_rgb(first_mask_np, num_masks)],
        titles=["Frame", "Mask"],
        height=DISPLAY_HEIGHT,
    )
    
    # Setup transforms
    transform = TVT.Compose(
        [
            ResizeToMultiple(short_side=SHORT_SIDE, multiple=patch_size),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    # Process video
    mask_predictions, mask_probabilities, selected_frames = process_video(
        frames, first_mask_np, num_masks, model, transform
    )
    
    # Visualize results
    visualize_results(frames, mask_predictions, mask_probabilities, selected_frames, num_masks)
    
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
