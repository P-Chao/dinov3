import torch
import os
import torchvision
from torchvision.transforms import v2

REPO_DIR = r'D:/code/dinov3'
WEIGHT_DIR = r'D:/code/dinov3/weights'
weight_files = {'vit7b': os.path.join(WEIGHT_DIR, 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth'),
                'vith16_plus': os.path.join(WEIGHT_DIR, 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'),
                'vitl16': os.path.join(WEIGHT_DIR, 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'),
                'vitb16': os.path.join(WEIGHT_DIR, 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),
                'vits16_plus': os.path.join(WEIGHT_DIR, 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'),
                'vits16': os.path.join(WEIGHT_DIR, 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'),
                'convnext_small': os.path.join(WEIGHT_DIR, 'dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth'),
                'detr_head': os.path.join(WEIGHT_DIR, 'dinov3_vit7b16_coco_detr_head-b0235ff7.pth'),
                'dpt_head': os.path.join(WEIGHT_DIR, 'dinov3_vit7b16_synthmix_dpt_head-02040be1.pth'),
                'm2f_head': os.path.join(WEIGHT_DIR, 'dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth'),
                'linear_head': os.path.join(WEIGHT_DIR, 'dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth'),
                'txt_vision_head_encoder': os.path.join(WEIGHT_DIR, 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth')}


import sys
sys.path.append(REPO_DIR)

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
from dinov3.eval.segmentation.inference import make_inference


def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def run_feature_extraction():
    # DINOv3 ViT models pretrained on web images
    # dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=os.path.join(WEIGHT_DIR, 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'))
    # dinov3_vits16plus = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=WEIGHT_DIR)
    # dinov3_vitb16 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=WEIGHT_DIR)
    # dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=WEIGHT_DIR)
    # dinov3_vith16plus = torch.hub.load(REPO_DIR, 'dinov3_vith16plus', source='local', weights=WEIGHT_DIR)
    dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local',
                                    weights=os.path.join(WEIGHT_DIR, 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth'))
    #
    # # DINOv3 ConvNeXt models pretrained on web images
    # dinov3_convnext_tiny = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny', source='local', weights=WEIGHT_DIR)
    # dinov3_convnext_small = torch.hub.load(REPO_DIR, 'dinov3_convnext_small', source='local', weights=WEIGHT_DIR)
    # dinov3_convnext_base = torch.hub.load(REPO_DIR, 'dinov3_convnext_base', source='local', weights=WEIGHT_DIR)
    # dinov3_convnext_large = torch.hub.load(REPO_DIR, 'dinov3_convnext_large', source='local', weights=WEIGHT_DIR)
    #
    # # DINOv3 ViT models pretrained on satellite imagery
    # dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=WEIGHT_DIR)
    # dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=WEIGHT_DIR)
    return dinov3_vit7b16


def run_segmentation():


    segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local", weights=os.path.join(WEIGHT_DIR, 'dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth'),
                               backbone_weights='dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth')
    segmentor = segmentor.to('cuda')

    img_size = 896
    img = get_img()
    transform = make_transform(img_size)
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            batch_img = transform(img)[None].to('cuda')
            pred_vit7b = segmentor(batch_img)  # raw predictions
            # actual segmentation map
            segmentation_map_vit7b = make_inference(
                batch_img,
                segmentor,
                inference_mode="slide",
                decoder_head_type="m2f",
                rescale_to=(img.size[-1], img.size[-2]),
                n_output_channels=150,
                crop_size=(img_size, img_size),
                stride=(img_size, img_size),
                output_activation=partial(torch.nn.functional.softmax, dim=1),
            ).argmax(dim=1, keepdim=True)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(segmentation_map_vit7b[0, 0].cpu(), cmap=colormaps["Spectral"])
    plt.axis("off")

    return segmentor


if __name__ == '__main__':
    #run_feature_extraction()
    run_segmentation()






