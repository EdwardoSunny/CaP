import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from segment.gen_masks import save_multi_mask_visualization
from segment.utils import capture_rs_pc, capture_rs_rgb, get_obs_objects
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModel, AutoProcessor

def extract_segmented_objects(image, masks):
   segmented_objects = []
   for mask in masks:
       # Create 3-channel mask
       mask_3d = np.stack([mask, mask, mask], axis=-1)
       # Apply mask to image (background becomes black)
       segmented = image * mask_3d
       segmented_objects.append(segmented)
   return segmented_objects

# set up camera serials
camera_serials = ["317422074281", "317422075456"]

# Setup CLIP
model = AutoModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# set up SAM and cameras 
sam2_checkpoint = "ckpt/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, apply_postprocessing=False)

# large masks config
# mask_generator = SAM2AutomaticMaskGenerator(
#     model=sam2,
#     points_per_side=16,  # Much fewer points = larger segments
#     points_per_batch=64,
#     pred_iou_thresh=0.6,  # Lower threshold = keep more masks
#     stability_score_thresh=0.85,  # Lower threshold = less strict filtering
#     stability_score_offset=0.5,
#     crop_n_layers=0,  # No cropping for simpler segmentation
#     box_nms_thresh=0.8,  # Higher NMS = remove more overlapping masks
#     crop_n_points_downscale_factor=1,
#     min_mask_region_area=500.0,  # Much larger minimum area
#     use_m2m=False,  # Disable mask-to-mask refinement
# )

# smaller masks config
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=32,  # Reduce from 64 - fewer initial points = larger segments
    points_per_batch=128,
    pred_iou_thresh=0.8,  # Increase threshold - only keep higher quality masks
    stability_score_thresh=0.95,  # Increase - more stable/confident masks only
    stability_score_offset=0.7,
    crop_n_layers=0,  # Reduce cropping layers
    box_nms_thresh=0.5,  # Lower NMS threshold = keep more overlapping masks
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100.0,  # Increase minimum area = filter out tiny segments
    use_m2m=True,
)

frame = capture_rs_rgb(camera_serials[0]).copy()

masks_data = mask_generator.generate(frame)

print(f"Found {len(masks_data)} masks")

# Extract just the mask arrays
masks = np.array([mask_dict['segmentation'] for mask_dict in masks_data])
print(f"Masks shape: {masks.shape}")

segmented_images = extract_segmented_objects(frame, masks)
print(f"Created {len(segmented_images)} segmented images")
print(segmented_images)

# for i, segmented_img in enumerate(segmented_images):
#    Image.fromarray(segmented_img.astype(np.uint8)).save(f"segment_{i:03d}.png")

Image.fromarray(frame.astype(np.uint8)).save("original_frame.png")

def find_best_matching_image(images, text_word):
    if not images:
        return None
        
    # Process inputs
    inputs = processor(text=[text_word], images=images, return_tensors="pt", padding=True)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
    
    # Compute cosine similarities
    similarities = torch.cosine_similarity(image_embeds, text_embeds, dim=1)
    
    # Return image with highest similarity
    best_idx = similarities.argmax().item()
    return images[best_idx]

best_img = find_best_matching_image(segmented_images, "a pink digital camera")
Image.fromarray(best_img.astype(np.uint8)).save("best_matching_image.png")

save_multi_mask_visualization(frame, masks, "output_masks.png")
