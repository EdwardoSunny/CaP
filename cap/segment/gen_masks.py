from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import cv2
from segment.utils import get_obs_objects, capture_rs_rgb
import json

colormap = [
    "blue",
    "orange",
    "green",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "red",
    "lime",
    "indigo",
    "violet",
    "aqua",
    "magenta",
    "coral",
    "gold",
    "tan",
    "skyblue",
]

# load models
model_id = "microsoft/Florence-2-base-ft"
model = (
    AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

checkpoint = "ckpt/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))


def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    # print(f"inputs ---> {inputs}")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    # print(f"generated_ids ---> {generated_ids}")

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # print(f"generated_text ---> {generated_text}")

    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )

    return parsed_answer


def plot_bbox(image, data):
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data["bboxes"], data["labels"]):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
        )
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(
            x1,
            y1,
            label,
            color="white",
            fontsize=8,
            bbox=dict(facecolor="red", alpha=0.5),
        )

    # Remove the axis ticks and labels
    ax.axis("off")

    # Show the plot
    fig.savefig("bbox.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_multi_mask_visualization(
    image: np.ndarray, masks: np.ndarray, filename: str, alpha: float = 0.5
) -> None:
    """
    Visualizes any number of segmentation masks (0, 1, or N) on an image
    with different colors and saves the result.
    """
    # 1. Handle the edge case of zero masks
    if len(masks) == 0:
        print("⚠️ No masks to visualize. Saving original image.")
        cv2.imwrite(filename, image)
        return

    # 2. Normalize the masks array to a consistent (N, H, W) shape
    if masks.ndim == 4:
        # Case: (N, 1, H, W) -> Squeeze to (N, H, W)
        masks = np.squeeze(masks, axis=1)
    elif masks.ndim == 2:
        # Case: (H, W) -> Add a batch dimension to become (1, H, W)
        masks = masks[np.newaxis, :, :]
    # If masks.ndim is already 3 (like the (1, H, W) case), it's already correct.

    visualized_image = image.copy()
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (0, 128, 255),
    ]

    # 3. Loop through the now-consistent (N, H, W) masks array
    for i, float_mask in enumerate(masks):  # float_mask is now always (H, W)
        # Convert float mask to boolean
        mask = float_mask > 0.0

        color = colors[i % len(colors)]

        colored_overlay = np.zeros_like(image, dtype=np.uint8)
        colored_overlay[mask] = color

        visualized_image = cv2.addWeighted(
            visualized_image, 1.0, colored_overlay, alpha, 0
        )

    cv2.imwrite(filename, visualized_image)
    print(f"✅ Image with {len(masks)} masks saved to: {filename}")


def get_masks(frame, object_names):
    image_np = np.array(frame, dtype=np.uint8)
    image = Image.fromarray(image_np).convert("RGB")

    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    results = run_example(task_prompt, image, text_input=object_names)
    # plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
    bbox = results["<CAPTION_TO_PHRASE_GROUNDING>"]
    bboxes = np.array(bbox["bboxes"], dtype=int)

    masks = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_np)

        input_box = np.array(bboxes)

        # 4. Predict the mask using the bounding box
        masks, scores, logits = predictor.predict(
            box=input_box,
            multimask_output=False,  # Set to False to get the single best mask
        )

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 2. Call the function with the BGR NumPy array
    save_multi_mask_visualization(image_bgr, masks, "masked.png")

    return masks, scores
