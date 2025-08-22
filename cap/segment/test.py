from utils import *
import matplotlib.pyplot as plt
import json


def strip_markdown_code_fences(text):
    """Remove Markdown code fences from the text."""
    return text.replace("```json", "").replace("```", "").strip()


def visualize_points(points, image, output_path="points.png"):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)

    for point_data in points:
        point = point_data.get("point")
        if point is not None:
            x, y = point
            plt.scatter(x, y, s=100, c="red", edgecolors="white", linewidth=2)

    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


camera_serials = ["317422074281", "317422075456"]

goal = "red cube on the drawer"

point_prompt = f"""
You extract points from images and return only valid JSON that matches the schema.
Coordinate system: pixel coordinates with (0,0) at the top-left. X increases right, Y increases down.
If uncertain or the target is not visible, return {{"point": null, "confidence": 0, "reason": "<short reason>"}}.
Rounding: round to nearest integer. No prose, no extra keys.

here's the goal of where to draw the point: {goal}

Make sure you place your point in the center of the object where there is the most mass of the object visible.
"""

image_np = np.array(capture_rs_rgb(camera_serials[0]), dtype=np.uint8)
image = Image.fromarray(image_np).convert("RGB")

response = call_vlm([image_np], point_prompt, "gpt-4.1-mini", temperature=0.0)
parsed_answer = strip_markdown_code_fences(response)
points_data = json.loads(parsed_answer)

visualize_points([points_data], image_np, output_path="points_visualization.png")
