"""
Plot Molmo point predictions on the original image.

Molmo outputs points in a normalized 1000x1000 coordinate space via XML like:
    <points coords="x1 y1 x2 y2 ...">description</points>

Each pair (x, y) is a point on a [0, 999] grid. To convert to pixel coords:
    pixel_x = (x / 1000) * image_width
    pixel_y = (y / 1000) * image_height
"""

import re
import sys
from PIL import Image, ImageDraw, ImageFont


def parse_points_response(response_text: str) -> list[dict]:
    """Parse Molmo <points> tags from response text.

    Returns a list of dicts with keys: 'label', 'points' (list of (norm_x, norm_y)).
    """
    results = []
    pattern = r'<points coords="([^"]+)">([^<]*)</points>'
    for match in re.finditer(pattern, response_text):
        coords_str = match.group(1)
        label = match.group(2).strip()
        nums = list(map(float, coords_str.split()))
        points = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
        results.append({"label": label, "points": points})
    return results


def plot_points_on_image(
    image_path: str,
    response_text: str,
    output_path: str = "image_with_points.png",
    dot_radius: int = 12,
):
    """Draw predicted points on the image and save."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    parsed = parse_points_response(response_text)
    if not parsed:
        print("No <points> tags found in response.")
        return

    # Colors to cycle through for multiple point groups
    colors = ["red", "lime", "cyan", "magenta", "yellow", "orange"]

    for group_idx, group in enumerate(parsed):
        color = colors[group_idx % len(colors)]
        label = group["label"]
        for norm_x, norm_y in group["points"]:
            px = norm_x / 1000.0 * w
            py = norm_y / 1000.0 * h
            # Draw filled circle
            draw.ellipse(
                [px - dot_radius, py - dot_radius, px + dot_radius, py + dot_radius],
                fill=color,
                outline="white",
                width=2,
            )
            # Draw label next to point
            draw.text(
                (px + dot_radius + 4, py - dot_radius),
                f"{label}\n({norm_x:.0f},{norm_y:.0f})",
                fill=color,
            )
            print(
                f"  Point: norm=({norm_x:.0f}, {norm_y:.0f}) -> pixel=({px:.1f}, {py:.1f})  label=\"{label}\""
            )

    img.save(output_path)
    print(f"\nSaved annotated image to: {output_path}")


if __name__ == "__main__":
    # Default: use the example response from Molmo
    response = (
        sys.argv[1]
        if len(sys.argv) > 1
        else '<points coords="1 1 424 632">where the blue cube is</points>'
    )
    image_path = sys.argv[2] if len(sys.argv) > 2 else "image.png"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "image_with_points.png"

    print(f"Molmo response: {response}")
    print(f"Image: {image_path}\n")
    plot_points_on_image(image_path, response, output_path)
