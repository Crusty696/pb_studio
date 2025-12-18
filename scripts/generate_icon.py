#!/usr/bin/env python3
"""
Generate Placeholder Icon for PB_studio

Creates a simple icon with music note + film strip design.
Requirements: pip install pillow
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_icon(size=256):
    """Create a simple PB_studio icon."""
    # Create image with gradient background
    img = Image.new("RGB", (size, size), color="#1a1a2e")
    draw = ImageDraw.Draw(img)

    # Draw gradient background (simple version)
    for i in range(size):
        color_val = int(26 + (i / size) * 40)  # 26 to 66
        draw.line([(0, i), (size, i)], fill=(color_val, color_val, int(color_val * 1.5)))

    # Draw music note (simplified)
    note_x, note_y = size // 2 - 20, size // 2 - 30
    note_size = int(size * 0.2)

    # Note head (circle)
    draw.ellipse(
        [note_x, note_y + note_size, note_x + note_size, note_y + note_size * 2],
        fill="#16213e",
        outline="#0f3460",
        width=2,
    )

    # Note stem
    draw.rectangle(
        [note_x + note_size - 4, note_y, note_x + note_size, note_y + note_size], fill="#0f3460"
    )

    # Film strip frame (rectangle with perforations)
    frame_padding = int(size * 0.15)
    draw.rectangle(
        [frame_padding, frame_padding, size - frame_padding, size - frame_padding],
        outline="#533483",
        width=3,
    )

    # Film perforations (small rectangles on sides)
    perf_size = 6
    perf_count = 8
    for i in range(perf_count):
        y_pos = frame_padding + (i * (size - 2 * frame_padding) // perf_count)
        # Left side
        draw.rectangle(
            [frame_padding - 2, y_pos, frame_padding + 4, y_pos + perf_size], fill="#533483"
        )
        # Right side
        draw.rectangle(
            [size - frame_padding - 4, y_pos, size - frame_padding + 2, y_pos + perf_size],
            fill="#533483",
        )

    # Add "PB" text at bottom
    try:
        # Try to use a system font
        font_size = int(size * 0.15)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    text = "PB"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (size - text_width) // 2
    text_y = size - frame_padding - text_height - 10

    draw.text((text_x, text_y), text, fill="#e94560", font=font)

    return img


def main():
    """Generate all icon sizes."""
    script_dir = Path(__file__).parent
    icon_dir = script_dir.parent / "assets" / "icons"
    icon_dir.mkdir(parents=True, exist_ok=True)

    print("Generating PB_studio icons...")

    # Generate master 256x256
    master_icon = create_icon(256)
    master_path = icon_dir / "pb_studio.png"
    master_icon.save(master_path)
    print(f"[OK] Created {master_path}")

    # Generate smaller sizes
    sizes = [16, 32, 48, 128]
    for size in sizes:
        icon = create_icon(size)
        icon_path = icon_dir / f"pb_studio_{size}.png"
        icon.save(icon_path)
        print(f"[OK] Created {icon_path}")

    # Generate .ico (Windows multi-size icon)
    try:
        ico_images = [create_icon(s) for s in [16, 32, 48, 256]]
        ico_path = icon_dir / "pb_studio.ico"
        ico_images[0].save(
            ico_path,
            format="ICO",
            sizes=[(16, 16), (32, 32), (48, 48), (256, 256)],
            append_images=ico_images[1:],
        )
        print(f"[OK] Created {ico_path}")
    except Exception as e:
        print(f"[WARN] ICO generation failed (optional): {e}")

    print("\nIcon generation complete!")
    print(f"Icons saved to: {icon_dir}")


if __name__ == "__main__":
    main()
