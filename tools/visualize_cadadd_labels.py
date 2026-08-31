from __future__ import annotations

import argparse
import hashlib
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(r"G:\UAVdata\CAD-ADD")
SCENES = [
    ("Agricultural Detection", "agri"),
    ("Rescue Detection", "rescue"),
    ("Waste Detection", "waste"),
    ("Transport Detection", "transport"),
]


def natural_key(text: str) -> list[object]:
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def sorted_images(directory: Path) -> list[Path]:
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: natural_key(p.stem),
    )


def parse_voc(xml_path: Path) -> list[tuple[str, int, int, int, int]]:
    root = ET.parse(xml_path).getroot()
    objects: list[tuple[str, int, int, int, int]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "unknown")
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = int(float(bbox.findtext("xmin", "0")))
        ymin = int(float(bbox.findtext("ymin", "0")))
        xmax = int(float(bbox.findtext("xmax", "0")))
        ymax = int(float(bbox.findtext("ymax", "0")))
        objects.append((name, xmin, ymin, xmax, ymax))
    return objects


def color_for_name(name: str) -> tuple[int, int, int]:
    digest = hashlib.md5(name.encode("utf-8")).digest()
    r, g, b = digest[0], digest[1], digest[2]
    return (80 + r // 2, 80 + g // 2, 80 + b // 2)


def draw_boxes(image: Image.Image, boxes: list[tuple[str, int, int, int, int]]) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for name, xmin, ymin, xmax, ymax in boxes:
        color = color_for_name(name)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=max(2, image.size[0] // 256))
        label = str(name)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        pad = 2
        text_x = max(0, xmin)
        text_y = max(0, ymin - text_h - 2 * pad)
        bg = [text_x, text_y, text_x + text_w + 2 * pad, text_y + text_h + 2 * pad]
        draw.rectangle(bg, fill=color)
        draw.text((text_x + pad, text_y + pad), label, fill="white", font=font)

    return image


def fit_tile(image: Image.Image, tile_size: int) -> Image.Image:
    canvas = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
    fitted = ImageOps.contain(image, (tile_size, tile_size), method=Image.Resampling.LANCZOS)
    x = (tile_size - fitted.width) // 2
    y = (tile_size - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def build_contact_sheet(items: list[tuple[str, Image.Image]], scene_name: str, out_path: Path, tile_size: int) -> None:
    cols = 5
    rows = 2
    pad = 20
    title_h = 40
    header_h = 52
    width = pad + cols * tile_size + (cols - 1) * pad + pad
    height = header_h + rows * tile_size + (rows - 1) * pad + pad

    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((pad, 16), scene_name, fill="black", font=font)

    for idx, (label, image) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = pad + col * (tile_size + pad)
        y = header_h + row * (tile_size + pad)
        sheet.paste(image, (x, y))
        text = label
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        tx = x + 4
        ty = y + tile_size - text_h - 8
        draw.rectangle([tx - 2, ty - 1, tx + text_w + 4, ty + text_h + 2], fill=(255, 255, 255))
        draw.text((tx, ty), text, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def visualize_scene(scene_dir_name: str, scene_token: str, count: int, seed: int, tile_size: int, out_dir: Path) -> Path:
    image_dir = ROOT / scene_dir_name / "Clear" / "test"
    label_dir = ROOT / scene_dir_name / "Labels" / "test"

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    candidates: list[tuple[Path, Path]] = []
    for image_path in sorted_images(image_dir):
        label_path = label_dir / f"{image_path.stem}.xml"
        if label_path.exists():
            candidates.append((image_path, label_path))

    if not candidates:
        raise RuntimeError(f"No matched image/label pairs found for {scene_dir_name}")

    rng = random.Random(seed)
    if len(candidates) <= count:
        selected = candidates
    else:
        selected = rng.sample(candidates, count)
        selected.sort(key=lambda pair: natural_key(pair[0].stem))

    tiles: list[tuple[str, Image.Image]] = []
    for image_path, label_path in selected:
        with Image.open(image_path) as img:
            boxed = draw_boxes(img.copy(), parse_voc(label_path))
        tile = fit_tile(boxed, tile_size)
        title = f"{image_path.stem}"
        tiles.append((title, tile))

    out_path = out_dir / f"{scene_token}_test_labels.png"
    build_contact_sheet(tiles, scene_dir_name, out_path, tile_size)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile_size", type=int, default=384)
    parser.add_argument("--out_dir", type=str, default=str(Path("results") / "label_vis"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    outputs = []
    for scene_dir_name, scene_token in SCENES:
        outputs.append(
            visualize_scene(
                scene_dir_name=scene_dir_name,
                scene_token=scene_token,
                count=args.count,
                seed=args.seed,
                tile_size=args.tile_size,
                out_dir=out_dir,
            )
        )

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
