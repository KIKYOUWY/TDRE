from __future__ import annotations

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from PIL import Image


ROOT = Path(r"G:\UAVdata\CAD-ADD\Rescue Detection")
LABEL_ROOT = ROOT / "Labels"
BACKUP_ROOT = ROOT / "Labels_YOLO_backup"
IMAGE_SPLITS = [
    ROOT / "Clear",
    ROOT / "Foggy",
    ROOT / "Dusty",
    ROOT / "Lowlight",
]


def find_image(split: str, stem: str) -> Path:
    for image_root in IMAGE_SPLITS:
        candidate = image_root / split / f"{stem}.jpg"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for {split}/{stem}")


def parse_yolo_lines(txt_path: Path) -> list[tuple[str, float, float, float, float]]:
    objects = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid annotation line in {txt_path}: {line}")
        cls_id = parts[0]
        cx, cy, w, h = map(float, parts[1:])
        objects.append((cls_id, cx, cy, w, h))
    return objects


def yolo_to_voc_box(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    xmin = round((cx - w / 2.0) * img_w)
    ymin = round((cy - h / 2.0) * img_h)
    xmax = round((cx + w / 2.0) * img_w)
    ymax = round((cy + h / 2.0) * img_h)

    xmin = max(0, min(img_w - 1, xmin))
    ymin = max(0, min(img_h - 1, ymin))
    xmax = max(0, min(img_w - 1, xmax))
    ymax = max(0, min(img_h - 1, ymax))

    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    return xmin, ymin, xmax, ymax


def build_xml(image_name: str, size: tuple[int, int], objects: list[tuple[str, float, float, float, float]]) -> ET.ElementTree:
    img_w, img_h = size

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "VOC2007"
    ET.SubElement(root, "filename").text = image_name

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "annotation").text = "PASCAL VOC2007"
    ET.SubElement(source, "database").text = "Unknown"
    ET.SubElement(source, "image").text = "flickr"
    ET.SubElement(source, "flickrid").text = "35435"

    size_node = ET.SubElement(root, "size")
    ET.SubElement(size_node, "width").text = str(img_w)
    ET.SubElement(size_node, "height").text = str(img_h)
    ET.SubElement(size_node, "depth").text = "3"

    ET.SubElement(root, "segmented").text = "0"

    for cls_id, cx, cy, w, h in objects:
        xmin, ymin, xmax, ymax = yolo_to_voc_box(cx, cy, w, h, img_w, img_h)
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = str(cls_id)
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return tree


def convert_split(split: str) -> tuple[int, int]:
    label_dir = LABEL_ROOT / split
    backup_dir = BACKUP_ROOT / split
    backup_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(label_dir.glob("*.txt"))
    converted = 0

    for txt_path in txt_files:
        stem = txt_path.stem
        image_path = find_image(split, stem)
        with Image.open(image_path) as img:
            img_w, img_h = img.size

        objects = parse_yolo_lines(txt_path)
        xml_path = label_dir / f"{stem}.xml"
        build_xml(image_path.name, (img_w, img_h), objects).write(
            xml_path, encoding="utf-8", xml_declaration=True
        )

        shutil.move(str(txt_path), str(backup_dir / txt_path.name))
        converted += 1

    return converted, len(txt_files)


def main() -> None:
    total = 0
    for split in ("train", "test"):
        converted, source_count = convert_split(split)
        total += converted
        print(f"{split}: converted {converted}/{source_count} annotations")

    print(f"done: {total} txt files converted to VOC xml")


if __name__ == "__main__":
    main()
