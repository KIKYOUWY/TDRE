from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(r"G:\UAVdata\CAD-ADD")
MAPPING_PATH = ROOT / "rename_map.csv"


def natural_key(text: str) -> list[object]:
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def sorted_files(directory: Path, suffix: str) -> list[Path]:
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == suffix],
        key=lambda p: natural_key(p.stem),
    )


def update_voc_filename(xml_path: Path, new_filename: str) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename_node = root.find("filename")
    if filename_node is not None:
        filename_node.text = new_filename
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


@dataclass(frozen=True)
class PairDir:
    scene: str
    split: str
    image_dir: Path
    label_dir: Path | None


def build_pairs() -> list[PairDir]:
    pairs: list[PairDir] = []
    scene_map = {
        "Agricultural Detection": "agri",
        "Rescue Detection": "rescue",
        "Waste Detection": "waste",
        "Transport Detection": "transport",
    }
    weather_map = {
        "Clear": "clear",
        "Foggy": "foggy",
        "Dusty": "dusty",
        "Lowlight": "lowlight",
    }

    for scene_dir_name, scene_token in scene_map.items():
        scene_root = ROOT / scene_dir_name
        for weather_dir_name, weather_token in weather_map.items():
            for split in ("train", "test"):
                image_dir = scene_root / weather_dir_name / split
                if image_dir.exists():
                    pairs.append(
                        PairDir(
                            scene=scene_token,
                            split=split,
                            image_dir=image_dir,
                            label_dir=None,
                        )
                    )

        for split in ("train", "test"):
            label_dir = scene_root / "Labels" / split
            if label_dir.exists():
                pairs.append(
                    PairDir(
                        scene=scene_token,
                        split=split,
                        image_dir=label_dir,
                        label_dir=label_dir,
                    )
                )

    real_root = ROOT / "Real Transport Detection"
    pairs.append(
        PairDir(
            scene="real_transport",
            split="test",
            image_dir=real_root / "images",
            label_dir=real_root / "Labels",
        )
    )
    return pairs


def rename_pair(pair: PairDir, rows: list[dict[str, str]]) -> None:
    if pair.label_dir is None:
        images = sorted_files(pair.image_dir, ".jpg")
        new_stem_fn = lambda idx: f"{pair.scene}_{pair.split}_{idx:06d}"
        for idx, img_path in enumerate(images, 1):
            new_stem = new_stem_fn(idx)
            new_img_name = f"{new_stem}.jpg"
            new_img_path = img_path.with_name(new_img_name)
            if new_img_path.exists():
                raise RuntimeError(f"Target already exists: {new_img_path}")
            img_path.rename(new_img_path)
            rows.append(
                {
                    "scene": pair.scene,
                    "weather": "real" if pair.scene == "real_transport" else pair.image_dir.parent.name.lower(),
                    "split": pair.split,
                    "old_image": str(img_path.relative_to(ROOT)),
                    "new_image": str(new_img_path.relative_to(ROOT)),
                    "old_label": "",
                    "new_label": "",
                }
            )
        return

    labels = sorted_files(pair.label_dir, ".xml")
    new_stem_fn = lambda idx: f"{pair.scene}_{pair.split}_{idx:06d}"
    for idx, label_path in enumerate(labels, 1):
        new_stem = new_stem_fn(idx)
        new_label_name = f"{new_stem}.xml"
        new_label_path = label_path.with_name(new_label_name)
        if new_label_path.exists():
            raise RuntimeError(f"Target already exists: {new_label_path}")
        label_path.rename(new_label_path)
        update_voc_filename(new_label_path, f"{new_stem}.jpg")
        rows.append(
            {
                "scene": pair.scene,
                "weather": "",
                "split": pair.split,
                "old_image": "",
                "new_image": "",
                "old_label": str(label_path.relative_to(ROOT)),
                "new_label": str(new_label_path.relative_to(ROOT)),
            }
        )


def main() -> None:
    rows: list[dict[str, str]] = []
    pairs = build_pairs()
    for pair in pairs:
        print(f"Renaming {pair.image_dir}")
        rename_pair(pair, rows)

    with MAPPING_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "weather",
                "split",
                "old_image",
                "new_image",
                "old_label",
                "new_label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Renamed {len(rows)} image/label pairs.")
    print(f"Mapping saved to {MAPPING_PATH}")


if __name__ == "__main__":
    main()
