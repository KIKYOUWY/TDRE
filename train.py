from __future__ import annotations

import argparse
import math
import time
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Sampler

from TDRE import TDRE
from losses import ClearSkyObjective, Stage2Objective, Stage3Objective


SCENE_MAP = {
    "Agricultural Detection": "agri",
    "Rescue Detection": "rescue",
    "Waste Detection": "waste",
    "Transport Detection": "transport",
}

DEGRADED_WEATHERS = ("Foggy", "Dusty", "Lowlight")
WEATHER_TO_EXPERT = {"Foggy": 0, "Dusty": 1, "Lowlight": 2}


def natural_key(text: str):
    import re

    parts = re.split(r"(\d+)", text)
    key = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def sorted_files(directory: Path, suffix: str) -> list[Path]:
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == suffix],
        key=lambda p: natural_key(p.stem),
    )


@lru_cache(maxsize=None)
def parse_voc_cached(xml_path: str) -> tuple[int, int, tuple[tuple[float, float, float, float], ...]]:
    path = Path(xml_path)
    root = ET.parse(path).getroot()
    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"Missing <size> in {path}")

    width = int(size_node.findtext("width", "0"))
    height = int(size_node.findtext("height", "0"))
    boxes: list[tuple[float, float, float, float]] = []

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))
        boxes.append((xmin, ymin, xmax, ymax))

    return width, height, tuple(boxes)


def load_rgb_image(path: Path, img_size: int) -> torch.Tensor:
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    with Image.open(path) as img:
        return transform(img.convert("RGB"))


class Stage1Dataset(Dataset):
    def __init__(self, root: Path, split: str, img_size: int):
        self.samples: list[tuple[Path, int]] = []
        self.condition_indices = {"clear": [], "degraded": []}
        self.img_size = img_size

        for scene_dir_name in SCENE_MAP:
            scene_root = root / scene_dir_name
            clear_dir = scene_root / "Clear" / split
            if not clear_dir.exists():
                continue

            pos = [(p, 1) for p in sorted_files(clear_dir, ".jpg")]
            neg: list[tuple[Path, int]] = []
            for weather in DEGRADED_WEATHERS:
                degraded_dir = scene_root / weather / split
                if degraded_dir.exists():
                    neg.extend((p, 0) for p in sorted_files(degraded_dir, ".jpg"))

            if not pos or not neg:
                continue

            for sample in pos:
                self.condition_indices["clear"].append(len(self.samples))
                self.samples.append(sample)
            for sample in neg:
                self.condition_indices["degraded"].append(len(self.samples))
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = load_rgb_image(img_path, self.img_size)
        return image, torch.tensor(label, dtype=torch.float32)


class Stage23Dataset(Dataset):
    def __init__(self, root: Path, split: str, img_size: int):
        self.samples: list[dict] = []
        self.condition_indices = {
            "clear": [],
            "foggy": [],
            "dusty": [],
            "lowlight": [],
        }
        self.img_size = img_size

        for scene_dir_name, scene_token in SCENE_MAP.items():
            scene_root = root / scene_dir_name
            clear_dir = scene_root / "Clear" / split
            label_dir = scene_root / "Labels" / split
            if not clear_dir.exists() or not label_dir.exists():
                continue

            for clear_path in sorted_files(clear_dir, ".jpg"):
                stem = clear_path.stem
                label_path = label_dir / f"{stem}.xml"
                if not label_path.exists():
                    continue
                self.condition_indices["clear"].append(len(self.samples))
                self.samples.append(
                    {
                        "scene": scene_token,
                        "weather": "clear",
                        "degraded": clear_path,
                        "clear": clear_path,
                        "label": label_path,
                        "expert": -1,
                    }
                )

            for weather in DEGRADED_WEATHERS:
                degraded_dir = scene_root / weather / split
                if not degraded_dir.exists():
                    continue

                for degraded_path in sorted_files(degraded_dir, ".jpg"):
                    stem = degraded_path.stem
                    clear_path = clear_dir / degraded_path.name
                    label_path = label_dir / f"{stem}.xml"
                    if not clear_path.exists() or not label_path.exists():
                        continue
                    expert_label = WEATHER_TO_EXPERT[weather]
                    self.condition_indices[weather.lower()].append(len(self.samples))
                    self.samples.append(
                        {
                            "scene": scene_token,
                            "weather": weather.lower(),
                            "degraded": degraded_path,
                            "clear": clear_path,
                            "label": label_path,
                            "expert": expert_label,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        degraded = load_rgb_image(sample["degraded"], self.img_size)
        clear = load_rgb_image(sample["clear"], self.img_size)
        orig_w, orig_h, boxes = parse_voc_cached(str(sample["label"]))
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        return {
            "degraded": degraded,
            "clear": clear,
            "boxes": boxes_tensor,
            "orig_size": (orig_w, orig_h),
            "weather": torch.tensor(sample["expert"], dtype=torch.long),
        }


def collate_stage23(batch):
    degraded = torch.stack([item["degraded"] for item in batch], dim=0)
    clear = torch.stack([item["clear"] for item in batch], dim=0)
    boxes = [item["boxes"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]
    weather = torch.stack([item["weather"] for item in batch], dim=0)
    return {
        "degraded": degraded,
        "clear": clear,
        "boxes": boxes,
        "orig_sizes": orig_sizes,
        "weather": weather,
    }


class BalancedConditionBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        condition_indices: dict[str, list[int]],
        per_batch: list[tuple[str, int]],
        batches_per_epoch: int | None = None,
        shuffle: bool = True,
    ):
        self.condition_indices = {k: list(v) for k, v in condition_indices.items()}
        self.per_batch = per_batch
        self.shuffle = shuffle

        if batches_per_epoch is None:
            self.batches_per_epoch = max(
                1,
                max(
                    math.ceil(len(self.condition_indices[name]) / max(count, 1))
                    for name, count in self.per_batch
                ),
            )
        else:
            self.batches_per_epoch = batches_per_epoch

        for name, count in self.per_batch:
            if count <= 0:
                raise ValueError(f"Invalid batch count for {name}: {count}")
            if not self.condition_indices.get(name):
                raise ValueError(f"Missing samples for condition: {name}")

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        pools = {}
        pointers = {}
        for name, indices in self.condition_indices.items():
            pools[name] = torch.randperm(len(indices)).tolist() if self.shuffle else list(range(len(indices)))
            pointers[name] = 0

        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for name, count in self.per_batch:
                indices = self.condition_indices[name]
                pool = pools[name]
                ptr = pointers[name]
                for _ in range(count):
                    if ptr >= len(pool):
                        pool = torch.randperm(len(indices)).tolist() if self.shuffle else list(range(len(indices)))
                        ptr = 0
                    batch.append(indices[pool[ptr]])
                    ptr += 1
                pools[name] = pool
                pointers[name] = ptr
            yield batch


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def configure_stage(model: TDRE, stage: int) -> None:
    if stage == 1:
        set_requires_grad(model.clf, True)
        set_requires_grad(model.moe, False)
        set_requires_grad(model.enhancer, False)
        model.clf.train()
        model.moe.eval()
        model.enhancer.eval()
    elif stage == 2:
        set_requires_grad(model.clf, False)
        set_requires_grad(model.moe, True)
        set_requires_grad(model.enhancer, False)
        model.clf.eval()
        model.moe.train()
        model.enhancer.eval()
        model.set_bn_to_eval()
    elif stage == 3:
        set_requires_grad(model.clf, False)
        set_requires_grad(model.moe, False)
        set_requires_grad(model.enhancer, True)
        model.clf.eval()
        model.moe.eval()
        model.enhancer.train()
        model.set_bn_to_eval()
    else:
        raise ValueError(f"Unknown stage: {stage}")


def load_checkpoint(model: TDRE, ckpt_path: str | Path | None, device: torch.device) -> None:
    if not ckpt_path:
        return
    path = Path(ckpt_path)
    if not path.exists():
        return
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)


def save_checkpoint(path: Path, model: TDRE, optimizer: torch.optim.Optimizer, epoch: int, best_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        },
        path,
    )


def make_stage1_loaders(root: Path, split: str, img_size: int, batch_size: int, num_workers: int):
    train_ds = Stage1Dataset(root, split, img_size)
    if batch_size % 2 != 0:
        raise ValueError("Stage 1 batch_size must be even to keep a 1:1 clear/degraded split.")
    sampler = BalancedConditionBatchSampler(
        train_ds.condition_indices,
        per_batch=[("clear", batch_size // 2), ("degraded", batch_size // 2)],
    )
    return DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_stage23_loaders(root: Path, split: str, img_size: int, batch_size: int, num_workers: int):
    train_ds = Stage23Dataset(root, split, img_size)
    if batch_size % 4 != 0:
        raise ValueError("Stage 2/3 batch_size must be divisible by 4 to sample 4 images per condition.")
    per_cond = batch_size // 4
    sampler = BalancedConditionBatchSampler(
        train_ds.condition_indices,
        per_batch=[
            ("clear", per_cond),
            ("foggy", per_cond),
            ("dusty", per_cond),
            ("lowlight", per_cond),
        ],
    )
    return DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_stage23,
    )


def make_stage1_eval_loader(root: Path, split: str, img_size: int, batch_size: int, num_workers: int):
    ds = Stage1Dataset(root, split, img_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def make_stage23_eval_loader(root: Path, split: str, img_size: int, batch_size: int, num_workers: int):
    ds = Stage23Dataset(root, split, img_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_stage23,
    )


def make_optimizer(model: TDRE, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr)


def reduce_log(log: dict[str, torch.Tensor], batch_size: int, totals: dict[str, float], count: int) -> None:
    for key, value in log.items():
        totals[key] = totals.get(key, 0.0) + float(value) * batch_size


def finalize_log(totals: dict[str, float], count: int) -> dict[str, float]:
    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


def format_loss_log(log: dict[str, float]) -> str:
    if not log:
        return ""
    order = [
        "loss_clear",
        "loss_rgb",
        "loss_hsv",
        "loss_lab",
        "loss_restore",
        "loss_gate",
        "loss_total",
    ]
    items = []
    for key in order:
        if key in log:
            items.append(f"{key}={log[key]:.6f}")
    for key in sorted(log):
        if key not in order:
            items.append(f"{key}={log[key]:.6f}")
    return " | " + " ".join(items) if items else ""


def train_stage1(
    model: TDRE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ClearSkyObjective,
    device: torch.device,
):
    model.train()
    model.moe.eval()
    model.enhancer.eval()

    total_loss = 0.0
    total_num = 0
    log_totals: dict[str, float] = {}
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model.clf(images)
        loss, log = criterion(logits, targets, return_log=True)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        reduce_log(log, bs, log_totals, total_num)
        total_num += bs

    return total_loss / max(total_num, 1), finalize_log(log_totals, total_num)


@torch.no_grad()
def eval_stage1(model: TDRE, loader: DataLoader, criterion: ClearSkyObjective, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_num = 0
    log_totals: dict[str, float] = {}
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model.clf(images)
        loss, log = criterion(logits, targets, return_log=True)
        bs = images.size(0)
        total_loss += loss.item() * bs
        reduce_log(log, bs, log_totals, total_num)
        total_num += bs
    return total_loss / max(total_num, 1), finalize_log(log_totals, total_num)


def train_stage2(
    model: TDRE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Stage2Objective,
    device: torch.device,
):
    model.clf.eval()
    model.moe.train()
    model.enhancer.eval()

    total_loss = 0.0
    total_num = 0
    log_totals: dict[str, float] = {}
    for batch in loader:
        degraded = batch["degraded"].to(device)
        clear = batch["clear"].to(device)
        weather = batch["weather"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred, gate_logits = model.moe(degraded)
        loss, log = criterion(pred, clear, gate_logits, weather, return_log=True)
        loss.backward()
        optimizer.step()

        bs = degraded.size(0)
        total_loss += loss.item() * bs
        reduce_log(log, bs, log_totals, total_num)
        total_num += bs

    return total_loss / max(total_num, 1), finalize_log(log_totals, total_num)


@torch.no_grad()
def eval_stage2(model: TDRE, loader: DataLoader, criterion: Stage2Objective, device: torch.device):
    model.clf.eval()
    model.moe.eval()
    model.enhancer.eval()

    total_loss = 0.0
    total_num = 0
    log_totals: dict[str, float] = {}
    for batch in loader:
        degraded = batch["degraded"].to(device)
        clear = batch["clear"].to(device)
        weather = batch["weather"].to(device)
        pred, gate_logits = model.moe(degraded)
        loss, log = criterion(pred, clear, gate_logits, weather, return_log=True)
        bs = degraded.size(0)
        total_loss += loss.item() * bs
        reduce_log(log, bs, log_totals, total_num)
        total_num += bs
    return total_loss / max(total_num, 1), finalize_log(log_totals, total_num)


def train_stage3(
    model: TDRE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Stage3Objective,
    device: torch.device,
):
    model.clf.eval()
    model.moe.eval()
    model.enhancer.train()

    total_loss = 0.0
    total_num = 0
    log_totals: dict[str, float] = {}
    for batch in loader:
        degraded = batch["degraded"].to(device)
        clear = batch["clear"].to(device)
        boxes = batch["boxes"]
        orig_sizes = batch["orig_sizes"]

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            restored, _ = model.moe(degraded)
        enhanced = model.enhancer(restored)
        loss, log = criterion(enhanced, clear, boxes, orig_sizes, return_log=True)
        loss.backward()
        optimizer.step()

        bs = degraded.size(0)
        total_loss += loss.item() * bs
        reduce_log(log, bs, log_totals, total_num)
        total_num += bs

    return total_loss / max(total_num, 1), finalize_log(log_totals, total_num)


@torch.no_grad()
def eval_stage3(model: TDRE, loader: DataLoader, criterion: Stage3Objective, device: torch.device):
    model.clf.eval()
    model.moe.eval()
    model.enhancer.eval()

    total_loss = 0.0
    total_num = 0
    log_totals: dict[str, float] = {}
    for batch in loader:
        degraded = batch["degraded"].to(device)
        clear = batch["clear"].to(device)
        boxes = batch["boxes"]
        orig_sizes = batch["orig_sizes"]
        restored, _ = model.moe(degraded)
        enhanced = model.enhancer(restored)
        loss, log = criterion(enhanced, clear, boxes, orig_sizes, return_log=True)
        bs = degraded.size(0)
        total_loss += loss.item() * bs
        reduce_log(log, bs, log_totals, total_num)
        total_num += bs
    return total_loss / max(total_num, 1), finalize_log(log_totals, total_num)


def run_stage(
    stage: int,
    model: TDRE,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    args,
    init_ckpt: str | None = None,
) -> Path:
    configure_stage(model, stage)
    load_checkpoint(model, init_ckpt, device)

    if stage == 1:
        criterion = ClearSkyObjective()
    elif stage == 2:
        criterion = Stage2Objective(
            gate_weight=args.gate_weight,
            w_rgb=args.w_rgb,
            w_hsv=args.w_hsv,
            w_lab=args.w_lab,
        )
    elif stage == 3:
        criterion = Stage3Objective(w_rgb=args.w_rgb, w_hsv=args.w_hsv, w_lab=args.w_lab)
    else:
        raise ValueError(stage)

    optimizer = make_optimizer(model, args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    best_loss = float("inf")
    stage_ckpt = Path(args.save_dir) / f"stage{stage}_best.pth"

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        if stage == 1:
            train_loss, train_log = train_stage1(model, train_loader, optimizer, criterion, device)
            if val_loader:
                val_loss, val_log = eval_stage1(model, val_loader, criterion, device)
            else:
                val_loss, val_log = train_loss, train_log
        elif stage == 2:
            train_loss, train_log = train_stage2(model, train_loader, optimizer, criterion, device)
            if val_loader:
                val_loss, val_log = eval_stage2(model, val_loader, criterion, device)
            else:
                val_loss, val_log = train_loss, train_log
        else:
            train_loss, train_log = train_stage3(model, train_loader, optimizer, criterion, device)
            if val_loader:
                val_loss, val_log = eval_stage3(model, val_loader, criterion, device)
            else:
                val_loss, val_log = train_loss, train_log

        scheduler.step()
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[stage {stage}] epoch {epoch:03d}/{args.epochs} "
            f"lr={lr:.2e} train={train_loss:.6f} val={val_loss:.6f} time={elapsed:.1f}s"
            f"{format_loss_log(train_log)}"
        )
        if val_loader:
            print(f"           val{format_loss_log(val_log)}")

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(stage_ckpt, model, optimizer, epoch, best_loss)

    return stage_ckpt


def build_loaders(root: Path, stage: int, split: str, eval_split: str, img_size: int, batch_size: int, num_workers: int):
    if stage == 1:
        train_loader = make_stage1_loaders(root, split, img_size, batch_size, num_workers)
        val_loader = make_stage1_eval_loader(root, eval_split, img_size, batch_size, num_workers)
    else:
        train_loader = make_stage23_loaders(root, split, img_size, batch_size, num_workers)
        val_loader = make_stage23_eval_loader(root, eval_split, img_size, batch_size, num_workers)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=r"datasets\CAD-ADD")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epochs1", type=int, default=20)
    parser.add_argument("--epochs2", type=int, default=20)
    parser.add_argument("--epochs3", type=int, default=20)
    parser.add_argument("--batch_size1", type=int, default=12)
    parser.add_argument("--batch_size2", type=int, default=16)
    parser.add_argument("--batch_size3", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min_lr", type=float, default=2e-5)
    parser.add_argument("--gate_weight", type=float, default=1.0)
    parser.add_argument("--w_rgb", type=float, default=1.0)
    parser.add_argument("--w_hsv", type=float, default=1.0)
    parser.add_argument("--w_lab", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weights", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TDRE(n_expert=3, top_k=3, inter_ch=3).to(device)
    if args.weights:
        load_checkpoint(model, args.weights, device)

    root = Path(args.data_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stages = [1, 2, 3] if args.stage == "all" else [int(args.stage)]
    prev_ckpt = args.weights if args.weights else None

    for stage in stages:
        if stage == 1:
            epochs = args.epochs1 if args.stage == "all" else args.epochs
            batch_size = args.batch_size1
        elif stage == 2:
            epochs = args.epochs2 if args.stage == "all" else args.epochs
            batch_size = args.batch_size2
        else:
            epochs = args.epochs3 if args.stage == "all" else args.epochs
            batch_size = args.batch_size3

        stage_args = argparse.Namespace(**vars(args))
        stage_args.epochs = epochs
        stage_args.save_dir = str(save_dir)
        train_loader, val_loader = build_loaders(
            root=root,
            stage=stage,
            split=args.split,
            eval_split=args.eval_split,
            img_size=args.img_size,
            batch_size=batch_size,
            num_workers=args.num_workers,
        )
        prev_ckpt = run_stage(stage, model, train_loader, val_loader, device, stage_args, prev_ckpt)


if __name__ == "__main__":
    main()
