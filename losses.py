from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

try:
    import kornia
except ImportError:  # pragma: no cover
    kornia = None


def build_bbox_mask(
    boxes: torch.Tensor,
    orig_size: tuple[int, int],
    target_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Build a binary mask from VOC boxes.

    Args:
        boxes: Tensor of shape [N, 4] in VOC format (xmin, ymin, xmax, ymax).
        orig_size: (width, height) from the source image.
        target_size: (height, width) after resize.
    """
    target_h, target_w = target_size
    orig_w, orig_h = orig_size
    mask = torch.zeros(1, target_h, target_w, device=device)

    if boxes.numel() == 0:
        mask.fill_(1.0)
        return mask

    scale_x = target_w / max(float(orig_w), 1.0)
    scale_y = target_h / max(float(orig_h), 1.0)

    for box in boxes.to(device).float():
        xmin, ymin, xmax, ymax = box.tolist()
        x1 = int(round(xmin * scale_x))
        y1 = int(round(ymin * scale_y))
        x2 = int(round(xmax * scale_x))
        y2 = int(round(ymax * scale_y))

        x1 = max(0, min(target_w - 1, x1))
        y1 = max(0, min(target_h - 1, y1))
        x2 = max(0, min(target_w - 1, x2))
        y2 = max(0, min(target_h - 1, y2))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        mask[:, y1 : y2 + 1, x1 : x2 + 1] = 1.0

    return mask


class ClearSkyObjective(nn.Module):
    """Stage-1 objective: binary classification for clear vs degraded images."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_log: bool = False,
    ):
        loss = self.criterion(logits.view(-1), targets.float().view(-1))
        if return_log:
            return loss, {"loss_clear": loss.detach()}
        return loss


class MultiSpaceRestorationLoss(nn.Module):
    """Stage-2 restoration loss in RGB + HSV + LAB."""

    def __init__(self, w_rgb: float = 1.0, w_hsv: float = 1.0, w_lab: float = 1.0):
        super().__init__()
        self.w_rgb = w_rgb
        self.w_hsv = w_hsv
        self.w_lab = w_lab
        self.l1 = nn.L1Loss()

    def _check_kornia(self) -> None:
        if kornia is None:
            raise ImportError("kornia is required for multi-space losses.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_log: bool = False):
        self._check_kornia()
        pred = pred.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)

        loss_rgb = self.l1(pred, target)
        loss_hsv = self.l1(kornia.color.rgb_to_hsv(pred), kornia.color.rgb_to_hsv(target))
        loss_lab = self.l1(kornia.color.rgb_to_lab(pred), kornia.color.rgb_to_lab(target))

        total = self.w_rgb * loss_rgb + self.w_hsv * loss_hsv + self.w_lab * loss_lab

        if return_log:
            return total, {
                "loss_rgb": loss_rgb.detach(),
                "loss_hsv": loss_hsv.detach(),
                "loss_lab": loss_lab.detach(),
                "loss_restore": total.detach(),
            }
        return total


class WeatherRoutingObjective(nn.Module):
    """Stage-2 gate supervision for fog/dust/low-light routing."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_log: bool = False,
    ):
        valid = targets >= 0
        if valid.any():
            loss = self.criterion(logits[valid], targets[valid].long())
        else:
            loss = logits.sum() * 0.0
        if return_log:
            return loss, {"loss_gate": loss.detach()}
        return loss


class Stage2Objective(nn.Module):
    """Stage-2 total objective: routing CE + multi-space restoration."""

    def __init__(
        self,
        gate_weight: float = 1.0,
        w_rgb: float = 1.0,
        w_hsv: float = 1.0,
        w_lab: float = 1.0,
    ):
        super().__init__()
        self.gate_weight = gate_weight
        self.restore = MultiSpaceRestorationLoss(w_rgb=w_rgb, w_hsv=w_hsv, w_lab=w_lab)
        self.route = WeatherRoutingObjective()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gate_logits: torch.Tensor,
        weather_targets: torch.Tensor,
        return_log: bool = False,
    ):
        restore_loss, restore_log = self.restore(pred, target, return_log=True)
        gate_loss, gate_log = self.route(gate_logits, weather_targets, return_log=True)
        total = restore_loss + self.gate_weight * gate_loss
        if return_log:
            log = {**restore_log, **gate_log}
            log["loss_total"] = total.detach()
            return total, log
        return total


class DetectionMaskedRestorationLoss(nn.Module):
    """Stage-3 object-region masked multi-space loss."""

    def __init__(self, w_rgb: float = 1.0, w_hsv: float = 1.0, w_lab: float = 1.0):
        super().__init__()
        self.restore = MultiSpaceRestorationLoss(w_rgb=w_rgb, w_hsv=w_hsv, w_lab=w_lab)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        boxes_list: list[torch.Tensor],
        orig_sizes: list[tuple[int, int]],
        return_log: bool = False,
    ):
        if pred.dim() != 4:
            raise ValueError("pred must be a 4D tensor")

        masked_pred = []
        masked_target = []
        h, w = pred.shape[-2:]
        for i, (boxes, orig_size) in enumerate(zip(boxes_list, orig_sizes)):
            mask = build_bbox_mask(boxes, orig_size, (h, w), pred.device)
            masked_pred.append(pred[i : i + 1] * mask)
            masked_target.append(target[i : i + 1] * mask)

        masked_pred = torch.cat(masked_pred, dim=0)
        masked_target = torch.cat(masked_target, dim=0)
        return self.restore(masked_pred, masked_target, return_log=return_log)


class Stage3Objective(nn.Module):
    """Stage-3 total objective: masked multi-space enhancement loss."""

    def __init__(self, w_rgb: float = 1.0, w_hsv: float = 1.0, w_lab: float = 1.0):
        super().__init__()
        self.masked_restore = DetectionMaskedRestorationLoss(w_rgb=w_rgb, w_hsv=w_hsv, w_lab=w_lab)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        boxes_list: list[torch.Tensor],
        orig_sizes: list[tuple[int, int]],
        return_log: bool = False,
    ):
        loss, log = self.masked_restore(pred, target, boxes_list, orig_sizes, return_log=True)
        if return_log:
            log["loss_total"] = loss.detach()
            return loss, log
        return loss
