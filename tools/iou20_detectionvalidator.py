from typing import Any

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator


class IoU20DetectionValidator(DetectionValidator):
    """Detection validator calculating AP at IoU 0.20 through 0.95."""

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks: dict | None = None,
    ) -> None:
        super().__init__(
            dataloader=dataloader,
            save_dir=save_dir,
            args=args,
            _callbacks=_callbacks,
        )

        # IoU thresholds: 0.20, 0.25, ..., 0.95
        self.iouv = torch.arange(0.20, 0.951, 0.05)
        self.niou = self.iouv.numel()