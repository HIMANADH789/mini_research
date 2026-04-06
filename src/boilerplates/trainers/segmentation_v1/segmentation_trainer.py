"""
Segmentation Trainer v1
========================
Improvements over v0:
  - WeightedDice + WeightedCE + FocalLoss (class-imbalance aware)
  - CosineAnnealingLR scheduler (better convergence over 150+ epochs)
  - Per-epoch loss component logging (dice / ce / focal breakdown)
  - Class weights and focal_weight configurable from YAML

Config fields consumed (beyond v0):
  loss:
    class_weights: [0.1, 3.0, 1.0, 2.0]
    focal_weight: 0.5
  training:
    lr_min: 1.0e-6           # cosine annealing floor
"""

import os
import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.boilerplates.model_builder.build import build_model
from src.boilerplates.resolver import build_dataloader
from src.boilerplates.losses.weighted_dice_focal_ce import combined_loss

from src.utils.experiment_utils.device import get_device
from src.utils.experiment_utils.io import save_model


class Trainer:

    def __init__(self, config, exp_path, logger):
        self.config = config
        self.logger = logger
        self.exp_path = exp_path

        # DEVICE
        self.device = get_device(config)

        # MODEL
        self.model = build_model(config).to(self.device)

        # DATA
        self.loader = build_dataloader(config, split="train")

        # LOSS CONFIG
        loss_cfg = getattr(config, "loss", None)
        self.class_weights = getattr(loss_cfg, "class_weights", [0.1, 3.0, 1.0, 2.0])
        self.focal_weight   = float(getattr(loss_cfg, "focal_weight", 0.5))

        # OPTIMIZER
        lr = float(config.training.lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # COSINE ANNEALING LR SCHEDULER
        lr_min = float(getattr(config.training, "lr_min", 1e-6))
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=lr_min,
        )

        # AMP
        self.use_amp = getattr(config.training, "mixed_precision", True)
        self.scaler = GradScaler(enabled=self.use_amp)

    def train(self):
        self.model.train()

        for epoch in range(self.config.training.epochs):

            total_loss  = 0.0
            total_dice  = 0.0
            total_ce    = 0.0
            total_focal = 0.0

            for img, seg in self.loader:
                img = img.to(self.device, non_blocking=True)
                seg = seg.to(self.device, non_blocking=True)

                with autocast(device_type="cuda", enabled=self.use_amp):
                    out = self.model(img)
                    loss, components = combined_loss(
                        out, seg,
                        class_weights=self.class_weights,
                        focal_weight=self.focal_weight,
                    )

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss  += components["loss_total"]
                total_dice  += components["loss_dice"]
                total_ce    += components["loss_ce"]
                total_focal += components["loss_focal"]

            n = len(self.loader)
            current_lr = self.scheduler.get_last_lr()[0]

            self.logger.info(
                f"Epoch {epoch:03d} | "
                f"LR: {current_lr:.2e} | "
                f"Loss: {total_loss/n:.4f} | "
                f"Dice: {total_dice/n:.4f} | "
                f"CE: {total_ce/n:.4f} | "
                f"Focal: {total_focal/n:.4f}"
            )

            # STEP SCHEDULER
            self.scheduler.step()

            # SAVE CHECKPOINT
            ckpt_path = os.path.join(
                self.exp_path,
                "checkpoints",
                f"epoch_{epoch}.pth"
            )
            save_model(self.model, ckpt_path)

        self.logger.info("Training Finished")
