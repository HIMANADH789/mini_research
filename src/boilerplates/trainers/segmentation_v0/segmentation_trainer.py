import os
import torch

from torch.amp import autocast, GradScaler

from src.boilerplates.model_builder.build import build_model
from src.boilerplates.resolver import build_dataloader
from src.boilerplates.losses.dice_ce import dice_loss

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

        # TRAIN LOADER (split-based)
        self.loader = build_dataloader(config, split="train")

        # OPTIMIZER
        lr = float(config.training.lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # AMP
        self.use_amp = getattr(config.training, "mixed_precision", True)
        self.scaler = GradScaler(enabled=self.use_amp)

    def train(self):
        self.model.train()

        for epoch in range(self.config.training.epochs):
            total_loss = 0

            for img, seg in self.loader:
                img = img.to(self.device, non_blocking=True)
                seg = seg.to(self.device, non_blocking=True)

                with autocast(device_type="cuda", enabled=self.use_amp):
                    out = self.model(img)
                    loss = dice_loss(out, seg)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.loader)

            self.logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

            # SAVE CHECKPOINT
            ckpt_path = os.path.join(
                self.exp_path,
                "checkpoints",
                f"epoch_{epoch}.pth"
            )
            save_model(self.model, ckpt_path)

        self.logger.info("Training Finished")
