import torch
import numpy as np
from tqdm import tqdm

from src.boilerplates.losses.metrics import (
    dice_score,
    hausdorff_distance_95,
    sensitivity,
    specificity
)

from src.boilerplates.resolver import build_dataloader
from src.boilerplates.model_builder.build import build_model
from src.utils.experiment_utils.device import get_device


class Evaluator:

    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = get_device(config)

        # ======================
        # MODEL
        # ======================
        self.model = build_model(config).to(self.device)

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # ======================
        # DATA
        # ======================
        self.loader = build_dataloader(config, split="val")

        self.patch_size = config.data.patch_size
        self.stride = [p // 2 for p in self.patch_size]  # 🔥 OVERLAP

        self.num_classes = config.model.out_channels

    # ======================
    # GAUSSIAN WEIGHT MAP
    # ======================
    def get_gaussian_weight(self, shape):
        D, H, W = shape
        z = np.linspace(-1, 1, D)
        y = np.linspace(-1, 1, H)
        x = np.linspace(-1, 1, W)

        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        dist = zz**2 + yy**2 + xx**2

        sigma = 0.5
        weight = np.exp(-dist / (2 * sigma**2))

        return torch.tensor(weight, dtype=torch.float32)

    # ======================
    # SLIDING WINDOW
    # ======================
    def sliding_window_inference(self, volume):

        _, C, D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride

        output = torch.zeros((1, self.num_classes, D, H, W)).to(self.device)
        weight_map = torch.zeros_like(output)

        gaussian = self.get_gaussian_weight((pd, ph, pw)).to(self.device)

        for z in range(0, D, sd):
            for y in range(0, H, sh):
                for x in range(0, W, sw):

                    z1, y1, x1 = z, y, x
                    z2, y2, x2 = min(z + pd, D), min(y + ph, H), min(x + pw, W)

                    patch = volume[:, :, z1:z2, y1:y2, x1:x2]

                    # ======================
                    # PAD PATCH
                    # ======================
                    pad_d = pd - (z2 - z1)
                    pad_h = ph - (y2 - y1)
                    pad_w = pw - (x2 - x1)

                    if pad_d or pad_h or pad_w:
                        patch = torch.nn.functional.pad(
                            patch,
                            (0, pad_w, 0, pad_h, 0, pad_d)
                        )

                    # ======================
                    # FORWARD
                    # ======================
                    pred = self.model(patch)

                    pred = pred[:, :, :z2-z1, :y2-y1, :x2-x1]

                    weight = gaussian[:z2-z1, :y2-y1, :x2-x1]

                    # ======================
                    # ACCUMULATE
                    # ======================
                    output[:, :, z1:z2, y1:y2, x1:x2] += pred * weight
                    weight_map[:, :, z1:z2, y1:y2, x1:x2] += weight

        output = output / (weight_map + 1e-5)

        return output

    # ======================
    # EVALUATE
    # ======================
    def evaluate(self):

        total_dice = []
        total_hd = []
        total_sens = []
        total_spec = []

        with torch.no_grad():
            for img, seg in tqdm(self.loader):

                img = img.to(self.device)
                seg = seg.to(self.device)

                # 🔥 SLIDING WINDOW
                out = self.sliding_window_inference(img)

                pred = torch.argmax(out, dim=1)

                # ======================
                # METRICS
                # ======================
                dices = dice_score(out, seg)
                total_dice.append(dices)

                hd = hausdorff_distance_95(pred[0], seg[0])
                total_hd.append(hd)

                sens = sensitivity(pred, seg)
                spec = specificity(pred, seg)

                total_sens.append(sens)
                total_spec.append(spec)

        avg_dice = torch.tensor(total_dice).mean(dim=0)

        results = {
            "dice_per_class": avg_dice.tolist(),
            "mean_dice": avg_dice.mean().item(),
            "hd95": sum(total_hd) / len(total_hd),
            "sensitivity": sum(total_sens) / len(total_sens),
            "specificity": sum(total_spec) / len(total_spec),
        }

        return results
