import torch
import torch.nn.functional as F

def dice_loss(pred, target, eps=1e-5):
    pred = torch.softmax(pred, dim=1)
    target = F.one_hot(target, num_classes=4).permute(0,4,1,2,3)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return 1 - (2 * intersection + eps) / (union + eps)