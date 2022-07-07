import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure

# Loss functions

def L1_loss(pred, target, mask):
    masked_pred = pred * mask
    masked_target = target * mask
    l1 = torch.nn.functional.l1_loss(masked_pred, masked_target, reduction='sum') / mask.sum()
    return l1


# Evaluation metrics

def masked_MAE(pred, target, mask):
    pred = mask * pred
    target = mask * target
    mae = torch.nn.functional.l1_loss(pred, target, reduction='sum') / mask.sum()
    return mae

def masked_RMSE(pred, target, mask):
    masked_pred = pred * mask
    masked_target = target * mask

    mse = torch.nn.functional.mse_loss(masked_pred, masked_target, reduction='sum') / mask.sum()
    return mse, torch.sqrt(mse)

def SSIM_score(pred, target, mask):
    # Structural Similarity Index: -1, 1
    complete = mask * pred + (1 - mask) * target

    ssim = StructuralSimilarityIndexMeasure(kernel_size=5)
    score = ssim(complete, target)
    return score

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, depth):
        def second_derivative(x):
            assert x.dim(
            ) == 4, "expected 4-dimensional data, but instead got {}".format(
                x.dim())
            horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :
                                                     -2] - x[:, :, 1:-1, 2:]
            vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:
                                                   -1] - x[:, :, 2:, 1:-1]
            der_2nd = horizontal.abs() + vertical.abs()
            return der_2nd.mean()

        self.loss = second_derivative(depth)
        return self.loss
