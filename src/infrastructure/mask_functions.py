import numpy as np
import torch
from src.infrastructure.constants import GRADIENT_IDENTITY_SCALER

class MaskPruningFunctionConstant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask = mask_param
        mask_thresholded = (mask >= 0).float()

        ctx.save_for_backward(mask, mask_thresholded)
        return mask_thresholded

    @staticmethod
    def backward(ctx, grad_output):
        mask, _ = ctx.saved_tensors
        grad_mask_param = grad_output * GRADIENT_IDENTITY_SCALER

        return grad_mask_param
