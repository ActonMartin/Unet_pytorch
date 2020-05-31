import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def BCEWithLogitsLoss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)
