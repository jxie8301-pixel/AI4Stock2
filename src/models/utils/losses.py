"""Custom PyTorch loss functions for quantitative finance."""

import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    """Pearson Correlation Loss.
    
    Instead of minimizing the absolute error (MSE), this loss maximizes the 
    linear correlation between predictions and targets. This is equivalent to
    optimizing for the Information Coefficient (IC).
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, label, weight=None):
        # Create a mask to ignore NaNs in labels
        mask = ~torch.isnan(label)
        
        # If weight is provided, we mask it too. For simplicity in correlation, 
        # we often use unweighted correlation unless specific market-cap weighting is needed.
        # Here we do a standard unweighted Pearson correlation on valid data points.
        p = pred[mask]
        t = label[mask]

        # If there are fewer than 2 valid points, we can't compute variance
        if len(p) < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Center the predictions and targets
        p_mean = torch.mean(p)
        t_mean = torch.mean(t)
        
        p_centered = p - p_mean
        t_centered = t - t_mean

        # Compute covariance and variances
        cov = torch.sum(p_centered * t_centered)
        p_var = torch.sum(p_centered ** 2)
        t_var = torch.sum(t_centered ** 2)

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-8
        
        # Pearson correlation coefficient
        corr = cov / (torch.sqrt(p_var * t_var) + epsilon)

        # We want to MAXIMIZE correlation, so we MINIMIZE negative correlation
        return -corr


class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient (CCC) Loss.
    
    Similar to Pearson, but also penalizes predictions that shift away from the 
    target mean or variance. It's often more robust than pure Pearson.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, label, weight=None):
        mask = ~torch.isnan(label)
        p = pred[mask]
        t = label[mask]

        if len(p) < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        p_mean = torch.mean(p)
        t_mean = torch.mean(t)
        
        p_centered = p - p_mean
        t_centered = t - t_mean

        cov = torch.sum(p_centered * t_centered) / len(p)
        p_var = torch.sum(p_centered ** 2) / len(p)
        t_var = torch.sum(t_centered ** 2) / len(t)
        
        epsilon = 1e-8
        
        # CCC formula
        ccc = (2 * cov) / (p_var + t_var + (p_mean - t_mean)**2 + epsilon)
        
        return -ccc
