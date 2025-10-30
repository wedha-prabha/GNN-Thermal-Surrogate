# =================================================================================
# File: src/utils/metrics.py
# Purpose: Custom PyTorch metric functions.
# =================================================================================

import torch

def mean_absolute_error_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Absolute Error (MAE) using PyTorch tensors.

    Args:
        pred: The predicted values (e.g., predicted eta).
        target: The true values (e.g., true eta from CFD).

    Returns:
        A single-item tensor containing the MAE.
    """
    return torch.mean(torch.abs(pred - target))

def mean_squared_error_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Squared Error (MSE) using PyTorch tensors.
    """
    return torch.mean((pred - target) ** 2)

# Other custom metrics (e.g., R-squared, Max Absolute Error) can be added here
# for richer reporting during the project review.