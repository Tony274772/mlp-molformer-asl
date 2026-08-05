"""
Asymmetric Loss (ASL) for binary classification on raw logits.

Reference: Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification", 2020.

Handles the 9.7% class imbalance by applying different focusing parameters
for positive (incompatible) and negative (compatible) examples:
  - gamma_pos: focusing for positives (small → keeps gradient for easy positives)
  - gamma_neg: focusing for negatives (large → suppresses easy negatives)
  - clip: probability shifting — negatives with p < clip contribute zero loss

Section 7 of ARCHITECTURE.md:
  "ASL alone. Do not combine with SMOTE."
  "ASL operates on raw logits — classifier head outputs a logit, no sigmoid
   applied before the loss."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss operating on raw logits (applies sigmoid internally).

    Default hyper-parameters from ARCHITECTURE.md Section 7:
        gamma_neg=4, gamma_pos=1, clip=0.05

    Grid for tuning:
        gamma_neg ∈ {2, 3, 4}
        gamma_pos ∈ {0, 1}
        clip      ∈ {0, 0.05, 0.1}
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B]  raw logits from the classifier head (no sigmoid)
            targets: [B]  binary labels {0, 1}  (1 = incompatible)

        Returns:
            Scalar mean loss over the batch.
        """
        # Sigmoid → predicted probability of being positive (incompatible)
        p = torch.sigmoid(logits)

        # --- Positive loss term ---
        # L_pos = -(1-p)^gamma_pos * log(p)
        # Use logsigmoid for numerical stability on the log(p) part
        log_p = F.logsigmoid(logits)                          # log(sigmoid(x))
        pos_weight = (1.0 - p).pow(self.gamma_pos)            # focusing weight
        loss_pos = -targets * pos_weight * log_p

        # --- Negative loss term ---
        # Probability shifting: p_shift = max(p - clip, 0)
        # L_neg = -p_shift^gamma_neg * log(1 - p_shift)
        p_shift = (p - self.clip).clamp(min=0.0)
        neg_weight = p_shift.pow(self.gamma_neg)               # focusing weight
        log_1_minus_p_shift = torch.log(1.0 - p_shift + self.eps)
        loss_neg = -(1.0 - targets) * neg_weight * log_1_minus_p_shift

        loss = loss_pos + loss_neg
        return loss.mean()
