"""
Utility helpers — seed setting, parameter counting, download progress.
"""

import torch
import numpy as np
import random
import sys
import urllib.request


def set_seed(seed: int) -> None:
    """Set random seeds for full reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


class _DownloadProgressBar:
    """Simple progress bar for urllib downloads."""

    def __init__(self):
        self._last_pct = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            return
        pct = int(100 * block_num * block_size / total_size)
        pct = min(pct, 100)
        if pct != self._last_pct:
            self._last_pct = pct
            bar = "=" * (pct // 2) + ">" + " " * (50 - pct // 2)
            sys.stdout.write(f"\r  [{bar}] {pct}%")
            sys.stdout.flush()
            if pct == 100:
                sys.stdout.write("\n")


def download_file(url: str, dest_path: str) -> None:
    """Download a file with a progress bar."""
    print(f"  Downloading: {url}")
    print(f"  Destination: {dest_path}")
    urllib.request.urlretrieve(url, dest_path, reporthook=_DownloadProgressBar())
