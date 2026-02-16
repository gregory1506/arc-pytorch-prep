"""Test fixtures and utilities for all modules."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def dummy_image():
    """Create a dummy image tensor for testing."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def dummy_mask():
    """Create a dummy segmentation mask for testing."""
    return torch.randint(0, 2, (1, 256, 256))


@pytest.fixture
def dummy_batch():
    """Create a dummy batch of images and masks."""
    images = torch.randn(4, 3, 256, 256)
    masks = torch.randint(0, 2, (4, 256, 256))
    return images, masks


@pytest.fixture
def device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir
