"""Module 1: Data Pipeline - Tests for dataset loading and preprocessing."""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestDataPipeline:
    """Test suite for Module 1: Data Pipeline"""
    
    def test_dataset_imports(self):
        """TODO 1.0: Verify dataset module can be imported."""
        try:
            from src.data.dataset import RemoteSensingDataset
        except ImportError as e:
            pytest.fail(f"Failed to import RemoteSensingDataset: {e}")
    
    def test_sentinel_band_loading(self):
        """TODO 1.1: Test Sentinel-2 band loading functionality."""
        # This test will fail until students implement band loading
        pytest.skip("TODO: Implement test after completing TODO 1.1")
        
        from src.data.dataset import load_sentinel_bands
        
        # Mock test - replace with actual test
        # bands = load_sentinel_bands("path/to/sentinel/scene")
        # assert bands.shape[0] == 13  # 13 Sentinel-2 bands
        # assert bands.dtype == np.float32
    
    def test_tiling_logic(self):
        """TODO 1.2: Test image tiling creates correct number of patches."""
        pytest.skip("TODO: Implement test after completing TODO 1.2")
        
        from src.data.dataset import create_tiles
        
        # Create dummy image
        image = np.random.rand(1024, 1024, 3)
        tiles = create_tiles(image, tile_size=256, overlap=0)
        
        # Should create 16 tiles (4x4 grid)
        assert len(tiles) == 16
        assert all(tile.shape == (256, 256, 3) for tile in tiles)
    
    def test_normalization(self):
        """TODO 1.3: Test per-band normalization."""
        pytest.skip("TODO: Implement test after completing TODO 1.3")
        
        from src.data.dataset import normalize_bands
        
        # Create dummy bands
        bands = np.random.rand(256, 256, 13).astype(np.float32) * 10000
        normalized = normalize_bands(bands)
        
        # Check normalization
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert not np.isnan(normalized).any()
    
    def test_dataset_length(self):
        """TODO 1.4: Test Dataset __len__ method."""
        pytest.skip("TODO: Implement test after completing TODO 1.4")
        
        from src.data.dataset import RemoteSensingDataset
        
        # Create mock dataset
        dataset = RemoteSensingDataset(
            image_dir="dummy/path",
            mask_dir="dummy/path",
            tile_size=256
        )
        
        # Should return number of samples
        assert len(dataset) >= 0
    
    def test_dataset_getitem(self):
        """TODO 1.4: Test Dataset __getitem__ returns correct shapes."""
        pytest.skip("TODO: Implement test after completing TODO 1.4")
        
        from src.data.dataset import RemoteSensingDataset
        
        dataset = RemoteSensingDataset(
            image_dir="dummy/path",
            mask_dir="dummy/path",
            tile_size=256
        )
        
        image, mask = dataset[0]
        
        # Check shapes (C, H, W) for image, (H, W) for mask
        assert image.shape[0] == 3  # RGB bands
        assert image.shape[1] == 256
        assert image.shape[2] == 256
        assert mask.shape == (256, 256)
        
        # Check types
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
    
    def test_dataloader_batching(self):
        """TODO 1.4: Test DataLoader creates correct batches."""
        pytest.skip("TODO: Implement test after completing TODO 1.4")
        
        from torch.utils.data import DataLoader
        from src.data.dataset import RemoteSensingDataset
        
        dataset = RemoteSensingDataset(
            image_dir="dummy/path",
            mask_dir="dummy/path"
        )
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        images, masks = next(iter(loader))
        
        assert images.shape == (4, 3, 256, 256)
        assert masks.shape == (4, 256, 256)
    
    def test_no_nans_in_data(self):
        """TODO 1.1-1.4: Ensure no NaN values in loaded data."""
        pytest.skip("TODO: Implement after completing data pipeline")
        
        from src.data.dataset import RemoteSensingDataset
        
        dataset = RemoteSensingDataset("dummy/path", "dummy/path")
        image, mask = dataset[0]
        
        assert not torch.isnan(image).any()
        assert not torch.isnan(mask).any()
