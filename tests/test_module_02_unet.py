"""Module 2: UNet Architecture - Tests for model implementation."""

import pytest
import torch
import torch.nn as nn


class TestUNetArchitecture:
    """Test suite for Module 2: UNet Architecture"""
    
    def test_unet_imports(self):
        """TODO 2.0: Verify UNet module can be imported."""
        try:
            from src.models.unet import UNet
        except ImportError as e:
            pytest.fail(f"Failed to import UNet: {e}")
    
    def test_double_conv_block(self):
        """TODO 2.1: Test DoubleConv block architecture."""
        pytest.skip("TODO: Implement test after completing TODO 2.1")
        
        from src.models.unet import DoubleConv
        
        block = DoubleConv(in_channels=64, out_channels=128)
        x = torch.randn(1, 64, 256, 256)
        out = block(x)
        
        # Should maintain spatial dimensions with padding=1
        assert out.shape == (1, 128, 256, 256)
    
    def test_encoder_progression(self):
        """TODO 2.2: Test encoder reduces spatial dims and increases channels."""
        pytest.skip("TODO: Implement test after completing TODO 2.2")
        
        from src.models.unet import UNet
        
        model = UNet(n_channels=3, n_classes=1)
        x = torch.randn(1, 3, 256, 256)
        
        # Test encoder produces features at multiple scales
        # This requires exposing encoder features or testing forward pass
        features = model.encoder(x)
        
        # Check feature progression: 256→128→64→32→16
        assert features[-1].shape[-2:] == (16, 16)
    
    def test_decoder_upsampling(self):
        """TODO 2.3: Test decoder upsamples correctly."""
        pytest.skip("TODO: Implement test after completing TODO 2.3")
        
        from src.models.unet import UNet
        
        model = UNet(n_channels=3, n_classes=1)
        
        # Bottleneck feature
        x = torch.randn(1, 1024, 16, 16)
        
        # Test first decoder upsampling
        up = model.up1(x)
        assert up.shape[-2:] == (32, 32)
    
    def test_skip_connections(self):
        """TODO 2.4: Test skip connections concatenate features."""
        pytest.skip("TODO: Implement test after completing TODO 2.4")
        
        from src.models.unet import UNet
        
        model = UNet(n_channels=3, n_classes=1)
        x = torch.randn(1, 3, 256, 256)
        
        # Forward pass should complete without errors
        output = model(x)
        
        # Skip connections should preserve spatial info
        # This is implicit in successful forward pass
    
    def test_output_shape(self):
        """TODO 2.5: Test output shape matches input spatial dimensions."""
        pytest.skip("TODO: Implement test after completing TODO 2.5")
        
        from src.models.unet import UNet
        
        model = UNet(n_channels=3, n_classes=2)  # 2 classes
        x = torch.randn(1, 3, 256, 256)
        
        output = model(x)
        
        # Output should be (batch, n_classes, H, W)
        assert output.shape == (1, 2, 256, 256)
    
    def test_output_no_activation(self):
        """TODO 2.5: Verify output has no activation (logits)."""
        pytest.skip("TODO: Implement test after completing TODO 2.5")
        
        from src.models.unet import UNet
        
        model = UNet(n_channels=3, n_classes=2)
        x = torch.randn(1, 3, 256, 256)
        
        output = model(x)
        
        # Logits should not be bounded to [0, 1]
        assert output.min() < 0 or output.max() > 1
    
    def test_model_parameters(self):
        """TODO 2.1-2.5: Verify model has learnable parameters."""
        pytest.skip("TODO: Implement after completing UNet")
        
        from src.models.unet import UNet
        
        model = UNet(n_channels=3, n_classes=1)
        params = sum(p.numel() for p in model.parameters())
        
        # UNet should have ~31M parameters
        assert params > 1_000_000
        assert params < 100_000_000
    
    def test_batch_processing(self):
        """TODO 2.5: Test model handles batches correctly."""
        pytest.skip("TODO: Implement test after completing TODO 2.5")
        
        from src.models.unet import UNet
        
        model = UNet(n_channels=3, n_classes=1)
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 256, 256)
            output = model(x)
            assert output.shape[0] == batch_size
    
    def test_different_input_sizes(self):
        """TODO 2.5: Test model with different input sizes."""
        pytest.skip("TODO: Implement test after completing TODO 2.5")
        
        from src.models.unet import UNet
        
        model = UNet(n_channels=3, n_classes=1)
        
        # Test different spatial sizes (must be divisible by 16 for 4 levels)
        for size in [256, 512]:
            x = torch.randn(1, 3, size, size)
            output = model(x)
            assert output.shape[-2:] == (size, size)
    
    def test_no_nans_in_forward(self):
        """TODO 2.1-2.5: Ensure forward pass doesn't produce NaN."""
        pytest.skip("TODO: Implement after completing UNet")
        
        from src.models.unet import UNet
        
        model = UNet(n_channels=3, n_classes=1)
        x = torch.randn(1, 3, 256, 256)
        
        output = model(x)
        assert not torch.isnan(output).any()
