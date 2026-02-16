"""Module 3: Training Loop - Tests for training implementation."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path


class TestTrainingLoop:
    """Test suite for Module 3: Training Loop"""
    
    def test_training_imports(self):
        """TODO 3.0: Verify training module can be imported."""
        try:
            from src.training.train import train_epoch, validate
        except ImportError as e:
            pytest.fail(f"Failed to import training functions: {e}")
    
    def test_device_selection(self):
        """TODO 3.1: Test device selection works correctly."""
        pytest.skip("TODO: Implement test after completing TODO 3.1")
        
        # Should select best available device
        import torch
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_model_initialization(self):
        """TODO 3.2: Test model, loss, and optimizer initialization."""
        pytest.skip("TODO: Implement test after completing TODO 3.2")
        
        from src.training.train import setup_training
        from src.models.unet import UNet
        
        model, criterion, optimizer = setup_training(
            model=UNet(n_channels=3, n_classes=1),
            learning_rate=1e-4
        )
        
        assert isinstance(model, UNet)
        assert criterion is not None
        assert optimizer is not None
    
    def test_training_epoch_runs(self):
        """TODO 3.3: Test training loop completes one epoch."""
        pytest.skip("TODO: Implement test after completing TODO 3.3")
        
        from src.training.train import train_epoch
        from src.models.unet import UNet
        
        model = UNet(n_channels=3, n_classes=1)
        
        # Create dummy data
        dummy_loader = torch.utils.data.DataLoader(
            [(torch.randn(3, 256, 256), torch.randint(0, 2, (256, 256))) 
             for _ in range(4)],
            batch_size=2
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Should run without errors
        avg_loss = train_epoch(model, dummy_loader, criterion, optimizer, None, "cpu")
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
    
    def test_validation_loop(self):
        """TODO 3.4: Test validation loop runs correctly."""
        pytest.skip("TODO: Implement test after completing TODO 3.4")
        
        from src.training.train import validate
        from src.models.unet import UNet
        
        model = UNet(n_channels=3, n_classes=1)
        
        dummy_loader = torch.utils.data.DataLoader(
            [(torch.randn(3, 256, 256), torch.randint(0, 2, (256, 256))) 
             for _ in range(4)],
            batch_size=2
        )
        
        criterion = nn.CrossEntropyLoss()
        
        val_loss, metrics = validate(model, dummy_loader, criterion, "cpu")
        
        assert isinstance(val_loss, float)
        assert val_loss >= 0
        assert "iou" in metrics or "dice" in metrics
    
    def test_checkpoint_saving(self):
        """TODO 3.5: Test checkpoint is saved correctly."""
        pytest.skip("TODO: Implement test after completing TODO 3.5")
        
        from src.training.train import save_checkpoint
        from src.models.unet import UNet
        import tempfile
        
        model = UNet(n_channels=3, n_classes=1)
        optimizer = torch.optim.Adam(model.parameters())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=10,
                val_iou=0.85,
                filepath=checkpoint_path
            )
            
            assert checkpoint_path.exists()
            
            # Verify checkpoint can be loaded
            checkpoint = torch.load(checkpoint_path)
            assert "epoch" in checkpoint
            assert checkpoint["epoch"] == 10
    
    def test_checkpoint_loading(self):
        """TODO 3.5: Test checkpoint can be loaded to resume training."""
        pytest.skip("TODO: Implement test after completing TODO 3.5")
        
        from src.training.train import save_checkpoint, load_checkpoint
        from src.models.unet import UNet
        import tempfile
        
        model = UNet(n_channels=3, n_classes=1)
        optimizer = torch.optim.Adam(model.parameters())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            
            # Save
            save_checkpoint(model, optimizer, epoch=5, val_iou=0.8, 
                          filepath=checkpoint_path)
            
            # Load
            model2 = UNet(n_channels=3, n_classes=1)
            optimizer2 = torch.optim.Adam(model2.parameters())
            
            epoch = load_checkpoint(checkpoint_path, model2, optimizer2)
            
            assert epoch == 6  # Should resume from next epoch
    
    def test_amp_training(self):
        """TODO 3.3: Test Automatic Mixed Precision training."""
        pytest.skip("TODO: Implement test after completing TODO 3.3")
        
        from torch.cuda.amp import autocast, GradScaler
        
        # Skip if no GPU
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        scaler = GradScaler()
        
        # Should be able to create scaler
        assert scaler is not None
    
    def test_loss_decreases(self):
        """TODO 3.3: Test that loss decreases during training (overfit test)."""
        pytest.skip("TODO: Implement test after completing TODO 3.3")
        
        from src.training.train import train_epoch
        from src.models.unet import UNet
        
        model = UNet(n_channels=3, n_classes=2)
        
        # Single batch - should overfit
        single_batch = [(torch.randn(3, 256, 256), torch.randint(0, 2, (256, 256)))]
        loader = torch.utils.data.DataLoader(single_batch, batch_size=1)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train for a few iterations
        losses = []
        for _ in range(10):
            loss = train_epoch(model, loader, criterion, optimizer, None, "cpu")
            losses.append(loss)
        
        # Loss should decrease
        assert losses[-1] < losses[0]
    
    def test_logging(self):
        """TODO 3.6: Test training logs are created."""
        pytest.skip("TODO: Implement test after completing TODO 3.6")
        
        # This test verifies logging is set up
        import logging
        
        logger = logging.getLogger("src.training.train")
        assert logger is not None
