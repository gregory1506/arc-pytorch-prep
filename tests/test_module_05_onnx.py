"""Module 5: ONNX Export - Tests for model export and optimization."""

import pytest
import torch
from pathlib import Path


class TestONNXExport:
    """Test suite for Module 5: ONNX Export & Optimization"""
    
    def test_onnx_imports(self):
        """TODO 5.0: Verify ONNX module can be imported."""
        try:
            from src.optimization.optimize import export_to_onnx
        except ImportError as e:
            pytest.fail(f"Failed to import export function: {e}")
    
    def test_onnx_export_creates_file(self):
        """TODO 5.2: Test ONNX file is created."""
        pytest.skip("TODO: Implement test after completing TODO 5.2")
        
        from src.optimization.optimize import export_to_onnx
        from src.models.unet import UNet
        import tempfile
        
        model = UNet(n_channels=3, n_classes=1)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 256, 256)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            
            export_to_onnx(model, dummy_input, onnx_path)
            
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0
    
    def test_onnx_file_readable(self):
        """TODO 5.2: Test ONNX file can be read by onnxruntime."""
        pytest.skip("TODO: Implement test after completing TODO 5.2")
        
        import onnxruntime as ort
        from src.optimization.optimize import export_to_onnx
        from src.models.unet import UNet
        import tempfile
        
        model = UNet(n_channels=3, n_classes=1)
        model.eval()
        dummy_input = torch.randn(1, 3, 256, 256)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(model, dummy_input, onnx_path)
            
            # Should be loadable
            session = ort.InferenceSession(str(onnx_path))
            assert session is not None
    
    def test_onnx_output_matches_pytorch(self):
        """TODO 5.3: Test ONNX output matches PyTorch within tolerance."""
        pytest.skip("TODO: Implement test after completing TODO 5.3")
        
        import onnxruntime as ort
        import numpy as np
        from src.optimization.optimize import export_to_onnx
        from src.models.unet import UNet
        import tempfile
        
        model = UNet(n_channels=3, n_classes=1)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 256, 256)
        
        # PyTorch output
        with torch.no_grad():
            pytorch_output = model(dummy_input).numpy()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(model, dummy_input, onnx_path)
            
            # ONNX output
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]
            
            # Compare
            np.testing.assert_allclose(
                pytorch_output, onnx_output,
                rtol=1e-3, atol=1e-5
            )
    
    def test_checkpoint_loading(self):
        """TODO 5.1: Test checkpoint can be loaded before export."""
        pytest.skip("TODO: Implement test after completing TODO 5.1")
        
        from src.optimization.optimize import load_checkpoint
        from src.models.unet import UNet
        import tempfile
        
        # Create and save a checkpoint
        model = UNet(n_channels=3, n_classes=1)
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 10,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint
            model2 = UNet(n_channels=3, n_classes=1)
            load_checkpoint(checkpoint_path, model2)
            
            # Verify weights loaded
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)
    
    def test_onnx_dynamic_axes(self):
        """TODO 5.2: Test ONNX model handles different batch sizes."""
        pytest.skip("TODO: Implement test after completing TODO 5.2")
        
        import onnxruntime as ort
        from src.optimization.optimize import export_to_onnx
        from src.models.unet import UNet
        import tempfile
        
        model = UNet(n_channels=3, n_classes=1)
        model.eval()
        
        # Export with batch size 1
        dummy_input = torch.randn(1, 3, 256, 256)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(model, dummy_input, onnx_path, dynamic_axes=True)
            
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            
            # Test with different batch sizes
            for batch_size in [1, 2, 4]:
                test_input = torch.randn(batch_size, 3, 256, 256)
                output = session.run(None, {input_name: test_input.numpy()})
                assert output[0].shape[0] == batch_size
    
    def test_inference_speed(self):
        """TODO 5.4: Benchmark ONNX vs PyTorch inference speed."""
        pytest.skip("TODO: Implement test after completing TODO 5.4")
        
        import time
        import onnxruntime as ort
        from src.optimization.optimize import export_to_onnx
        from src.models.unet import UNet
        import tempfile
        
        model = UNet(n_channels=3, n_classes=1)
        model.eval()
        dummy_input = torch.randn(1, 3, 256, 256)
        
        # Warm-up
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # PyTorch timing
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = model(dummy_input)
        pytorch_time = time.time() - start
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(model, dummy_input, onnx_path)
            
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            
            # ONNX timing
            start = time.time()
            for _ in range(100):
                _ = session.run(None, {input_name: dummy_input.numpy()})
            onnx_time = time.time() - start
            
            # ONNX should be comparable or faster
            # Allow some tolerance for overhead
            assert onnx_time <= pytorch_time * 1.5
