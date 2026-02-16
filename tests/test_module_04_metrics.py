"""Module 4: Evaluation Metrics - Tests for IoU and Dice coefficient."""

import pytest
import torch
import numpy as np


class TestMetrics:
    """Test suite for Module 4: Evaluation Metrics"""
    
    def test_metrics_imports(self):
        """TODO 4.0: Verify metrics module can be imported."""
        try:
            from src.evaluation.metrics import iou_score, dice_score
        except ImportError as e:
            pytest.fail(f"Failed to import metrics: {e}")
    
    def test_iou_perfect_overlap(self):
        """TODO 4.1: Test IoU = 1.0 for perfect overlap."""
        pytest.skip("TODO: Implement test after completing TODO 4.1")
        
        from src.evaluation.metrics import iou_score
        
        pred = torch.ones(1, 256, 256)
        target = torch.ones(1, 256, 256)
        
        iou = iou_score(pred, target)
        
        assert abs(iou - 1.0) < 1e-6
    
    def test_iou_no_overlap(self):
        """TODO 4.1: Test IoU = 0.0 for no overlap."""
        pytest.skip("TODO: Implement test after completing TODO 4.1")
        
        from src.evaluation.metrics import iou_score
        
        pred = torch.ones(1, 256, 256)
        target = torch.zeros(1, 256, 256)
        
        iou = iou_score(pred, target)
        
        assert abs(iou - 0.0) < 1e-6
    
    def test_iou_partial_overlap(self):
        """TODO 4.1: Test IoU for partial overlap."""
        pytest.skip("TODO: Implement test after completing TODO 4.1")
        
        from src.evaluation.metrics import iou_score
        
        # Create arrays with known overlap
        pred = torch.zeros(1, 4, 4)
        target = torch.zeros(1, 4, 4)
        
        # 50% overlap
        pred[0, :2, :] = 1
        target[0, 1:3, :] = 1
        
        # Intersection = 4, Union = 12, IoU = 4/12 = 0.333
        iou = iou_score(pred, target)
        
        assert abs(iou - 0.333) < 0.01
    
    def test_dice_perfect_overlap(self):
        """TODO 4.2: Test Dice = 1.0 for perfect overlap."""
        pytest.skip("TODO: Implement test after completing TODO 4.2")
        
        from src.evaluation.metrics import dice_score
        
        pred = torch.ones(1, 256, 256)
        target = torch.ones(1, 256, 256)
        
        dice = dice_score(pred, target)
        
        assert abs(dice - 1.0) < 1e-6
    
    def test_dice_no_overlap(self):
        """TODO 4.2: Test Dice = 0.0 for no overlap."""
        pytest.skip("TODO: Implement test after completing TODO 4.2")
        
        from src.evaluation.metrics import dice_score
        
        pred = torch.ones(1, 256, 256)
        target = torch.zeros(1, 256, 256)
        
        dice = dice_score(pred, target)
        
        assert abs(dice - 0.0) < 1e-6
    
    def test_dice_iou_relationship(self):
        """TODO 4.1-4.2: Verify Dice = 2*IoU/(1+IoU)."""
        pytest.skip("TODO: Implement test after completing TODO 4.1 and 4.2")
        
        from src.evaluation.metrics import iou_score, dice_score
        
        # Random prediction and target
        pred = torch.rand(1, 64, 64) > 0.5
        target = torch.rand(1, 64, 64) > 0.5
        
        iou = iou_score(pred.float(), target.float())
        dice = dice_score(pred.float(), target.float())
        
        # Check relationship
        expected_dice = 2 * iou / (1 + iou)
        assert abs(dice - expected_dice) < 1e-5
    
    def test_multiclass_iou(self):
        """TODO 4.3: Test multi-class IoU calculation."""
        pytest.skip("TODO: Implement test after completing TODO 4.3")
        
        from src.evaluation.metrics import multiclass_iou
        
        # 3 classes, batch of 2
        pred = torch.randint(0, 3, (2, 256, 256))
        target = torch.randint(0, 3, (2, 256, 256))
        
        ious = multiclass_iou(pred, target, num_classes=3)
        
        # Should return per-class IoU and mean
        assert len(ious) == 4  # 3 classes + mean
        assert all(0 <= i <= 1 for i in ious)
    
    def test_empty_mask_handling(self):
        """TODO 4.4: Test handling of empty masks with smoothing."""
        pytest.skip("TODO: Implement test after completing TODO 4.4")
        
        from src.evaluation.metrics import iou_score, dice_score
        
        # Both empty
        pred = torch.zeros(1, 256, 256)
        target = torch.zeros(1, 256, 256)
        
        iou = iou_score(pred, target, smooth=1e-6)
        dice = dice_score(pred, target, smooth=1e-6)
        
        # With smoothing, should be close to 1.0
        assert iou > 0.9
        assert dice > 0.9
    
    def test_batch_processing(self):
        """TODO 4.1-4.3: Test metrics work with batches."""
        pytest.skip("TODO: Implement test after completing metrics")
        
        from src.evaluation.metrics import iou_score
        
        batch_pred = torch.rand(4, 256, 256) > 0.5
        batch_target = torch.rand(4, 256, 256) > 0.5
        
        # Should handle batch dimension
        iou = iou_score(batch_pred.float(), batch_target.float())
        
        assert isinstance(iou, float)
        assert 0 <= iou <= 1
