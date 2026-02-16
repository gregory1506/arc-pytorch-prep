# Remote Sensing ML Tutorial - Task List

> **Format**: Each module follows Teach → Code → Test with Pomodoro timing (25 min focus sessions)

---

## Overview
- **Total Modules**: 7
- **Total Pomodoros**: 11 (~4.5 hours of focused work)
- **Format**: Tutorial-based with fill-in-the-blank coding exercises
- **Goal**: End-to-end Sentinel-2 image segmentation pipeline

---

## Module 1: Data Pipeline - Satellite to Tiles ⏱️ 2 Pomodoros (50 min)

### Pomodoro 1: Teaching + Setup (25 min)
- [ ] **Teaching (10 min)**: Sentinel-2 bands, tiling strategies, normalization
  - Key concepts: Multi-spectral imagery, spatial resolution, tile size selection
  - File formats: JP2, GeoTIFF, COG
- [ ] **Setup (5 min)**: Review src/data/dataset.py structure
- [ ] **Environment Check (5 min)**: Verify GDAL, rasterio installation
- [ ] **Student Questions (5 min)**: Q&A on data formats

### Pomodoro 2: Implementation + Testing (25 min)
- [ ] **Code (15 min)**: Fill in dataset.py sections
  - TODO 1.1: Implement Sentinel-2 band loading
  - TODO 1.2: Add tiling logic (256x256 patches)
  - TODO 1.3: Implement normalization (per-band scaling)
  - TODO 1.4: Create PyTorch Dataset class
- [ ] **Test (5 min)**: Run `pytest tests/test_module_01_data.py -v`
- [ ] **Verification (5 min)**: Check shapes, visualize sample tiles

**Deliverable**: Working data loader with tests passing

---

## Module 2: UNet Architecture - Encoder-Decoder Design ⏱️ 2 Pomodoros (50 min)

### Pomodoro 1: Teaching + Architecture Design (25 min)
- [ ] **Teaching (10 min)**: UNet architecture, skip connections, semantic segmentation
  - Encoder: Contracting path (feature extraction)
  - Bottleneck: Deepest features
  - Decoder: Expanding path (localization)
  - Skip connections: Preserve spatial info
- [ ] **Review (5 min)**: Look at src/models/unet.py skeleton
- [ ] **Architecture Planning (5 min)**: Discuss channel dimensions, depth
- [ ] **Discussion (5 min)**: Why UNet for remote sensing?

### Pomodoro 2: Implementation + Testing (25 min)
- [ ] **Code (15 min)**: Fill in unet.py sections
  - TODO 2.1: Implement DoubleConv block (Conv→BN→ReLU→Conv→BN→ReLU)
  - TODO 2.2: Build encoder with max pooling
  - TODO 2.3: Implement decoder with upsampling
  - TODO 2.4: Add skip connection concatenation
  - TODO 2.5: Final 1x1 convolution for class predictions
- [ ] **Test (5 min)**: Run `pytest tests/test_module_02_unet.py -v`
- [ ] **Shape Verification (5 min)**: Confirm output matches input spatial dims

**Deliverable**: Functional UNet model with forward pass working

---

## Module 3: Training Loop - AMP & Checkpointing ⏱️ 2 Pomodoros (50 min)

### Pomodoro 1: Teaching + Setup (25 min)
- [ ] **Teaching (10 min)**: Training loop components, AMP, checkpointing
  - Automatic Mixed Precision (AMP): Memory & speed benefits
  - Checkpointing: Save best model, resume training
  - Loss functions: Dice Loss vs CrossEntropy for segmentation
  - Optimizers: Adam, AdamW, learning rate scheduling
- [ ] **Review (5 min)**: src/training/train.py structure
- [ ] **Loss Function Discussion (5 min)**: Why Dice for imbalanced classes?
- [ ] **Metrics Overview (5 min)**: IoU, Dice coefficient preview

### Pomodoro 2: Implementation + Testing (25 min)
- [ ] **Code (15 min)**: Fill in train.py sections
  - TODO 3.1: Setup device (CUDA/MPS/CPU)
  - TODO 3.2: Initialize model, loss, optimizer
  - TODO 3.3: Implement training loop with AMP
  - TODO 3.4: Add validation loop
  - TODO 3.5: Implement checkpoint saving (best model)
  - TODO 3.6: Add logging (loss curves)
- [ ] **Test (5 min)**: Run `pytest tests/test_module_03_training.py -v`
- [ ] **Overfit Test (5 min)**: Verify model can overfit single batch

**Deliverable**: Training script that saves checkpoints and logs metrics

---

## Module 4: Evaluation Metrics - IoU & Dice ⏱️ 1 Pomodoro (25 min)

### Full Pomodoro: Teaching + Implementation + Testing
- [ ] **Teaching (8 min)**: Metrics for semantic segmentation
  - IoU (Intersection over Union): Jaccard index
  - Dice Coefficient: F1 score for segmentation
  - Confusion Matrix: Per-class performance
  - Why both IoU and Dice?
- [ ] **Code (10 min)**: Fill in src/evaluation/metrics.py
  - TODO 4.1: Implement IoU calculation
  - TODO 4.2: Implement Dice coefficient
  - TODO 4.3: Add per-class and mean metrics
  - TODO 4.4: Handle edge cases (empty masks)
- [ ] **Test (5 min)**: Run `pytest tests/test_module_04_metrics.py -v`
- [ ] **Verification (2 min)**: Test with known values

**Deliverable**: Metrics module with IoU and Dice implementations

---

## Module 5: ONNX Export & Optimization ⏱️ 1 Pomodoro (25 min)

### Full Pomodoro: Teaching + Implementation + Testing
- [ ] **Teaching (8 min)**: Model deployment preparation
  - ONNX: Open Neural Network Exchange format
  - Why export?: Framework interoperability, optimization
  - Quantization: INT8 for faster inference
  - Dynamic vs static axes
- [ ] **Code (10 min)**: Fill in src/optimization/optimize.py
  - TODO 5.1: Load trained checkpoint
  - TODO 5.2: Export to ONNX format
  - TODO 5.3: Verify ONNX model with onnxruntime
  - TODO 5.4: Add optimization passes (optional)
- [ ] **Test (5 min)**: Run `pytest tests/test_module_05_onnx.py -v`
- [ ] **Benchmark (2 min)**: Compare PyTorch vs ONNX inference time

**Deliverable**: Exportable ONNX model with verification tests

---

## Module 6: FastAPI Deployment - REST API ⏱️ 2 Pomodoros (50 min)

### Pomodoro 1: Teaching + API Design (25 min)
- [ ] **Teaching (10 min)**: Serving models via REST API
  - FastAPI: Modern, fast Python web framework
  - Async inference: Handling concurrent requests
  - Request/Response schemas: Pydantic models
  - Batching: Single vs batch predictions
  - Model loading: Singleton pattern, lazy loading
- [ ] **Review (5 min)**: src/api/main.py structure
- [ ] **API Design (5 min)**: Define endpoints (/predict, /health, /metrics)
- [ ] **Error Handling (5 min)**: Discuss failure modes

### Pomodoro 2: Implementation + Testing (25 min)
- [ ] **Code (15 min)**: Fill in main.py sections
  - TODO 6.1: Create FastAPI app instance
  - TODO 6.2: Define Pydantic request/response models
  - TODO 6.3: Implement model loading (ONNX)
  - TODO 6.4: Create /health endpoint
  - TODO 6.5: Create /predict endpoint (single image)
  - TODO 6.6: Add error handling & validation
- [ ] **Test (5 min)**: Run `pytest tests/test_module_06_api.py -v`
- [ ] **Manual Test (5 min)**: Test with curl: `curl -X POST http://localhost:8000/predict`

**Deliverable**: Running FastAPI service with working endpoints

---

## Module 7: Monitoring & Observability ⏱️ 1 Pomodoro (25 min)

### Full Pomodoro: Teaching + Implementation + Testing
- [ ] **Teaching (8 min)**: Production monitoring
  - Prometheus metrics: Latency, throughput, errors
  - Logging: Structured logs for debugging
  - Health checks: Liveness and readiness probes
  - Alerting: When to notify on-call
- [ ] **Code (10 min)**: Add monitoring to src/api/main.py
  - TODO 7.1: Add prometheus-client instrumentation
  - TODO 7.2: Track inference latency histogram
  - TODO 7.3: Add request counter metrics
  - TODO 7.4: Implement /metrics endpoint
  - TODO 7.5: Add structured logging
- [ ] **Test (5 min)**: Run `pytest tests/test_module_07_monitoring.py -v`
- [ ] **Verification (2 min)**: Check metrics endpoint returns Prometheus format

**Deliverable**: Monitored API with metrics and logging

---

## Summary

| Module | Topic | Pomodoros | Status |
|--------|-------|-----------|--------|
| 1 | Data Pipeline | 2 | ⬜ |
| 2 | UNet Architecture | 2 | ⬜ |
| 3 | Training Loop | 2 | ⬜ |
| 4 | Evaluation Metrics | 1 | ⬜ |
| 5 | ONNX Export | 1 | ⬜ |
| 6 | FastAPI Deployment | 2 | ⬜ |
| 7 | Monitoring | 1 | ⬜ |
| **Total** | | **11** | **⬜** |

---

## Quick Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_module_01_data.py -v

# Start API server
uvicorn src.api.main:app --reload

# Check API health
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics
```

---

## Resources

- **Teaching Notes**: See TEACHING.md for each module's lesson plan
- **Architecture**: See architecture.md for data flow diagrams
- **Tests**: See tests/ directory for test specifications
- **Slides**: Create from TEACHING.md Key Concepts sections

---

*Last Updated*: 2024
*Tutorial Format*: Teach → Code → Test (Pomodoro-based)
