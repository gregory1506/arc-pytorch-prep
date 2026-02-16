# Remote Sensing Image-to-Image ML Project

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **End-to-end ML pipeline for Sentinel-2 image segmentation** with a tutorial-based learning approach.

---

## 🎯 Project Overview

This project teaches you how to build a complete machine learning pipeline for remote sensing image segmentation using PyTorch. You'll work with Sentinel-2 satellite imagery to perform burned area detection and land cover classification.

**What you'll build:**
- 🛰️ Data pipeline for multi-spectral satellite imagery
- 🧠 UNet model for semantic segmentation
- ⚡ Training with mixed precision and checkpointing
- 📊 Evaluation with IoU and Dice metrics
- 🚀 ONNX export for optimized inference
- 🌐 FastAPI REST service for model serving
- 📈 Monitoring with Prometheus metrics

---

## 📚 Tutorial Structure

This is a **tutorial-based project** with 7 modules. Each module follows the **Teach → Code → Test** pattern with **Pomodoro timing** (25-minute focused sessions).

| Module | Topic | Duration | Status |
|--------|-------|----------|--------|
| [01](#module-1-data-pipeline) | Data Pipeline | 2 pomodoros (50 min) | ⬜ |
| [02](#module-2-unet-architecture) | UNet Architecture | 2 pomodoros (50 min) | ⬜ |
| [03](#module-3-training-loop) | Training Loop | 2 pomodoros (50 min) | ⬜ |
| [04](#module-4-evaluation-metrics) | Evaluation Metrics | 1 pomodoro (25 min) | ⬜ |
| [05](#module-5-onnx-export) | ONNX Export | 1 pomodoro (25 min) | ⬜ |
| [06](#module-6-fastapi-deployment) | FastAPI Deployment | 2 pomodoros (50 min) | ⬜ |
| [07](#module-7-monitoring) | Monitoring | 1 pomodoro (25 min) | ⬜ |

**Total Time**: ~4.5 hours of focused learning

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.12 or higher
- **uv**: Fast Python package installer (install via `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **GPU**: Optional but recommended (CUDA-capable or Apple Silicon MPS)
- **Storage**: ~10GB for sample data and models
- **Memory**: 8GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd arc-pytorch-prep

# Create virtual environment with uv (Python 3.12)
uv venv --python 3.12

# Install dependencies
uv pip install -r requirements.txt

# Activate environment (optional, uv commands work without activation)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify installation
pytest tests/ -v
```

### Environment Setup

Create a `.env` file:

```env
# Data paths
DATA_DIR=./data
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed

# Model paths
MODEL_DIR=./models
CHECKPOINT_DIR=./checkpoints

# Training
BATCH_SIZE=16
NUM_WORKERS=4
LEARNING_RATE=1e-4
NUM_EPOCHS=100

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Monitoring
LOG_LEVEL=INFO
METRICS_PORT=9090
```

---

## 📖 Module Guides

### Module 1: Data Pipeline

**Goal**: Load Sentinel-2 imagery and create trainable data loaders

**Key Concepts**:
- Multi-spectral satellite bands (RGB, NIR, SWIR)
- Spatial resolution and tiling strategies
- Data normalization techniques
- PyTorch Dataset and DataLoader

**Files to Edit**:
- `src/data/dataset.py`

**Test Command**:
```bash
pytest tests/test_module_01_data.py -v
```

**Teaching Notes**: See [TEACHING.md](TEACHING.md#module-1-data-pipeline)

---

### Module 2: UNet Architecture

**Goal**: Implement UNet model for semantic segmentation

**Key Concepts**:
- Encoder-decoder architecture
- Skip connections and feature concatenation
- Receptive field and spatial resolution
- Output activation for multi-class segmentation

**Files to Edit**:
- `src/models/unet.py`

**Test Command**:
```bash
pytest tests/test_module_02_unet.py -v
```

**Teaching Notes**: See [TEACHING.md](TEACHING.md#module-2-unet-architecture)

---

### Module 3: Training Loop

**Goal**: Build training loop with AMP and checkpointing

**Key Concepts**:
- Automatic Mixed Precision (AMP)
- Loss functions: Dice Loss vs CrossEntropy
- Optimizers and learning rate scheduling
- Model checkpointing and resumption

**Files to Edit**:
- `src/training/train.py`

**Test Command**:
```bash
pytest tests/test_module_03_training.py -v
```

**Teaching Notes**: See [TEACHING.md](TEACHING.md#module-3-training-loop)

---

### Module 4: Evaluation Metrics

**Goal**: Implement IoU and Dice coefficient metrics

**Key Concepts**:
- Intersection over Union (IoU / Jaccard index)
- Dice coefficient (F1 score)
- Per-class vs mean metrics
- Handling class imbalance

**Files to Edit**:
- `src/evaluation/metrics.py`

**Test Command**:
```bash
pytest tests/test_module_04_metrics.py -v
```

**Teaching Notes**: See [TEACHING.md](TEACHING.md#module-4-evaluation-metrics)

---

### Module 5: ONNX Export

**Goal**: Export trained model to ONNX format

**Key Concepts**:
- ONNX (Open Neural Network Exchange)
- Model serialization and optimization
- Framework interoperability
- Inference acceleration

**Files to Edit**:
- `src/optimization/optimize.py`

**Test Command**:
```bash
pytest tests/test_module_05_onnx.py -v
```

**Teaching Notes**: See [TEACHING.md](TEACHING.md#module-5-onnx-export)

---

### Module 6: FastAPI Deployment

**Goal**: Create REST API for model inference

**Key Concepts**:
- FastAPI framework
- Async request handling
- Request/response validation (Pydantic)
- Error handling and API design

**Files to Edit**:
- `src/api/main.py`

**Test Command**:
```bash
pytest tests/test_module_06_api.py -v
```

**Run API**:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Test Endpoint**:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-image>"}'
```

**Teaching Notes**: See [TEACHING.md](TEACHING.md#module-6-fastapi-deployment)

---

### Module 7: Monitoring

**Goal**: Add observability with metrics and logging

**Key Concepts**:
- Prometheus metrics (latency, throughput, errors)
- Structured logging
- Health checks
- Production monitoring

**Files to Edit**:
- `src/api/main.py` (add instrumentation)

**Test Command**:
```bash
pytest tests/test_module_07_monitoring.py -v
```

**View Metrics**:
```bash
curl http://localhost:8000/metrics
```

**Teaching Notes**: See [TEACHING.md](TEACHING.md#module-7-monitoring)

---

## 🏗️ Project Architecture

See [architecture.md](architecture.md) for detailed data flow diagrams.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Sentinel-2     │────▶│   Preprocessing  │────▶│   Training      │
│  Imagery        │     │   (Tiling/Norm)  │     │   (UNet)        │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Monitoring    │◀────│   FastAPI        │◀────│   ONNX Export   │
│  (Prometheus)   │     │   Inference      │     │   (Optimized)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 📁 Directory Structure

```
arc-pytorch-prep/
├── README.md                 # This file
├── TEACHING.md              # Module teaching notes
├── architecture.md          # Data flow diagrams
├── todo.md                  # Module checklist
├── requirements.txt         # Python dependencies
├── pytest.ini              # Test configuration
├── .env.example            # Environment variables template
├── data/                   # Data directory (gitignored)
│   ├── raw/               # Original Sentinel-2 data
│   └── processed/         # Tiled and normalized data
├── models/                 # Saved models (gitignored)
│   └── checkpoints/       # Training checkpoints
├── src/                    # Source code
│   ├── data/
│   │   └── dataset.py     # Module 1: Data loading
│   ├── models/
│   │   └── unet.py        # Module 2: Model architecture
│   ├── training/
│   │   └── train.py       # Module 3: Training loop
│   ├── evaluation/
│   │   └── metrics.py     # Module 4: Metrics
│   ├── optimization/
│   │   └── optimize.py    # Module 5: ONNX export
│   ├── api/
│   │   └── main.py        # Module 6-7: API & monitoring
│   └── inference/
│       └── infer.py       # Batch inference utilities
├── tests/                  # Test suite
│   ├── conftest.py        # Shared fixtures
│   └── test_module_*.py   # Per-module tests
├── configs/               # Configuration files
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── train_config.yaml
├── docker/                # Docker configurations
│   ├── Dockerfile
│   └── docker-compose.yml
└── monitoring/            # Monitoring configs
    ├── prometheus.yml
    └── grafana/
```

---

## 🧪 Testing

Run the entire test suite:

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_module_01_data.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Stop on first failure
pytest tests/ -x
```

Tests are designed to:
1. **Fail initially** (students fill in code)
2. **Pass after implementation** (verification)
3. **Validate edge cases** (robustness)

---

## 🎓 Learning Path

### For Students

1. **Read TEACHING.md** for the current module
2. **Watch/listen to teaching segment** (5-10 min)
3. **Fill in code** following TODO comments
4. **Run tests** to verify your implementation
5. **Repeat** for next module

### For Instructors

1. **Prepare slides** from TEACHING.md "Key Concepts"
2. **Follow Pomodoro timing** strictly (25 min sessions)
3. **Use provided tests** for instant feedback
4. **Reference architecture.md** for data flow visuals
5. **Adjust pace** based on student progress

---

## 📦 Dependencies

Core stack:

```
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Geospatial
rasterio>=1.3.0
xarray>=2023.1.0

# API
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6

# Monitoring
prometheus-client>=0.17.0

# Export
tonnx>=1.14.0
onnxruntime>=1.15.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Utilities
numpy>=1.24.0
pillow>=10.0.0
tqdm>=4.65.0
pyyaml>=6.0
python-dotenv>=1.0.0
```

See `requirements.txt` for complete list.

---

## 🛠️ Development Commands

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Run API with auto-reload
uvicorn src.api.main:app --reload

# Run training
python -m src.training.train --config configs/train_config.yaml

# Export model
python -m src.optimization.optimize --checkpoint models/best.pth
```

---

## 📊 Data Source

**Sentinel-2** (Copernicus Programme)
- **Resolution**: 10m, 20m, 60m (multi-spectral)
- **Bands**: 13 spectral bands (RGB, NIR, SWIR)
- **Revisit**: Every 5 days
- **Access**: [Copernicus Open Access Hub](https://scihub.copernicus.eu/) or [AWS Open Data](https://registry.opendata.aws/sentinel-2/)

**Tutorial Dataset**:
- We'll use a small subset for training (~100 tiles)
- Sample data can be downloaded with: `python scripts/download_sample_data.py`

---

## 🤝 Contributing

This is a tutorial project. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/module-improvement`
3. Make changes following existing patterns
4. Ensure tests pass: `pytest tests/`
5. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **Sentinel-2**: European Space Agency (ESA) Copernicus Programme
- **PyTorch**: Facebook AI Research (FAIR)
- **FastAPI**: Sebastián Ramírez
- **UNet**: Olaf Ronneberger et al. (2015)

---

## 📞 Support

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Linear**: Project tracking (see issues)

---

**Happy Learning! 🚀**

*Start with Module 1 and work your way through the tutorial.*
