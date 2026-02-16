# Appendix: Quick Reference

## Module checklist

| Module | Key file | Test command | Time |
| --- | --- | --- | --- |
| 1 | src/data/dataset.py | pytest tests/test_module_01_data.py -v | 50 min |
| 2 | src/models/unet.py | pytest tests/test_module_02_unet.py -v | 50 min |
| 3 | src/training/train.py | pytest tests/test_module_03_training.py -v | 50 min |
| 4 | src/evaluation/metrics.py | pytest tests/test_module_04_metrics.py -v | 25 min |
| 5 | src/optimization/optimize.py | pytest tests/test_module_05_onnx.py -v | 25 min |
| 6 | src/api/main.py | pytest tests/test_module_06_api.py -v | 50 min |
| 7 | src/api/main.py | pytest tests/test_module_07_monitoring.py -v | 25 min |

## Common commands
- python -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt
- pytest tests/ -v
- uvicorn src.api.main:app --reload

## Troubleshooting
- CUDA out of memory: reduce batch size, use AMP
- Tests failing: confirm TODOs are complete and file paths are correct
- Import errors: check that your venv is active
- Port in use: stop the process using port 8000
