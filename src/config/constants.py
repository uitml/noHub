import os
import torch as th
from pathlib import Path


CUDA_AVAILABLE = th.cuda.is_available()
DEVICE = th.device("cuda:0" if CUDA_AVAILABLE else "cpu")
GPUS = 1 if CUDA_AVAILABLE else 0

PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[2]

# Set environment variables DATA_DIR, MODELS_DIR and FEATURES_DIR to override the defaults.
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))
FEATURES_DIR = Path(os.environ.get("FEATURES_DIR", PROJECT_ROOT / "features"))
CUSTOM_MODELS_DIR = MODELS_DIR / "custom"

LOG_FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
