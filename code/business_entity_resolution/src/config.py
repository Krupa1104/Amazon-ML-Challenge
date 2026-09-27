"""Central paths and constants. Override any path with an environment variable
(useful on Kaggle / SageMaker), e.g.  ER_DATA_DIR=/kaggle/input/amlc26/dataset
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../business_entity_resolution

# Default layout:  amazon ML challenge/
#                    ├── student_resource/dataset/{train,test}/...
#                    └── business_entity_resolution/   (this repo)
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", REPO_ROOT.parent / "student_resource" / "dataset"))
UTILS_DIR = Path(os.environ.get("ER_UTILS_DIR", DATA_DIR.parent / "utils"))
CACHE_DIR = Path(os.environ.get("ER_CACHE_DIR", REPO_ROOT / "data_cache"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", REPO_ROOT / "output"))

SOURCES = ("source1", "source2", "source3")
VAL_PERCENT = 20  # % of Source 1 entities held out for validation
