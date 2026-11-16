
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "dataset.csv"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Reproducibility settings
RANDOM_STATE = 42

# Model settings
TEST_SIZE = 0.20
CV_FOLDS = 5
