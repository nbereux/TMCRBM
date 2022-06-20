from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR.joinpath("src")
DATASET_DIR = ROOT_DIR.joinpath("dataset")
MODEL_DIR = ROOT_DIR.joinpath("model")
OUTPUT_DIR = ROOT_DIR.joinpath("output")
