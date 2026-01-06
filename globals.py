import os
import sys

# Dynamically find the repo root (no hardcoding!)
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__))
)

# Optional: define commonly used subdirectories
TOOLS_DIR = os.path.join(REPO_ROOT, "fairmast-data-preprocessing")
CONFIG_DIR = os.path.join(REPO_ROOT, "scripts", "pipelines", "configs")
DATA_DIR = os.path.join(REPO_ROOT, "metadata")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

# You can print to verify when developing
if __name__ == "__main__":
    print("Repo root:", REPO_ROOT)
    print("Config dir:", CONFIG_DIR)
    print("Data dir:", DATA_DIR)
