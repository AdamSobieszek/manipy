
# scripts/prepare_environment.py
import os
import sys
# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.stylegan_utils import setup_stylegan_environment

if __name__ == "__main__":
    print("--- Preparing Environment ---")
    # This function installs packages, clones repos, downloads data/models
    # Check utils/stylegan_utils.py for specifics and adjust paths/commands if needed
    setup_stylegan_environment()
    print("--- Environment Preparation Attempt Finished ---")
    print("Please ensure all paths in configs/config.py are correct for your system.")
