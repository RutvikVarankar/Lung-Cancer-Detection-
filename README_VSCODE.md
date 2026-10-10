# PROJECT_LUNGS notebook for VS Code

1. Open this folder in VS Code and install the Python and Jupyter extensions.
2. Recommended: create a Python 3.10 (64-bit) virtual environment.
3. In the VS Code terminal:
   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Place the dataset so the chosen dataset directory contains three class folders (for example `Benign`, `Malignant`, and `Normal`).
5. Open `PROJECT_LUNGS_VSCode.ipynb`, select the `.venv` kernel, update `DATASET_DIR` if needed, then run cells from top to bottom.

The notebook trains a VGG16 transfer-learning model and saves the model plus plots locally.
This is an educational project and not a medical diagnostic tool.
