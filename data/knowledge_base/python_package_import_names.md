# Python PyPI Package Distribution Names vs Module Import Names
Category: Python Package Management & Packaging
Technologies: Python, pip, PyPI

In Python, the name used to install a package via `pip install` often differs from the namespace identifier used in `import <name>`. When developers install using the import name, `pip install` may fail or install an unrelated package, resulting in `ModuleNotFoundError`.

## Root Cause
PyPI project distribution names and the top-level Python import modules do not always share identical naming conventions. Installing the wrong name leaves the required import module missing.

## Resolution Steps
1. Identify the correct PyPI distribution name for your required import:
   - `import cv2` -> `pip install opencv-python`
   - `import yaml` -> `pip install pyyaml`
   - `import sklearn` -> `pip install scikit-learn`
   - `import PIL` -> `pip install pillow`
   - `from bs4 import ...` -> `pip install beautifulsoup4`
   - `import dotenv` -> `pip install python-dotenv`
   - `import dateutil` -> `pip install python-dateutil`
2. Uninstall any erroneous packages: `pip uninstall -y <wrong_name>`.
3. Install the official PyPI package into your active environment: `pip install <correct_pypi_name>`.
4. Test the import directly from terminal: `python -c "import <name>; print('Success!')"`.
