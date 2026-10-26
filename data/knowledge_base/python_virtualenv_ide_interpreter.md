# Python Virtual Environment Activation and IDE Interpreter Configuration
Category: Python Environment & Tooling
Technologies: Python, Virtualenv, venv, VS Code, PyCharm, IDE

When running Python scripts or web servers, developers often encounter `ModuleNotFoundError: No module named '<package>'` even after installing the library.

## Root Cause
The package is installed inside a specific virtual environment (`.venv` or `venv`), but the active terminal session or IDE (VS Code, PyCharm, Cursor) is pointing to a different Python interpreter (e.g. system `/usr/bin/python3`), which cannot access the virtual environment's site-packages.

## Resolution Steps
1. In VS Code or Cursor, open the Command Palette (`Cmd+Shift+P` on macOS or `Ctrl+Shift+P` on Windows/Linux) and search for `Python: Select Interpreter`.
2. Choose the interpreter located in your project directory: `./.venv/bin/python` (macOS/Linux) or `.\\.venv\\Scripts\\python.exe` (Windows).
3. In your terminal, verify the virtual environment is activated before launching your program: `source .venv/bin/activate` (macOS/Linux) or `.\\.venv\\Scripts\\activate` (Windows).
4. Run `which python` to confirm the active interpreter path matches `.venv`.
5. Verify the package is present in this environment: `python -m pip list`.
