# pip cannot install a Python package
Category: Python package management
Technologies: Python, pip, Windows, Linux

When pip install fails, capture the full error and identify whether it is a network, package-resolution, Python-version, build-tool, or permissions failure. Run python -m pip --version to confirm pip belongs to the intended interpreter, then use python -m pip install package-name. Check the package's supported Python versions and your interpreter version. Upgrade pip and wheel inside the active virtual environment if the error concerns old build tooling.

For name-resolution or connection errors, check DNS, proxy, firewall, and package-index reachability. For a wheel build failure, read the first compiler error and install the documented platform build prerequisites. Prefer a virtual environment; avoid administrator installs as a first response. Do not paste private index credentials into logs or shared issue reports.
