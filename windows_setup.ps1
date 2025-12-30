python -m venv ven

# Activate virtual environment (Windows venv)
.\venv\Scripts\Activate.ps1

# Install CUDA requirements
pip install -r cuda-requirements.txt

# Unset LD_LIBRARY_PATH (mostly irrelevant on Windows, but safe)
Remove-Item Env:LD_LIBRARY_PATH -ErrorAction SilentlyContinue