python -m venv venv

source venv/bin/activate
pip install -r cuda-requirements.txt # If your cuda version is different than 12.x, modify the gpu-requirements file to reflect that

unset LD_LIBRARY_PATH
