#!/bin/bash

# Set Python and CUDA versions
PYTHON_VERSION=3.10
ENVIRONMENT=conv_onet

# Path to Conda initialization script (this path may vary)
source ~/miniconda3/etc/profile.d/conda.sh

# Step 1: Deactivate and remove the existing environment
echo "==> Removing the current environment."
conda deactivate
conda remove --all -n ${ENVIRONMENT} -y

# Step 2: Create the Conda environment with the specified Python version
echo "==> Creating environment with Python ${PYTHON_VERSION}."
conda create -n ${ENVIRONMENT} python=${PYTHON_VERSION} -y

# Step 3: Activate the newly created environment
echo "==> Activating environment."
conda activate ${ENVIRONMENT}

# Step 4: Install PyTorch
echo "==> Installing PyTorch."
pip install torch torchvision

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu126.html

# Step 6: Install the dependencies from the pyproject.toml
echo "==> Install missing dependencies."
pip install -r requirement.txt
python setup.py build_ext --inplace