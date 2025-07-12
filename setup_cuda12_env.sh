#!/bin/bash
# Setup environment to use CUDA 12 libraries from Python packages

VENV_PATH="/workspace/playground_test/venv"
SITE_PACKAGES="$VENV_PATH/lib/python3.10/site-packages"

# Find CUDA 12 library paths
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cuda_cupti/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cuda_nvcc/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"

# Set CUDA_HOME to use nvcc from the package
export CUDA_HOME="$SITE_PACKAGES/nvidia/cuda_nvcc"
export PATH="$CUDA_HOME/bin:$PATH"

# Unset CUDA_VISIBLE_DEVICES to use GPU
unset CUDA_VISIBLE_DEVICES
unset JAX_PLATFORMS

echo "CUDA 12 environment setup complete"
echo "LD_LIBRARY_PATH includes CUDA 12 libraries"
echo ""

# Test JAX
python -c "
import jax
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
print('JAX backend:', jax.default_backend())
"