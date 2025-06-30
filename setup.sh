#!/bin/bash

# Setup script for Locomotion Training
# This script sets up the environment and installs dependencies

set -e  # Exit on any error

echo "🚀 Setting up Locomotion Training environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.8+ is available
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
}

# Check if we're on a system with GPU support
check_gpu() {
    print_status "Checking for GPU support..."
    
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        GPU_AVAILABLE=true
    else
        print_warning "No NVIDIA GPU detected. Training will be slower on CPU."
        GPU_AVAILABLE=false
    fi
}

# Create virtual environment
create_venv() {
    print_status "Setting up virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Skipping creation."
    else
        print_status "Creating new virtual environment..."
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    elif [ -f "venv/Scripts/activate" ]; then
        # Windows Git Bash / WSL
        source venv/Scripts/activate
        print_success "Virtual environment activated (Windows)"
    else
        print_error "Could not find virtual environment activation script"
        exit 1
    fi
}

# Upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    pip install --upgrade pip
    print_success "Pip upgraded"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies from requirements.txt..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Install base requirements
    pip install -r requirements.txt
    
    # Install JAX with GPU support if available
    if [ "$GPU_AVAILABLE" = true ]; then
        print_status "Installing JAX with CUDA support..."
        pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        print_success "JAX with CUDA support installed"
    else
        print_status "Installing JAX CPU-only version..."
        pip install "jax[cpu]"
        print_success "JAX CPU version installed"
    fi
    
    print_success "All dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p checkpoints
    mkdir -p videos
    mkdir -p logs
    
    print_success "Directories created"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test basic imports
    python -c "
import jax
import mujoco
import brax
from mujoco_playground import registry
print('✓ All core packages imported successfully')
print(f'✓ JAX version: {jax.__version__}')
print(f'✓ JAX devices: {jax.devices()}')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation verification passed"
    else
        print_error "Installation verification failed"
        return 1
    fi
    
    # Test CLI
    print_status "Testing CLI interface..."
    python main.py --help > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        print_success "CLI interface working"
    else
        print_error "CLI interface test failed"
        return 1
    fi
}

# Print usage instructions
print_usage() {
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "To get started:"
    echo "  1. Activate the virtual environment:"
    echo "     source venv/bin/activate"
    echo ""
    echo "  2. List available environments:"
    echo "     python main.py list-envs"
    echo ""
    echo "  3. Train a policy:"
    echo "     python main.py train Go1JoystickFlatTerrain"
    echo ""
    echo "  4. For more options:"
    echo "     python main.py --help"
    echo ""
    
    if [ "$GPU_AVAILABLE" = false ]; then
        print_warning "Note: No GPU detected. For optimal performance, consider using a machine with NVIDIA GPU."
    fi
}

# Main setup function
main() {
    echo "=================================================="
    echo "  Locomotion Training Setup"
    echo "  Based on MuJoCo Playground"
    echo "=================================================="
    echo ""
    
    check_python
    check_gpu
    create_venv
    activate_venv
    upgrade_pip
    install_dependencies
    create_directories
    
    if verify_installation; then
        print_usage
    else
        print_error "Setup completed with errors. Please check the output above."
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    "--help" | "-h")
        echo "Locomotion Training Setup Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --force        Force recreate virtual environment"
        echo "  --cpu-only     Install CPU-only version of JAX"
        echo ""
        exit 0
        ;;
    "--force")
        print_warning "Force mode: removing existing virtual environment"
        rm -rf venv
        ;;
    "--cpu-only")
        print_status "CPU-only mode: will install JAX CPU version"
        GPU_AVAILABLE=false
        ;;
esac

# Run main setup
main