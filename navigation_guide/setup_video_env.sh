#!/bin/bash
# Setup script specifically for navigation video generation
# This includes all the fixes and requirements we discovered

set -e  # Exit on any error

echo "🎥 Setting up Navigation Video Generation Environment"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "[i] $1"
}

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Virtual environment not activated. Activating now..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found. Please run setup.sh first"
        exit 1
    fi
fi

echo ""
echo "1. Installing Required System Libraries"
echo "--------------------------------------"

# Check for required graphics libraries
check_and_install_lib() {
    local lib_name=$1
    local package_name=$2
    
    if ldconfig -p 2>/dev/null | grep -q "$lib_name"; then
        print_success "$lib_name already installed"
    else
        print_info "Installing $package_name..."
        if command -v apt >/dev/null 2>&1; then
            apt install -y $package_name || {
                print_error "Failed to install $package_name"
                print_warning "Try running: sudo apt install $package_name"
                return 1
            }
            print_success "$package_name installed"
        else
            print_error "apt not found. Please manually install: $package_name"
            return 1
        fi
    fi
}

# Install required libraries
check_and_install_lib "libOSMesa" "libosmesa6-dev"
check_and_install_lib "libGL" "libgl1-mesa-glx"
check_and_install_lib "libGLU" "libglu1-mesa-dev"

# Check for ffmpeg
if command -v ffmpeg >/dev/null 2>&1; then
    print_success "ffmpeg already installed"
else
    print_info "Installing ffmpeg..."
    apt install -y ffmpeg || {
        print_error "Failed to install ffmpeg"
        print_warning "Try running: sudo apt install ffmpeg"
    }
fi

echo ""
echo "2. Installing Python Dependencies"
echo "--------------------------------"

# Install OpenCV for video annotation
print_info "Installing OpenCV..."
pip install opencv-python >/dev/null 2>&1 && print_success "OpenCV installed" || print_warning "OpenCV installation failed"

# Ensure mediapy is installed
print_info "Checking mediapy..."
pip install mediapy >/dev/null 2>&1 && print_success "mediapy ready" || print_warning "mediapy installation failed"

echo ""
echo "3. Creating CUDA 12 Setup Script"
echo "--------------------------------"

# Check if setup_cuda12_env.sh exists
if [ ! -f "setup_cuda12_env.sh" ]; then
    print_info "Creating setup_cuda12_env.sh..."
    cat > setup_cuda12_env.sh << 'EOF'
#!/bin/bash
# Setup environment to use CUDA 12 libraries from Python packages

VENV_PATH="$(pwd)/venv"
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
EOF
    chmod +x setup_cuda12_env.sh
    print_success "Created setup_cuda12_env.sh"
else
    print_success "setup_cuda12_env.sh already exists"
fi

echo ""
echo "4. Testing Video Generation Setup"
echo "--------------------------------"

# Test OSMesa rendering
print_info "Testing OSMesa rendering..."
python -c "
import os
os.environ['MUJOCO_GL'] = 'osmesa'
try:
    import mujoco
    print('✓ OSMesa backend working')
except Exception as e:
    print(f'✗ OSMesa test failed: {e}')
" || print_warning "OSMesa test failed - video generation may not work"

# Test CUDA setup
print_info "Testing CUDA 12 setup..."
if [ -f "setup_cuda12_env.sh" ]; then
    source setup_cuda12_env.sh >/dev/null 2>&1
    if python -c "import jax; print('gpu' in str(jax.devices()).lower())" | grep -q "True"; then
        print_success "CUDA 12 setup working"
    else
        print_warning "GPU not detected - will use CPU (slower)"
    fi
fi

echo ""
echo "5. Creating Video Generation Helper"
echo "----------------------------------"

# Create a helper script for video generation
cat > navigation_guide/generate_video.sh << 'EOF'
#!/bin/bash
# Helper script to generate navigation videos

# Set up environment
source setup_cuda12_env.sh
export MUJOCO_GL=osmesa

# Check if checkpoint directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_directory> [seed]"
    echo "Example: $0 checkpoints/navigation_ppo_20250711_235524/best 42"
    exit 1
fi

CHECKPOINT_DIR=$1
SEED=${2:-42}

# Generate video with info overlay
python generate_navigation_video_with_info.py \
    --checkpoint "$CHECKPOINT_DIR" \
    --seed "$SEED"

echo "Video saved to videos/ directory"
EOF

chmod +x navigation_guide/generate_video.sh
print_success "Created helper script: navigation_guide/generate_video.sh"

echo ""
echo "======================================"
echo "✅ Video Generation Setup Complete!"
echo "======================================"
echo ""
echo "To generate videos, use these commands:"
echo ""
echo "1. Set up CUDA environment:"
echo "   ${GREEN}source setup_cuda12_env.sh${NC}"
echo ""
echo "2. Set rendering backend:"
echo "   ${GREEN}export MUJOCO_GL=osmesa${NC}"
echo ""
echo "3. Generate video:"
echo "   ${GREEN}python generate_navigation_video_with_info.py${NC}"
echo ""
echo "Or use the helper script:"
echo "   ${GREEN}./navigation_guide/generate_video.sh <checkpoint_path>${NC}"
echo ""

# Check for potential issues
if ! ldconfig -p 2>/dev/null | grep -q "libOSMesa"; then
    print_warning "OSMesa libraries not found in system"
    print_warning "You may need to run with sudo: sudo bash navigation_guide/setup_video_env.sh"
fi

echo ""
print_info "For troubleshooting, see: navigation_guide/video_generation_issues.md"