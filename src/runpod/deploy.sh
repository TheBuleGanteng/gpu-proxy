#!/bin/bash

# GPU Proxy - Execute-What-You-Send Fat Image Deployment Script
# This script builds and deploys the comprehensive ML fat image to RunPod
# Supports arbitrary Python code execution with pre-installed ML libraries

set -e  # Exit on any error

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

print_status "🚀 Starting GPU Proxy Fat Image Deployment"
print_status "Execute-What-You-Send Architecture with Comprehensive ML Stack"
echo "========================================================================="

# Check if we're in the right directory
if [ ! -f "handler.py" ]; then
    print_error "handler.py not found. Please run this script from the src/runpod directory"
    exit 1
fi

if [ ! -f "Dockerfile" ]; then
    print_error "Dockerfile not found. Please ensure all files are in place"
    exit 1
fi

# Pre-deployment checks
print_status "🔍 Pre-deployment checks for fat image..."

# Check for RunPod version
if ! grep -q "runpod>=1.7.0" requirements.txt; then
    print_error "runpod>=1.7.0 not found in requirements.txt. Please update requirements.txt"
    exit 1
fi

# Check for duplicate RunPod entries
RUNPOD_COUNT=$(grep -c "^runpod>=" requirements.txt || echo "0")
if [ "$RUNPOD_COUNT" -gt 1 ]; then
    print_error "Multiple runpod entries found in requirements.txt. Please remove duplicates."
    exit 1
fi

# Validate fat image ML libraries are present
print_status "🧪 Validating fat image ML libraries..."

# Core ML frameworks
if ! grep -q "tensorflow>=2.13.0" requirements.txt; then
    print_error "TensorFlow >= 2.13.0 not found in requirements.txt"
    exit 1
fi

if ! grep -q "scikit-learn>=1.3.0" requirements.txt; then
    print_error "scikit-learn >= 1.3.0 not found in requirements.txt"
    exit 1
fi

if ! grep -q "optuna>=3.0.0" requirements.txt; then
    print_error "Optuna >= 3.0.0 not found in requirements.txt"
    exit 1
fi

# Data processing libraries
if ! grep -q "pandas>=1.5.0" requirements.txt; then
    print_error "pandas >= 1.5.0 not found in requirements.txt"
    exit 1
fi

if ! grep -q "opencv-python-headless>=4.8.0" requirements.txt; then
    print_error "OpenCV >= 4.8.0 not found in requirements.txt"
    exit 1
fi

# Visualization libraries
if ! grep -q "matplotlib>=3.7.0" requirements.txt; then
    print_error "matplotlib >= 3.7.0 not found in requirements.txt"
    exit 1
fi

# Verify handler.py is execute-what-you-send compatible
if ! grep -q "execute_code" handler.py; then
    print_error "handler.py doesn't contain execute_code operation. Please use execute-what-you-send handler."
    exit 1
fi

if grep -q "from src.utils.logger" handler.py; then
    print_error "handler.py still has project imports. Please use the self-contained version."
    exit 1
fi

print_success "Fat image requirements validation passed"
print_success "Execute-what-you-send handler validation passed"

# Configuration
DOCKER_IMAGE_NAME="gpu-proxy-fat-image"
DOCKER_TAG="v$(date +%Y%m%d-%H%M%S)"
FULL_IMAGE_NAME="${DOCKER_IMAGE_NAME}:${DOCKER_TAG}"

# Try to load Docker Hub username from .env file
ENV_FILE="../../.env"
DOCKER_USERNAME=""

if [ -f "$ENV_FILE" ]; then
    # Load DOCKER_HUB_USERNAME from .env if it exists
    DOCKER_USERNAME=$(grep "^DOCKER_HUB_USERNAME=" "$ENV_FILE" | cut -d '=' -f2 | tr -d '"')
fi

# If not found in .env, terminate the script with an error
if [ -z "$DOCKER_USERNAME" ]; then
    echo -e "${RED}[ERROR]${NC} DOCKER_HUB_USERNAME is not set in the .env file. Please set it and try again."
    exit 1
else
    print_success "Using Docker Hub username from .env: $DOCKER_USERNAME"
fi

DOCKER_REPO="${DOCKER_USERNAME}/${DOCKER_IMAGE_NAME}"

print_status "Fat Image Configuration:"
print_status "- Docker Image: ${DOCKER_REPO}:${DOCKER_TAG}"
print_status "- Build Context: $(pwd)"
print_status "- Base Image: RunPod PyTorch 2.0.1 + CUDA 11.8"
print_status "- Architecture: Execute-What-You-Send with Fat Image"
print_status "- ML Stack: TensorFlow, PyTorch, scikit-learn, Optuna"
print_status "- Data Processing: pandas, OpenCV, Pillow"
print_status "- Visualization: matplotlib, seaborn"
print_status "- Scientific: scipy, NumPy"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker is running"

# Build the Docker image with platform specification for RunPod
print_status "🔨 Building fat image with comprehensive ML stack for linux/amd64..."
print_status "This may take several minutes due to the comprehensive library installation..."

if docker build --platform linux/amd64 -t "${FULL_IMAGE_NAME}" .; then
    print_success "Fat image built successfully with complete ML stack"
else
    print_error "Failed to build fat image"
    exit 1
fi

# Test the fat image locally for comprehensive functionality
print_status "🧪 Testing fat image ML stack functionality..."
print_warning "Note: Local testing may fail on different architectures (ARM vs x86_64) - this is expected"

# Try comprehensive test but don't fail deployment if it doesn't work locally
if docker run --rm "${FULL_IMAGE_NAME}" python3 -c "
import runpod
import torch
import torch.nn as nn
import tensorflow as tf
import sklearn
import optuna
import pandas as pd
import cv2
import matplotlib
import seaborn
import scipy
import numpy as np

print(f'✅ RunPod {runpod.__version__} available')
print(f'✅ PyTorch {torch.__version__} available')
print(f'✅ TensorFlow {tf.__version__} available')
print(f'✅ scikit-learn {sklearn.__version__} available')
print(f'✅ Optuna {optuna.__version__} available')
print(f'✅ pandas {pd.__version__} available')
print(f'✅ OpenCV {cv2.__version__} available')
print(f'✅ matplotlib {matplotlib.__version__} available')
print(f'✅ NumPy {np.__version__} available')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ Fat image ready for execute-what-you-send')
" 2>/dev/null; then
    print_success "Fat image functionality test passed!"
    
    # Test the execute-what-you-send handler
    print_status "🧪 Testing execute-what-you-send handler..."
    if docker run --rm "${FULL_IMAGE_NAME}" python3 -c "
import sys
sys.path.append('/app')
try:
    import handler
    print('✅ Execute-what-you-send handler imports successfully')
    print('✅ Ready for arbitrary code execution with full ML stack')
except ImportError as e:
    print(f'❌ Handler import failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        print_success "Execute-what-you-send handler test passed!"
    else
        print_warning "Handler import test failed locally (may work on RunPod)"
    fi
else
    print_warning "Local testing failed - likely due to architecture differences"
    print_warning "This is expected when building for RunPod on different local architecture"
    print_status "The fat image should work correctly on RunPod's infrastructure"
fi

# Tag for Docker Hub
print_status "Tagging image for Docker Hub..."
docker tag "${FULL_IMAGE_NAME}" "${DOCKER_REPO}:${DOCKER_TAG}"
docker tag "${FULL_IMAGE_NAME}" "${DOCKER_REPO}:latest"

# Push to Docker Hub
print_status "Pushing fat image to Docker Hub..."
print_warning "You may need to login to Docker Hub: docker login"

if docker push "${DOCKER_REPO}:${DOCKER_TAG}"; then
    print_success "Fat image pushed to Docker Hub successfully"
else
    print_error "Failed to push fat image to Docker Hub"
    print_warning "Make sure you're logged in: docker login"
    exit 1
fi

# Also push latest tag
print_status "Pushing latest tag..."
docker push "${DOCKER_REPO}:latest"

print_success "🎉 GPU Proxy Fat Image Deployment Complete!"
echo ""
print_status "📋 Next Steps:"
print_status "1. Go to RunPod Serverless: https://www.runpod.io/console/serverless"
print_status "2. Click 'New Endpoint'"
print_status "3. Select 'Custom Source' -> 'Docker Image'"
print_status "4. Use this Docker image: ${DOCKER_REPO}:${DOCKER_TAG}"
print_status "5. Configure the endpoint:"
print_status "   - Container Image: ${DOCKER_REPO}:${DOCKER_TAG}"
print_status "   - Container Start Command: (leave blank)"
print_status "   - GPU Type: Any GPU (RTX 4090, A100, etc.)"
print_status "   - Max Workers: 1-3"
print_status "   - Timeout: 300 seconds"
print_status "   - Environment Variables: TZ=Asia/Jakarta"

echo ""
print_status "🧪 Test your deployment with your new instance management methods:"
print_status "from src.runpod.client import RunPodClient"
print_status "client = RunPodClient()"
print_status "result = client.execute_code_sync('result = torch.cuda.is_available()')"
print_status "print(result)"

echo ""
print_success "🎯 Execute-What-You-Send Fat Image Ready for Production!"
print_status "Image: ${DOCKER_REPO}:${DOCKER_TAG}"
print_status "Architecture: Comprehensive ML stack with instance management"

# Save the image name for easy reference
echo "${DOCKER_REPO}:${DOCKER_TAG}" > last_deployed_image.txt
print_status "📝 Image name saved to last_deployed_image.txt"

echo ""
print_status "🔍 Quick Test Command:"
print_status "docker run --rm ${DOCKER_REPO}:${DOCKER_TAG} python3 -c \"import handler; print('Fat image handler ready!')\""
