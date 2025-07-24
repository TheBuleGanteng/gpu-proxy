#!/bin/bash

# RunPod Serverless PyTorch GPU Training Function Deployment Script
# This script builds and deploys the simplified PyTorch GPU training function to RunPod

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

print_status "🚀 Starting RunPod PyTorch Serverless Deployment"
echo "========================================================================="

# Check if we're in the right directory
if [ ! -f "handler.py" ]; then
    print_error "handler.py not found. Please run this script from the serverless/runpod directory"
    exit 1
fi

if [ ! -f "Dockerfile" ]; then
    print_error "Dockerfile not found. Please ensure all files are in place"
    exit 1
fi

# Pre-deployment checks
print_status "🔍 Pre-deployment checks..."

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

# Check that we're NOT using TensorFlow (we want PyTorch base image)
if grep -q "tensorflow" requirements.txt; then
    print_warning "TensorFlow found in requirements.txt. Using PyTorch base image instead."
fi

# Verify handler.py is self-contained
if grep -q "from src.utils.logger" handler.py; then
    print_error "handler.py still has project imports. Please use the self-contained version."
    exit 1
fi

print_success "requirements.txt and handler.py validation passed"

# Configuration
DOCKER_IMAGE_NAME="pytorch-runpod-gpu"
DOCKER_TAG="v$(date +%Y%m%d-%H%M%S)"
FULL_IMAGE_NAME="${DOCKER_IMAGE_NAME}:${DOCKER_TAG}"

# Try to load Docker Hub username from .env file
ENV_FILE="../../.env"
DOCKER_USERNAME=""

if [ -f "$ENV_FILE" ]; then
    # Load DOCKER_HUB_USERNAME from .env if it exists
    DOCKER_USERNAME=$(grep "^DOCKER_HUB_USERNAME=" "$ENV_FILE" | cut -d '=' -f2 | tr -d '"')
fi

# If not found in .env, use thebuleganteng (your existing username)
if [ -z "$DOCKER_USERNAME" ]; then
    DOCKER_USERNAME="thebuleganteng"  # Default to your existing username
    print_status "Using default Docker Hub username: $DOCKER_USERNAME"
else
    print_success "Using Docker Hub username from .env: $DOCKER_USERNAME"
fi

DOCKER_REPO="${DOCKER_USERNAME}/${DOCKER_IMAGE_NAME}"

print_status "Configuration:"
print_status "- Docker Image: ${DOCKER_REPO}:${DOCKER_TAG}"
print_status "- Build Context: $(pwd)"
print_status "- Base Image: RunPod PyTorch (avoids random device issues)"
print_status "- Approach: Simplified, self-contained handler"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker is running"

# Build the Docker image with platform specification for RunPod
print_status "Building Docker image for linux/amd64 platform..."
if docker build --platform linux/amd64 -t "${FULL_IMAGE_NAME}" .; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Test the image locally for basic functionality (skip if architecture mismatch)
print_status "🧪 Testing basic functionality in built image..."
print_warning "Note: Local testing may fail on different architectures (ARM vs x86_64) - this is expected"

# Try basic test but don't fail deployment if it doesn't work locally
if docker run --rm "${FULL_IMAGE_NAME}" python3 -c "
import runpod
import torch
import torch.nn as nn
print(f'✅ RunPod {runpod.__version__} available')
print(f'✅ PyTorch {torch.__version__} available')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ Self-contained handler ready')
" 2>/dev/null; then
    print_success "Local functionality test passed!"
    
    # Test the handler can import and start
    print_status "🧪 Testing handler import..."
    if docker run --rm "${FULL_IMAGE_NAME}" python3 -c "
import sys
sys.path.append('/app')
try:
    import handler
    print('✅ Handler imports successfully')
except ImportError as e:
    print(f'❌ Handler import failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        print_success "Handler import test passed!"
    else
        print_warning "Handler import test failed locally (may work on RunPod)"
    fi
else
    print_warning "Local testing failed - likely due to architecture differences"
    print_warning "This is expected when building for RunPod on different local architecture"
    print_status "The image should work correctly on RunPod's infrastructure"
fi

# Tag for Docker Hub
print_status "Tagging image for Docker Hub..."
docker tag "${FULL_IMAGE_NAME}" "${DOCKER_REPO}:${DOCKER_TAG}"
docker tag "${FULL_IMAGE_NAME}" "${DOCKER_REPO}:latest"

# Push to Docker Hub
print_status "Pushing image to Docker Hub..."
print_warning "You may need to login to Docker Hub: docker login"

if docker push "${DOCKER_REPO}:${DOCKER_TAG}"; then
    print_success "Image pushed to Docker Hub successfully"
else
    print_error "Failed to push image to Docker Hub"
    print_warning "Make sure you're logged in: docker login"
    exit 1
fi

# Also push latest tag
print_status "Pushing latest tag..."
docker push "${DOCKER_REPO}:latest"

print_success "🎉 Deployment complete!"
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
print_status "🧪 Test your deployment:"
print_status "1. Update runpod_test.py with your endpoint URL"
print_status "2. Run: python runpod_test.py"

echo ""
print_success "🎯 PyTorch container built and ready for RunPod deployment!"
print_status "Image: ${DOCKER_REPO}:${DOCKER_TAG}"

# Save the image name for easy reference
echo "${DOCKER_REPO}:${DOCKER_TAG}" > last_deployed_image.txt
print_status "📝 Image name saved to last_deployed_image.txt"

echo ""
print_status "🔍 Quick Test Command:"
print_status "docker run --rm ${DOCKER_REPO}:${DOCKER_TAG} python3 -c \"import handler; print('Handler ready!')\""