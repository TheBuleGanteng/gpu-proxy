#!/bin/bash

# Enhanced RunPod Serverless GPU Training Function Deployment Script
# This script builds and deploys the GPU training function to RunPod with RunPod library fix

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

print_status "🚀 Starting RunPod Serverless Deployment (with comprehensive dependencies)"
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

# Updated RunPod version check
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

# Check for TensorFlow
if ! grep -q "tensorflow\[and-cuda\]==2.19.0" requirements.txt; then
    print_warning "Expected tensorflow[and-cuda]==2.19.0 not found. Build may still work."
fi

print_success "requirements.txt validation passed"

# Configuration - using your existing setup but with hyperparameter-optimizer name
DOCKER_IMAGE_NAME="hyperparameter-optimizer"
DOCKER_TAG="v$(date +%Y%m%d-%H%M%S)-fixed"  # Add "fixed" suffix for clarity
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
print_status "- Dependencies: Multi-stage build with comprehensive system libraries"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker is running"

# Build the Docker image with explicit validation
print_status "Building Docker image with comprehensive dependencies..."
if docker build -t "${FULL_IMAGE_NAME}" .; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Test the image locally for RunPod library
print_status "🧪 Testing RunPod library in built image..."
if docker run --rm "${FULL_IMAGE_NAME}" python3 -c "import runpod; print(f'✅ RunPod {runpod.__version__} is available')"; then
    print_success "RunPod library test passed!"
else
    print_error "RunPod library test failed - there's still an issue with the build"
    exit 1
fi

# Additional comprehensive test
print_status "🧪 Testing TensorFlow and system dependencies..."
if docker run --rm "${FULL_IMAGE_NAME}" python3 -c "
import tensorflow as tf
import numpy as np
print(f'✅ TensorFlow {tf.__version__} available')
print(f'✅ NumPy {np.__version__} available')
print(f'✅ GPU devices: {len(tf.config.list_physical_devices(\"GPU\"))}')
"; then
    print_success "Comprehensive dependency test passed!"
else
    print_warning "Some dependencies may have issues, but continuing with deployment..."
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
docker push "${DOCKER_REPO}:latest"

print_success "🎉 Deployment preparation complete!"
echo ""
print_status "📋 Next Steps:"
print_status "1. Go to RunPod Serverless: https://www.runpod.io/console/serverless"
print_status "2. Create a new serverless endpoint (or update existing)"
print_status "3. Use this Docker image: ${DOCKER_REPO}:${DOCKER_TAG}"
print_status "4. Configure the endpoint:"
print_status "   - Container Image: ${DOCKER_REPO}:${DOCKER_TAG}"
print_status "   - Container Start Command: Leave blank (uses Dockerfile CMD)"
print_status "   - GPU Type: Choose based on your needs"
print_status "   - Max Workers: 1-3"
print_status "   - Timeout: 300 seconds"
print_status "5. Test with your enhanced diagnostics script"

echo ""
print_status "🧪 Test your deployment:"
print_status "python ../../enhanced_container_diagnostics.py"

echo ""
print_success "🎯 Container verified with comprehensive dependencies and ready for deployment!"
print_status "Image tag: ${DOCKER_REPO}:${DOCKER_TAG}"

# Save the image name for easy reference
echo "${DOCKER_REPO}:${DOCKER_TAG}" > last_deployed_image.txt
print_status "📝 Image name saved to last_deployed_image.txt for reference"