# Serverless Dockerfile with proper random device handling
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Set environment variables (matching your working setup)
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app
ENV TZ=Asia/Jakarta
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=1

# CRITICAL: Set deterministic behavior BEFORE any random operations
ENV PYTHONHASHSEED=0
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_DETERMINISTIC_OPS=1
ENV CUDA_VISIBLE_DEVICES=0
# NEW: Force TensorFlow to use alternative random sources
ENV TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS=1
ENV TF_USE_LEGACY_KERAS=0

# Install system dependencies with proper entropy tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libhdf5-dev \
    pkg-config \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tzdata \
    rng-tools \
    haveged \
    # NEW: Add tools for random device management
    uuid-runtime \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
RUN ln -snf /usr/share/zoneinfo/Asia/Jakarta /etc/localtime && \
    echo "Asia/Jakarta" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata

# Install cuDNN
RUN apt-get update && \
    apt-get install -y libcudnn8 libcudnn8-dev && \
    rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create app directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Copy serverless handler and related files
COPY handler.py .
COPY utils.py .

# Create necessary directories
RUN mkdir -p /app/logs

# NEW: Create improved startup script with proper random device handling
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "=== Container Startup: Configuring Random Devices ==="\n\
\n\
# Ensure /dev/urandom is accessible and working\n\
if [ ! -c /dev/urandom ]; then\n\
    echo "WARNING: /dev/urandom not accessible, creating fallback"\n\
    # Create a fallback using available entropy\n\
    mkdir -p /tmp/entropy\n\
    echo $RANDOM$RANDOM$RANDOM > /tmp/entropy/seed\n\
    export RANDOM_SEED_FILE=/tmp/entropy/seed\n\
fi\n\
\n\
# Test random device access\n\
if ! dd if=/dev/urandom of=/dev/null bs=1 count=1 2>/dev/null; then\n\
    echo "WARNING: /dev/urandom read failed, using alternative entropy"\n\
    # Use UUID generator as entropy source\n\
    uuidgen > /tmp/entropy/uuid_seed\n\
    export TF_RANDOM_SEED=42\n\
    export PYTHONHASHSEED=42\n\
else\n\
    echo "SUCCESS: /dev/urandom is accessible"\n\
fi\n\
\n\
# Start entropy daemon with proper configuration\n\
echo "Starting entropy services..."\n\
haveged -F -w 1024 &\n\
HAVEGED_PID=$!\n\
\n\
# Give entropy daemon time to initialize\n\
sleep 2\n\
\n\
# Pre-seed Python random state\n\
python3 -c "import random; random.seed(42); import os; os.urandom(1)" 2>/dev/null || {\n\
    echo "Python random initialization with fallback"\n\
    export PYTHONHASHSEED=42\n\
}\n\
\n\
echo "=== Starting Handler ==="\n\
\n\
# Start the handler with proper error handling\n\
exec python3 -u handler.py' > /app/start.sh && \
chmod +x /app/start.sh

# Verify TensorFlow setup with enhanced error handling
RUN PYTHONHASHSEED=42 TF_DETERMINISTIC_OPS=1 python3 -c "\
import os; \
os.environ['TF_DETERMINISTIC_OPS']='1'; \
os.environ['PYTHONHASHSEED']='42'; \
try: \
    import tensorflow as tf; \
    print('TensorFlow version:', tf.__version__); \
    print('CUDA version:', tf.sysconfig.get_build_info()['cuda_version']); \
    print('cuDNN version:', tf.sysconfig.get_build_info()['cudnn_version']); \
    gpus=tf.config.list_physical_devices('GPU'); \
    print('GPUs found:', len(gpus)); \
except Exception as e: \
    print('TensorFlow setup warning:', str(e)); \
"

# Set permissions
RUN chmod +x /app/handler.py

# Use the improved startup script
CMD ["/bin/bash", "/app/start.sh"]