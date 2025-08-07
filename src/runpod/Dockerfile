# Use RunPod's official PyTorch base image to avoid random device issues
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# CUDA Environment Setup - Critical for GPU detection
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install system dependencies for CUDA and ML libraries
RUN apt-get update && apt-get install -y \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages from requirements.txt (fat image approach)
# Force reinstall TensorFlow-GPU to ensure CUDA compatibility
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y tensorflow && \
    pip install --no-cache-dir tensorflow[and-cuda]>=2.13.0

# Copy the handler
COPY handler.py /app/handler.py

# Create enhanced test input file for execute-what-you-send testing with CUDA verification
RUN echo '{"input": {"operation": "execute_code", "code": "import torch; import tensorflow as tf; import numpy as np; print(f\"CUDA available (PyTorch): {torch.cuda.is_available()}\"); print(f\"CUDA available (TensorFlow): {len(tf.config.list_physical_devices(\"GPU\")) > 0}\"); print(f\"GPU devices (TensorFlow): {tf.config.list_physical_devices(\"GPU\")}\"); result = {\"torch_version\": torch.__version__, \"tf_version\": tf.__version__, \"numpy_version\": np.__version__, \"cuda_available_torch\": torch.cuda.is_available(), \"cuda_available_tf\": len(tf.config.list_physical_devices(\"GPU\")) > 0, \"gpu_name\": torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\", \"device_count\": torch.cuda.device_count() if torch.cuda.is_available() else 0, \"tf_gpu_devices\": [device.name for device in tf.config.list_physical_devices(\"GPU\")]}", "timeout_seconds": 60}}' > /app/test_input.json

# Set timezone to Jakarta
ENV TZ=Asia/Jakarta
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Verify CUDA installation at build time
RUN python3 -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}'); import tensorflow as tf; print(f'TensorFlow GPUs: {tf.config.list_physical_devices(\"GPU\")}')" || echo "CUDA verification failed at build time - will retry at runtime"

# Start the handler
CMD ["python", "handler.py"]