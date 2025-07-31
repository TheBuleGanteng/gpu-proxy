# Use RunPod's official PyTorch base image to avoid random device issues
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python packages from requirements.txt (fat image approach)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler
COPY handler.py /app/handler.py

# Create test input file for execute-what-you-send testing
RUN echo '{"input": {"operation": "execute_code", "code": "import torch; import tensorflow as tf; import numpy as np; result = {\"torch_version\": torch.__version__, \"tf_version\": tf.__version__, \"numpy_version\": np.__version__, \"cuda_available\": torch.cuda.is_available(), \"gpu_name\": torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\", \"device_count\": torch.cuda.device_count() if torch.cuda.is_available() else 0}", "timeout_seconds": 60}}' > /app/test_input.json

# Set timezone to Jakarta
ENV TZ=Asia/Jakarta
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Start the handler
CMD ["python", "handler.py"]