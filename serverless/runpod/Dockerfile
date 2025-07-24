# Use RunPod's official PyTorch base image to avoid random device issues
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install additional Python packages
RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    numpy \
    requests

# Copy the handler (with corrected logging)
COPY handler.py /app/handler.py

# Create test input file for local testing
RUN echo '{"input": {"operation": "train_epoch", "model_config": {"type": "sequential", "layers": [{"type": "linear", "in_features": 10, "out_features": 3}]}, "training_data": {"features": [[1,2,3,4,5,6,7,8,9,10], [2,3,4,5,6,7,8,9,10,11]], "labels": [0, 1]}, "hyperparameters": {"optimizer": "adam", "learning_rate": 0.001, "loss_function": "cross_entropy"}}}' > /app/test_input.json

# Set timezone to Jakarta
ENV TZ=Asia/Jakarta
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Start the handler
CMD ["python", "handler.py"]