# Clean Multi-stage Dockerfile with Proper Dependency Separation

# Stage 1: Base environment with system dependencies only
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04 as base

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app
ENV TZ=Asia/Jakarta

# Install ONLY system-level dependencies
RUN apt-get update && apt-get install -y \
    # Python runtime (but NOT Python packages)
    python3 \
    python3-pip \
    python3-dev \
    # Build tools
    build-essential \
    git \
    wget \
    curl \
    # System libraries needed by Python packages
    libhdf5-dev \
    pkg-config \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # CUDA deep learning libraries (system level)
    libcudnn8 \
    libcudnn8-dev \
    # Timezone data
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Configure timezone
RUN ln -snf /usr/share/zoneinfo/Asia/Jakarta /etc/localtime && \
    echo "Asia/Jakarta" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata

# Verify Python installation
RUN python3 --version && \
    python3 -m pip --version && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Stage 2: Python dependencies
FROM base as dependencies

# Copy requirements.txt ONLY
COPY requirements.txt .

# Clean requirements.txt and install Python packages ONLY
RUN sed -i 's/scipy==1.16.0/scipy==1.13.1/g' requirements.txt && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/logs /app/optimization_results

# Stage 3: Application code
FROM dependencies as final

# Copy application files
COPY . .

# Set permissions
RUN chmod +x /app/handler.py

CMD ["python3", "handler.py"]