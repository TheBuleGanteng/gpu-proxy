# GPU Proxy - Execute-What-You-Send Cloud GPU Access

A production-ready Python library that provides transparent access to cloud GPU resources through arbitrary code execution. Use remote GPU infrastructure as if it were local hardware.

## 🎯 Vision & Current Status

**Problem**: Machine learning development often requires expensive GPU hardware that many developers can't access locally. Cloud GPU services exist but require complex setup, vendor-specific APIs, and manual resource management.

**Solution**: GPU Proxy provides a unified, execute-what-you-send interface that makes remote GPU resources appear as local hardware. Write your Python code locally, execute it on remote GPUs transparently.

### Current Architecture: Execute-What-You-Send

```python
# Simple, unified interface for any Python code
from src.runpod.client import RunPodClient

# Connect to GPU provider (endpoint must be created manually via RunPod console)
client = RunPodClient()  # Reads from .env

# Execute any Python code on remote GPU
result = client.execute_code_sync("""
import torch
import tensorflow as tf
model = torch.nn.Linear(1000, 10).cuda()
data = torch.randn(32, 1000).cuda()
output = model(data)
result = {
    "output_shape": list(output.shape),
    "gpu_used": torch.cuda.get_device_name(0),
    "memory_allocated": torch.cuda.memory_allocated(0),
    "tensorflow_version": tf.__version__
}
""")

print(result['execution_result']['result'])
```

### Key Benefits

- **🔄 True Project Agnosticism**: Execute any Python code, not just ML training
- **💰 Cost Efficient**: Pay only for GPU compute time, not idle infrastructure  
- **🔧 Developer Friendly**: Write code locally, execute remotely with same syntax
- **📊 Real-time Monitoring**: Live execution output, errors, and performance metrics
- **🛡️ Production Ready**: Robust error handling, timeouts, session management
- **🚀 GPU Acceleration**: Confirmed working with pre-installed ML libraries
- **⚡ Streamlined Setup**: Clear manual setup process with automatic validation

## 🏗️ Current Architecture Overview

```
┌─────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│   Your Project  │───▶│   GPU Proxy Core  │───▶│  RunPod GPU     │
│                 │    │                   │    │                 │
│ Python Code     │    │ ┌───────────────┐ │    │ ┌─────────────┐ │
│ Generation      │    │ │ RunPod Client │ │    │ │ Serverless  │ │
│ Context Data    │    │ │ Job Execution │ │    │ │ Handler     │ │
└─────────────────┘    │ └───────────────┘ │    │ └─────────────┘ │
                       │ ┌───────────────┐ │    │ ┌─────────────┐ │
                       │ │ All Job       │ │    │ │ Fat Image   │ │
                       │ │ Operations    │ │    │ │ Complete ML │ │
                       │ └───────────────┘ │    │ │ Stack       │ │
                       └───────────────────┘    │ └─────────────┘ │
                                                └─────────────────┘
```

## 📦 Current Project Structure

```
gpu-proxy/
├── LICENSE
├── README.md
├── requirements.txt
├── .env                        # ✅ Environment variables
├── examples/
│   └── basic_usage.py          # Usage examples
├── src/
│   ├── runpod/
│   │   ├── client.py           # ✅ Complete RunPod job execution client
│   │   ├── handler.py          # ✅ Execute-what-you-send handler
│   │   ├── test.py             # ✅ Comprehensive test suite
│   │   ├── Dockerfile          # ✅ Fat image container configuration
│   │   ├── deploy.sh           # ✅ Updated deployment script
│   │   └── requirements.txt    # ✅ Comprehensive ML dependencies
│   ├── utils/
│   │   └── logger.py           # ✅ Enhanced logging utility
│   ├── aws/                    # 🔮 Future: AWS provider
│   └── gcp/                    # 🔮 Future: GCP provider
└── tests/                      # ✅ Complete test coverage
```

## ✅ Current Status - PRODUCTION READY

### Fully Working Components ✅

#### 1. **RunPod Job Execution Client** ✅
- **Complete API Coverage**: All RunPod serverless job endpoints implemented
  - ✅ `/run` - Asynchronous job submission
  - ✅ `/runsync` - Synchronous execution
  - ✅ `/health` - Health monitoring  
  - ✅ `/cancel/{job_id}` - Job cancellation
  - ✅ `/purge-queue` - Queue management
  - ✅ `/status/{job_id}` - Status tracking
  - ✅ `/stream/{job_id}` - Real-time streaming
- **Endpoint Validation**: Automatic verification that endpoint exists and is accessible
- **Environment Integration**: Reads credentials from `.env` file with validation
- **Type Safety**: Full type hints and error handling
- **Convenience Methods**: High-level functions like `execute_code_sync()`
- **Convenience Functions**: Direct function access without class instantiation

#### 2. **Execute-What-You-Send Handler** ✅
- **Arbitrary Code Execution**: Run any Python code with GPU access
- **Fat Image Support**: Pre-imported PyTorch, TensorFlow, NumPy, scikit-learn, Optuna
- **Operational Safety**: Configurable timeouts and error handling
- **System Monitoring**: GPU info, benchmarking, health checks
- **Context Support**: Pass data to remote code execution

#### 3. **Fat Image Container** ✅
- **Comprehensive ML Stack**: 
  - **Core Frameworks**: PyTorch 2.0.1+cu118, TensorFlow 2.19+
  - **Optimization**: Optuna 3.0+ for hyperparameter tuning
  - **Classical ML**: scikit-learn 1.3+ for traditional algorithms
  - **Data Processing**: pandas, OpenCV, Pillow
  - **Visualization**: matplotlib, seaborn
  - **Scientific Computing**: scipy, NumPy
- **Optimized Deployment**: Docker best practices with requirements.txt
- **Production Deployed**: `thebuleganteng/gpu-proxy-fat-image:latest`

#### 4. **Infrastructure Operations** ✅
- **Health Monitoring**: Real-time endpoint and GPU status
- **Performance Benchmarking**: GPU acceleration verification (1.57ms matrix ops)
- **Error Handling**: Comprehensive error reporting and recovery
- **Logging**: Detailed execution logging with configurable levels

#### 5. **Testing & Validation** ✅
- **Comprehensive Test Suite**: 100% pass rate on all tests (9/9 passed)
  - ✅ Health monitoring tests
  - ✅ Synchronous code execution tests  
  - ✅ Context data passing tests
  - ✅ System info and benchmarking tests
  - ✅ Async job workflow tests
  - ✅ Convenience function tests
- **Environment Management**: `.env` file loading with python-dotenv
- **Production Validation**: Confirmed working on 2-GPU RunPod infrastructure
- **Automatic Setup Guidance**: Shows detailed instructions when manual setup needed

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+
- RunPod account and API key
- Docker (for deployment)
- python-dotenv (`pip install python-dotenv`)

### Step 1: Deploy Container

```bash
# Navigate to RunPod deployment
cd src/runpod

# Deploy fat image with comprehensive ML libraries
./deploy.sh
```

**Fat Image Includes:**
- **ML Frameworks**: PyTorch 2.0.1+cu118, TensorFlow 2.19+
- **Optimization**: Optuna for hyperparameter tuning
- **Classical ML**: scikit-learn for traditional algorithms
- **Data Processing**: pandas, OpenCV, Pillow
- **Visualization**: matplotlib, seaborn
- **Scientific**: scipy, NumPy for mathematical operations

### Step 2: Manual RunPod Setup (Required)

**⚠️ IMPORTANT**: Serverless endpoints must be created manually via the RunPod console due to API limitations.

1. **Create Template**:
   - Go to: https://www.runpod.io/console/serverless/user/templates
   - Click 'New Template'
   - Enter:
     - Template Name: `gpu-proxy-template`
     - Container Image: `thebuleganteng/gpu-proxy-fat-image:latest`
     - Container Disk: 5 GB
     - Docker Command: `python handler.py`
     - Volume Size: 0 GB

2. **Create Endpoint**:
   - Go to: https://www.runpod.io/console/serverless/user/endpoints
   - Click 'New Endpoint'
   - Configure:
     - Endpoint Name: `gpu-proxy-endpoint`
     - Select your template: `gpu-proxy-template`
     - GPU Type: RTX 4090 or similar
     - Max Workers: 1-3
     - Min Workers: 0
     - Idle Timeout: 5 seconds

3. **Environment Setup**:
   ```bash
   # Create .env file in project root
   cat > .env << EOF
   RUNPOD_API_KEY=your_api_key_here
   RUNPOD_ENDPOINT_ID=your_endpoint_id_here
   DOCKER_HUB_USERNAME=thebuleganteng
   EOF
   ```

### Step 3: Validate Setup

```bash
# Test your configuration
cd src/runpod
python test.py
```

**Expected Result**: 100% test success rate (9/9 tests passed)

## 💻 API Reference

### RunPodClient Class

```python
from src.runpod.client import RunPodClient

client = RunPodClient(endpoint_id=None, api_key=None)  # Uses .env if None
```

#### Core Methods

- **`execute_code_sync(code, context=None, timeout=300)`** - Execute code and wait for results
- **`run(input_data, timeout=300)`** - Submit asynchronous job
- **`runsync(input_data, timeout=300)`** - Submit synchronous job
- **`health()`** - Check endpoint health
- **`status(job_id)`** - Get job status  
- **`cancel(job_id)`** - Cancel running job
- **`stream(job_id)`** - Stream real-time updates
- **`wait_for_completion(job_id, poll_interval=1.0, max_wait=600)`** - Poll until complete

#### Convenience Functions

```python
from src.runpod.client import execute_code_sync, health

# Direct function calls (no class instantiation)
result = execute_code_sync("result = torch.cuda.device_count()")
status = health()
```

### Handler Operations

The serverless handler supports these operations:

- **`execute_code`** - Run arbitrary Python code (default)
- **`system_info`** - Get GPU and system information  
- **`benchmark`** - Run performance benchmarks
- **`health_check`** - Basic health verification

## 🔧 Advanced Usage

### Context Data Passing

```python
# Pass data to remote execution
result = client.execute_code_sync(
    code="""
    # Access data via 'context' variable
    x_train = context['training_data']
    y_train = context['labels']
    
    # Your ML code here
    import torch
    model = torch.nn.Linear(len(x_train[0]), 10).cuda()
    # ... training logic ...
    result = {"accuracy": 0.95, "loss": 0.1}
    """,
    context={
        "training_data": your_training_data.tolist(),
        "labels": your_labels.tolist()
    }
)
```

### Error Handling

```python
try:
    result = client.execute_code_sync(your_code, timeout_seconds=600)
    
    if result['execution_result']['success']:
        print("Success:", result['execution_result']['result'])
    else:
        print("Execution error:", result['execution_result']['error'])
        print("Stdout:", result['execution_result']['stdout'])
        print("Stderr:", result['execution_result']['stderr'])
        
except ValueError as e:
    print("Request error:", e)
except TimeoutError as e:
    print("Timeout:", e)
```

### Async Job Management

```python
# Submit long-running job
response = client.run({
    "operation": "execute_code",
    "code": your_long_training_code,
    "timeout_seconds": 3600
})
job_id = response['id']

# Monitor progress with streaming
for update in client.stream(job_id):
    print(f"Update: {update}")
    if update.get('status') in ['COMPLETED', 'FAILED']:
        break

# Or wait for completion
final_result = client.wait_for_completion(job_id, max_wait=3600)
```

## 🧪 Testing

### Run Comprehensive Test Suite

```bash
cd src/runpod
python test.py
```

**Test Coverage:**
- ✅ Health monitoring and endpoint validation
- ✅ Synchronous code execution with ML libraries
- ✅ Context data passing and processing
- ✅ System info and GPU benchmarking
- ✅ Async job workflows with streaming
- ✅ Convenience functions
- ✅ Error handling and edge cases

**Latest Test Results:**
- **Total Tests**: 9
- **Passed**: 9 (100%)
- **Failed**: 0
- **Success Rate**: 100%

## 🔌 Integration Examples

### Basic ML Training

```python
# Add to your project
import sys
sys.path.append('/path/to/gpu-proxy')

from src.runpod.client import RunPodClient

# Initialize client (reads from .env)
client = RunPodClient()

# Execute ML training on remote GPU
result = client.execute_code_sync("""
import torch
import torch.nn as nn
import torch.optim as optim

# Build model on GPU
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).cuda()

# Example training step (replace with your data)
dummy_data = torch.randn(32, 784).cuda()
dummy_labels = torch.randint(0, 10, (32,)).cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training step
optimizer.zero_grad()
output = model(dummy_data)
loss = criterion(output, dummy_labels)
loss.backward()
optimizer.step()

result = {
    "loss": loss.item(),
    "model_parameters": sum(p.numel() for p in model.parameters()),
    "gpu_memory_used": torch.cuda.memory_allocated(0),
    "device": torch.cuda.get_device_name(0)
}
""")

print(f"Training result: {result['execution_result']['result']}")
```

### Hyperparameter Optimization Integration

```python
from src.runpod.client import RunPodClient
import optuna

class GPUAcceleratedOptimizer:
    def __init__(self):
        self.gpu_client = RunPodClient()
    
    def objective(self, trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
        
        # Generate training code with hyperparameters
        training_code = f"""
import torch
import torch.nn as nn
import torch.optim as optim

# Model with suggested hyperparameters
model = nn.Sequential(
    nn.Linear(784, {hidden_dim}),
    nn.ReLU(),
    nn.Linear({hidden_dim}, 10)
).cuda()

# Training configuration
batch_size = {batch_size}
learning_rate = {lr}

# Your training loop here...
# (replace with actual training data and loop)

# Return performance metric
result = {{"accuracy": 0.95, "loss": 0.1}}  # Replace with actual metrics
"""
        
        # Execute on remote GPU
        result = self.gpu_client.execute_code_sync(training_code)
        
        # Return objective value
        return result['execution_result']['result']['accuracy']
    
    def optimize(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params

# Usage
optimizer = GPUAcceleratedOptimizer()
best_params = optimizer.optimize(n_trials=50)
print(f"Best hyperparameters: {best_params}")
```

### Data Processing Pipeline

```python
from src.runpod.client import RunPodClient

def process_large_dataset(data):
    client = RunPodClient()
    
    # Process data on GPU
    result = client.execute_code_sync(
        code="""
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

# Get data from context
raw_data = np.array(context['data'])

# GPU-accelerated preprocessing
data_tensor = torch.tensor(raw_data, dtype=torch.float32).cuda()

# Normalize on GPU
mean = data_tensor.mean(dim=0)
std = data_tensor.std(dim=0)
normalized_data = (data_tensor - mean) / (std + 1e-8)

# Additional GPU processing
processed_data = torch.relu(normalized_data)  # Example processing

result = {
    "processed_data": processed_data.cpu().numpy().tolist(),
    "mean": mean.cpu().numpy().tolist(),
    "std": std.cpu().numpy().tolist(),
    "gpu_used": torch.cuda.get_device_name(0)
}
""",
        context={"data": data}
    )
    
    return result['execution_result']['result']

# Usage
large_dataset = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] * 1000  # Example data
processed = process_large_dataset(large_dataset)
print(f"Processed on: {processed['gpu_used']}")
```

## 🎯 Integration Philosophy

### Design Principles

1. **Infrastructure Transparency**: GPU proxy acts as hardware extension, not ML service
2. **Project Agnosticism**: Works with any Python code, any ML framework
3. **Developer Control**: You maintain complete control over training logic
4. **Minimal Coupling**: No vendor lock-in, easy to switch providers
5. **Performance Focus**: Minimal overhead, maximum GPU utilization

### For Your Projects

GPU Proxy integrates seamlessly with existing workflows:

- **ML Training**: Execute training loops on remote GPUs
- **Data Processing**: GPU-accelerated data preprocessing
- **Model Inference**: Run inference on large datasets
- **Research**: Experiment with different architectures and approaches
- **Hyperparameter Optimization**: Distributed parameter search

## 🧪 Validation Results

- **Test Success Rate**: 100% (9/9 tests passed)
- **GPU Performance**: 1.57ms computation time confirmed
- **Library Support**: PyTorch 2.0.1+cu118, TensorFlow 2.19.0 validated
- **CUDA Availability**: 2 GPUs detected and accessible
- **Production Ready**: All core functionality tested and working

## 🤝 Contributing

Key areas for contribution:

1. **Provider Extensions**: Add support for new cloud GPU providers
2. **Framework Support**: Optimize for specific ML frameworks  
3. **Performance Optimization**: Reduce latency and improve throughput
4. **Documentation**: Examples, tutorials, best practices

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Current Status**: ✅ **PRODUCTION READY** - Execute-what-you-send architecture with comprehensive ML stack

**Latest Achievement**: 100% test success rate with complete ML library validation

**Vision**: The standard interface for transparent cloud GPU access