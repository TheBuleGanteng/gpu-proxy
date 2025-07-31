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
- Git
- Docker (for deployment)

### Step 1: Clone and Set Up GPU Proxy

```bash
# Clone the repository
git clone https://github.com/TheBuleGanteng/gpu-proxy.git
cd gpu-proxy

# Install required dependencies
pip install python-dotenv requests

# Verify the installation
python -c "from src.runpod.client import RunPodClient; print('✅ GPU Proxy imported successfully')"
```

### Step 2: Deploy Container

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

### Step 3: Manual RunPod Setup (Required)

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

### Step 4: Validate Setup

```bash
# Test your configuration
cd src/runpod
python test.py
```

**Expected Result**: 100% test success rate (9/9 tests passed)

## 🔌 Integration Into Your Projects

### Method 1: Direct Repository Integration (Recommended)

```bash
# In your project directory
git clone https://github.com/TheBuleGanteng/gpu-proxy.git

# Add to your Python path
import sys
sys.path.append('./gpu-proxy')

# Import and use
from src.runpod.client import RunPodClient

client = RunPodClient()
result = client.execute_code_sync("import torch; result = torch.cuda.is_available()")
print(result)
```

### Method 2: Submodule Integration

```bash
# Add as a Git submodule to your project
git submodule add https://github.com/TheBuleGanteng/gpu-proxy.git gpu-proxy
git submodule update --init --recursive

# In your Python code
import sys
sys.path.append('./gpu-proxy')
from src.runpod.client import RunPodClient
```

### Method 3: Environment Setup

```bash
# Clone to a central location
git clone https://github.com/TheBuleGanteng/gpu-proxy.git ~/gpu-proxy

# Add to your PYTHONPATH in ~/.bashrc or ~/.zshrc
export PYTHONPATH="${PYTHONPATH}:${HOME}/gpu-proxy"

# Or set in your project's .env file
echo "PYTHONPATH=~/gpu-proxy:$PYTHONPATH" >> .env
```

### Project Structure After Integration

```
your-project/
├── gpu-proxy/                 # Cloned repository
│   ├── src/
│   │   ├── runpod/
│   │   │   ├── client.py      # Main client
│   │   │   └── ...
│   │   └── utils/
│   └── .env                   # GPU Proxy config
├── your_code.py
├── requirements.txt
└── .env                       # Your project config
```

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

## 🔌 Integration Examples

### Example 1: Simple Integration

```python
# your_ml_project.py
import sys
sys.path.append('./gpu-proxy')  # Add GPU Proxy to path

from src.runpod.client import RunPodClient
import numpy as np

def train_model_on_gpu(training_data, labels):
    """Train a model using remote GPU resources"""
    
    client = RunPodClient()  # Reads from gpu-proxy/.env
    
    # Prepare your training code
    training_code = """
import torch
import torch.nn as nn
import torch.optim as optim

# Get data from context
X = torch.tensor(context['training_data'], dtype=torch.float32).cuda()
y = torch.tensor(context['labels'], dtype=torch.long).cuda()

# Define model
model = nn.Sequential(
    nn.Linear(len(X[0]), 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
).cuda()

# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Return results
with torch.no_grad():
    predictions = model(X)
    accuracy = (predictions.argmax(1) == y).float().mean().item()

result = {
    'final_loss': loss.item(),
    'accuracy': accuracy,
    'model_params': sum(p.numel() for p in model.parameters()),
    'gpu_used': torch.cuda.get_device_name(0)
}
"""
    
    # Execute on remote GPU
    result = client.execute_code_sync(
        training_code,
        context={
            'training_data': training_data.tolist(),
            'labels': labels.tolist()
        },
        timeout_seconds=300
    )
    
    if result['execution_result']['success']:
        return result['execution_result']['result']
    else:
        raise RuntimeError(f"Training failed: {result['execution_result']['error']}")

# Usage
if __name__ == "__main__":
    # Your local data
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 10, 1000)
    
    # Train on remote GPU
    results = train_model_on_gpu(X_train, y_train)
    print(f"Training completed on {results['gpu_used']}")
    print(f"Final accuracy: {results['accuracy']:.4f}")
```

### Example 2: Hyperparameter Optimization Project

```python
# hyperparameter_optimizer.py
import sys
sys.path.append('./gpu-proxy')

from src.runpod.client import RunPodClient
import optuna
import json

class GPUHyperparameterOptimizer:
    def __init__(self, dataset_path=None):
        self.client = RunPodClient()
        self.dataset_path = dataset_path
        
    def optimize_model(self, n_trials=50):
        """Run hyperparameter optimization using remote GPU"""
        
        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            
            # Prepare optimization code
            optimization_code = f"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Hyperparameters from trial
learning_rate = {lr}
batch_size = {batch_size}
hidden_layers = {hidden_layers}
dropout_rate = {dropout}

# Load your dataset (replace with your data loading logic)
# For demo: generate synthetic data
X = torch.randn(5000, 100).cuda()
y = torch.randint(0, 10, (5000,)).cuda()

# Split data
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Build dynamic model based on hyperparameters
layers = [nn.Linear(100, 256), nn.ReLU(), nn.Dropout(dropout_rate)]
for i in range(hidden_layers - 1):
    layers.extend([nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout_rate)])
layers.append(nn.Linear(256, 10))

model = nn.Sequential(*layers).cuda()

# Training setup
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
model.train()
for epoch in range(20):  # Quick training for optimization
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 5 == 0:
        print(f'Epoch {{epoch}}, Avg Loss: {{total_loss/len(train_loader):.4f}}')

# Validation
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    val_predictions = val_outputs.argmax(1)
    accuracy = (val_predictions == y_val).float().mean().item()

result = {{
    'accuracy': accuracy,
    'final_loss': total_loss / len(train_loader),
    'hyperparameters': {{
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'hidden_layers': hidden_layers,
        'dropout': dropout_rate
    }},
    'model_size': sum(p.numel() for p in model.parameters())
}}
"""
            
            # Execute trial on remote GPU
            try:
                result = self.client.execute_code_sync(
                    optimization_code,
                    timeout_seconds=600  # 10 minutes per trial
                )
                
                if result['execution_result']['success']:
                    trial_result = result['execution_result']['result']
                    print(f"Trial {trial.number}: Accuracy = {trial_result['accuracy']:.4f}")
                    return trial_result['accuracy']
                else:
                    print(f"Trial {trial.number} failed: {result['execution_result']['error']}")
                    return 0.0
                    
            except Exception as e:
                print(f"Trial {trial.number} error: {str(e)}")
                return 0.0
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_accuracy': study.best_value,
            'n_trials': len(study.trials)
        }

# Usage
if __name__ == "__main__":
    optimizer = GPUHyperparameterOptimizer()
    
    print("Starting hyperparameter optimization on remote GPU...")
    results = optimizer.optimize_model(n_trials=25)
    
    print("\n🎯 Optimization Results:")
    print(f"Best Accuracy: {results['best_accuracy']:.4f}")
    print(f"Best Parameters: {json.dumps(results['best_params'], indent=2)}")
    print(f"Total Trials: {results['n_trials']}")
```

### Example 3: Data Processing Pipeline

```python
# data_processor.py
import sys
sys.path.append('./gpu-proxy')

from src.runpod.client import RunPodClient
import pandas as pd
import numpy as np

class GPUDataProcessor:
    def __init__(self):
        self.client = RunPodClient()
    
    def process_large_dataset(self, csv_path, processing_config):
        """Process large datasets using GPU acceleration"""
        
        # Read data locally (or could be from cloud storage)
        df = pd.read_csv(csv_path)
        
        processing_code = f"""
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn.functional as F

# Get data and config from context
data = pd.DataFrame(context['data'])
config = context['config']

print(f"Processing {{len(data)}} rows with GPU acceleration...")

# Convert to GPU tensors
numeric_cols = data.select_dtypes(include=[np.number]).columns
numeric_data = torch.tensor(data[numeric_cols].values, dtype=torch.float32).cuda()

# GPU-accelerated preprocessing
processed_data = {{}}

# Normalization on GPU
if config.get('normalize', True):
    mean = numeric_data.mean(dim=0)
    std = numeric_data.std(dim=0)
    normalized_data = (numeric_data - mean) / (std + 1e-8)
    processed_data['normalized'] = normalized_data.cpu().numpy()

# Feature engineering on GPU
if config.get('create_interactions', False):
    # Create polynomial features
    poly_features = torch.cat([
        numeric_data,
        numeric_data ** 2,
        numeric_data[:, :5] * numeric_data[:, 5:10]  # Cross terms
    ], dim=1)
    processed_data['polynomial_features'] = poly_features.cpu().numpy()

# Outlier detection using GPU
if config.get('detect_outliers', False):
    # Z-score based outlier detection
    z_scores = torch.abs((numeric_data - numeric_data.mean(dim=0)) / numeric_data.std(dim=0))
    outliers = (z_scores > 3).any(dim=1)
    processed_data['outlier_mask'] = outliers.cpu().numpy()
    processed_data['clean_data'] = numeric_data[~outliers].cpu().numpy()

# PCA-like dimensionality reduction on GPU
if config.get('reduce_dimensions', False):
    # Simple dimensionality reduction (not full PCA, but GPU-accelerated)
    U, S, V = torch.svd(normalized_data)
    reduced_data = torch.mm(normalized_data, V[:, :config.get('target_dims', 10)])
    processed_data['reduced_dimensions'] = reduced_data.cpu().numpy()

result = {{
    'processed_data': processed_data,
    'original_shape': list(numeric_data.shape),
    'processing_time': 'GPU-accelerated',
    'gpu_used': torch.cuda.get_device_name(0),
    'memory_used': torch.cuda.memory_allocated(0)
}}
"""
        
        # Execute processing on remote GPU
        result = self.client.execute_code_sync(
            processing_code,
            context={
                'data': df.to_dict('records'),
                'config': processing_config
            },
            timeout_seconds=900  # 15 minutes for large datasets
        )
        
        if result['execution_result']['success']:
            return result['execution_result']['result']
        else:
            raise RuntimeError(f"Processing failed: {result['execution_result']['error']}")

# Usage
if __name__ == "__main__":
    processor = GPUDataProcessor()
    
    # Configuration for processing
    config = {
        'normalize': True,
        'create_interactions': True,
        'detect_outliers': True,
        'reduce_dimensions': True,
        'target_dims': 20
    }
    
    # Process your dataset
    results = processor.process_large_dataset('your_dataset.csv', config)
    
    print(f"Processing completed on {results['gpu_used']}")
    print(f"Original shape: {results['original_shape']}")
    print(f"Available processed data: {list(results['processed_data'].keys())}")
    
    # Access processed data
    normalized_data = results['processed_data']['normalized']
    clean_data = results['processed_data']['clean_data']
    reduced_data = results['processed_data']['reduced_dimensions']
    
    print(f"Normalized data shape: {normalized_data.shape}")
    print(f"Clean data shape: {clean_data.shape}")
    print(f"Reduced data shape: {reduced_data.shape}")
```

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

## 📋 Environment Configuration

### GPU Proxy Configuration

Create a `.env` file in the `gpu-proxy` directory (not your project directory):

```bash
# gpu-proxy/.env
RUNPOD_API_KEY=your_api_key_here
RUNPOD_ENDPOINT_ID=your_endpoint_id_here
DOCKER_HUB_USERNAME=thebuleganteng
```

### Your Project Configuration

Your project can have its own `.env` file for other configurations:

```bash
# your-project/.env
DATABASE_URL=your_database_url
API_KEY=your_api_key
PYTHONPATH=./gpu-proxy:$PYTHONPATH
```

### Requirements Management

Add to your project's `requirements.txt`:

```txt
# Your existing requirements
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0

# GPU Proxy dependencies (if not using git clone)
requests>=2.28.0
python-dotenv>=0.19.0
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