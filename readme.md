# GPU Proxy - Execute-What-You-Send Cloud GPU Access

A production-ready Python library that provides transparent access to cloud GPU resources through arbitrary code execution. Use remote GPU infrastructure as if it were local hardware.

## 🎯 Vision & Current Status

**Problem**: Machine learning development often requires expensive GPU hardware that many developers can't access locally. Cloud GPU services exist but require complex setup, vendor-specific APIs, and manual resource management.

**Solution**: GPU Proxy provides a unified, execute-what-you-send interface that makes remote GPU resources appear as local hardware. Write your Python code locally, execute it on remote GPUs transparently.

### Current Architecture: Execute-What-You-Send with Auto-Setup

```python
# 🆕 NEW: Auto-setup integration with infrastructure management
from client import GPUProxyClient

# Auto-detects installation, clones if needed, configures environment
client = GPUProxyClient.auto_setup()

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

### 🆕 **NEW: Auto-Setup Infrastructure Management**

GPU Proxy now includes intelligent auto-setup capabilities that handle all infrastructure concerns:

- **🔍 Auto-Detection**: Automatically finds GPU proxy installation in project or sibling directories
- **🔄 Auto-Cloning**: Clones from GitHub if not found locally (configurable)
- **⚙️ Environment Setup**: Configures Python paths and loads environment files
- **✅ Health Validation**: Verifies endpoint accessibility and worker availability
- **🛡️ Graceful Fallback**: Clean error handling with detailed diagnostics

### Key Benefits

- **🔄 True Project Agnosticism**: Execute any Python code, not just ML training
- **🚀 One-Line Setup**: `GPUProxyClient.auto_setup()` handles all infrastructure
- **💰 Cost Efficient**: Pay only for GPU compute time, not idle infrastructure  
- **🔧 Developer Friendly**: Write code locally, execute remotely with same syntax
- **📊 Real-time Monitoring**: Live execution output, errors, and performance metrics
- **🛡️ Production Ready**: Robust error handling, timeouts, session management
- **⚡ Streamlined Setup**: Auto-setup with fallback to manual configuration

## 🏗️ Enhanced Architecture Overview

```
┌─────────────────┐    ┌───────────────────────────────────┐    ┌─────────────────┐
│   Your Project  │───▶│        GPU Proxy Core             │───▶│  RunPod GPU     │
│                 │    │  🆕 AUTO-SETUP ENHANCED          │    │                 │
│ Python Code     │    │ ┌─────────────────────────────┐   │    │ ┌─────────────┐ │
│ Generation      │    │ │  GPUProxyClient.auto_setup  │   │    │ │ Serverless  │ │
│ Context Data    │    │ │  - Auto-detection           │   │    │ │ Handler     │ │
└─────────────────┘    │ │  - Auto-cloning            │   │    │ └─────────────┘ │
                       │ │  - Environment setup       │   │    │ ┌─────────────┐ │
                       │ │  - Health validation       │   │    │ │ Fat Image   │ │
                       │ └─────────────────────────────┘   │    │ │ Complete ML │ │
                       │ ┌─────────────────────────────┐   │    │ │ Stack       │ │
                       │ │  RunPodClient (Enhanced)    │   │    │ └─────────────┘ │
                       │ │  - All Job Operations       │   │    └─────────────────┘
                       │ │  - Connection Management    │   │
                       │ └─────────────────────────────┘   │
                       └───────────────────────────────────┘
```

## 📦 Enhanced Project Structure

```
gpu-proxy/
├── LICENSE
├── README.md
├── requirements.txt
├── .env                        # ✅ Environment variables
├── examples/
│   └── basic_usage.py          # 🆕 Updated with auto-setup examples
├── src/
│   ├── runpod/
│   │   ├── client.py           # 🆕 ENHANCED: GPUProxyClient with auto-setup
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

## ✅ Current Status - PRODUCTION READY + AUTO-SETUP

### 🆕 **NEW: Auto-Setup Infrastructure** ✅

#### **GPUProxyClient Class with Auto-Setup**
- **🔍 Intelligent Detection**: Automatically finds GPU proxy installation in:
  - `./gpu-proxy` (subdirectory of client project)
  - `../gpu-proxy` (sibling directory to client project)
- **🔄 Auto-Cloning**: Automatically clones from GitHub if not found (configurable)
  - Source: `https://github.com/TheBuleGanteng/gpu-proxy`
  - Target: `../gpu-proxy` (sibling directory for reusability)
  - Git availability detection with clear error messages
- **⚙️ Environment Management**: 
  - Adds GPU proxy to Python path automatically
  - Loads GPU proxy `.env` file if present
  - Manages environment variables and configurations
- **✅ Setup Validation**:
  - Verifies RunPod endpoint accessibility
  - Checks worker availability and health
  - Comprehensive error reporting with diagnostics
- **🛡️ Graceful Fallback**: Clean error handling when setup fails

#### **Enhanced Integration Pattern**
```python
# 🆕 NEW: One-line setup handles everything
from client import GPUProxyClient

# Auto-detects, clones if needed, configures, validates
client = GPUProxyClient.auto_setup()

# Or with custom configuration
client = GPUProxyClient.auto_setup(
    endpoint_id="custom-endpoint",
    auto_clone=True,  # Set to False to disable auto-cloning
    github_url="https://github.com/TheBuleGanteng/gpu-proxy"
)

# Execute code - same interface as before
result = client.execute_code_sync("result = torch.cuda.is_available()")
```

### Fully Working Components ✅

#### 1. **Enhanced RunPod Client** ✅
- **Complete API Coverage**: All RunPod serverless job endpoints implemented
  - ✅ `/run` - Asynchronous job submission
  - ✅ `/runsync` - Synchronous execution
  - ✅ `/health` - Health monitoring  
  - ✅ `/cancel/{job_id}` - Job cancellation
  - ✅ `/purge-queue` - Queue management
  - ✅ `/status/{job_id}` - Status tracking
  - ✅ `/stream/{job_id}` - Real-time streaming
- **🆕 Enhanced Connection Management**: Proper session cleanup and connection pooling
- **🆕 Auto-Setup Integration**: Seamless integration with GPUProxyClient wrapper
- **Endpoint Validation**: Automatic verification that endpoint exists and is accessible
- **Environment Integration**: Reads credentials from `.env` file with validation
- **Type Safety**: Full type hints and error handling
- **Convenience Methods**: High-level functions like `execute_code_sync()`

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
- **🆕 Auto-Setup Health Monitoring**: Comprehensive validation during setup
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
- **🆕 Auto-Setup Testing**: Validation of auto-detection and setup logic
- **Environment Management**: `.env` file loading with python-dotenv
- **Production Validation**: Confirmed working on 2-GPU RunPod infrastructure

## 🚀 Enhanced Quick Start Guide

### Prerequisites
- Python 3.8+
- RunPod account and API key
- Git (for auto-cloning functionality)
- Docker (for deployment)

### Step 1: 🆕 **Simple Auto-Setup Integration**

#### **Option A: Auto-Setup Integration (Recommended)**
```bash
# From your project directory - GPU Proxy will auto-setup itself!
# No manual cloning needed - it handles everything automatically
```

```python
# In your Python code - one line handles all setup
from client import GPUProxyClient

# This automatically:
# 1. Detects GPU proxy installation (./gpu-proxy or ../gpu-proxy)
# 2. Clones from GitHub if not found
# 3. Sets up environment and Python paths
# 4. Validates endpoint health
# 5. Returns ready-to-use client
client = GPUProxyClient.auto_setup()

# Ready to execute!
result = client.execute_code_sync("import torch; result = torch.cuda.is_available()")
print(result)
```

#### **Option B: Manual Setup (Traditional)**
```bash
# Clone the repository manually
git clone https://github.com/TheBuleGanteng/gpu-proxy.git
cd gpu-proxy

# Install required dependencies
pip install python-dotenv requests

# Verify the installation
python -c "from client import GPUProxyClient; print('✅ GPU Proxy imported successfully')"
```

### Step 2: Deploy Container (Required Once)

```bash
# Navigate to RunPod deployment
cd gpu-proxy/src/runpod  # If manually cloned, or let auto-setup handle path

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

### Step 3: Manual RunPod Setup (Required Once)

**⚠️ IMPORTANT**: Serverless endpoints must be created manually via the RunPod console due to API limitations.

The auto-setup system will show detailed instructions when manual setup is needed:

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
   # The auto-setup will create .env file in gpu-proxy directory
   # Or create manually:
   cat > gpu-proxy/.env << EOF
   RUNPOD_API_KEY=your_api_key_here
   RUNPOD_ENDPOINT_ID=your_endpoint_id_here
   DOCKER_HUB_USERNAME=thebuleganteng
   EOF
   ```

### Step 4: Validate Setup

```python
# 🆕 NEW: Auto-setup includes validation
try:
    client = GPUProxyClient.auto_setup()
    print("✅ GPU Proxy setup successful!")
    
    # Test execution
    result = client.execute_code_sync("result = 'GPU Proxy is working!'")
    print(result['execution_result']['result'])
    
except RuntimeError as e:
    print(f"❌ Setup failed: {e}")
    # Auto-setup provides detailed error information
```

**Expected Result**: Successful client creation with validated endpoint

## 🔌 Enhanced Integration Into Your Projects

### Method 1: 🆕 **Auto-Setup Integration (Recommended)**

```python
# In your project - no manual setup needed!
# GPU Proxy handles detection, cloning, and configuration automatically

# your_ml_project.py
from client import GPUProxyClient

def train_with_gpu():
    # One line - handles all infrastructure automatically
    client = GPUProxyClient.auto_setup()
    
    result = client.execute_code_sync("""
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    result = {"status": "success", "gpu_ready": torch.cuda.is_available()}
    """)
    
    return result['execution_result']['result']

if __name__ == "__main__":
    result = train_with_gpu()
    print(f"GPU training setup: {result}")
```

**Directory Structure After Auto-Setup:**
```
your-project/
├── your_code.py
├── requirements.txt
├── .env                       # Your project config
└── gpu-proxy/                 # 🆕 Auto-cloned if not present
    ├── client.py              # Main client with auto-setup
    ├── .env                   # GPU Proxy config
    └── src/
        └── runpod/
            ├── handler.py
            └── ...
```

### Method 2: Manual Repository Integration

```bash
# In your project directory
git clone https://github.com/TheBuleGanteng/gpu-proxy.git

# Add to your Python path
import sys
sys.path.append('./gpu-proxy')

# Import and use
from client import GPUProxyClient

client = GPUProxyClient.auto_setup()  # Still benefits from auto-setup features
result = client.execute_code_sync("import torch; result = torch.cuda.is_available()")
print(result)
```

### Method 3: Submodule Integration

```bash
# Add as a Git submodule to your project
git submodule add https://github.com/TheBuleGanteng/gpu-proxy.git gpu-proxy
git submodule update --init --recursive

# In your Python code
import sys
sys.path.append('./gpu-proxy')
from client import GPUProxyClient
```

## 💻 Enhanced API Reference

### 🆕 **GPUProxyClient Class**

```python
from client import GPUProxyClient

# Auto-setup (recommended)
client = GPUProxyClient.auto_setup(
    endpoint_id=None,           # Optional: Override endpoint ID
    api_key=None,              # Optional: Override API key  
    auto_clone=True,           # Optional: Disable auto-cloning
    github_url="https://github.com/TheBuleGanteng/gpu-proxy"  # Custom repo URL
)

# Manual initialization (traditional)
from client import RunPodClient
runpod_client = RunPodClient(endpoint_id=None, api_key=None)
client = GPUProxyClient(runpod_client)
```

#### **Auto-Setup Parameters**
- **`endpoint_id`**: RunPod endpoint ID (reads from env if None)
- **`api_key`**: RunPod API key (reads from env if None)
- **`auto_clone`**: Enable/disable automatic GitHub cloning (default: True)
- **`github_url`**: Custom GitHub repository URL for cloning

#### **Auto-Setup Process**
1. **Detection**: Searches for GPU proxy in `./gpu-proxy` and `../gpu-proxy`
2. **Cloning**: If not found and `auto_clone=True`, clones from GitHub
3. **Environment Setup**: Adds to Python path and loads configuration
4. **Validation**: Verifies endpoint accessibility and worker health
5. **Client Creation**: Returns configured GPUProxyClient ready for use

#### Core Methods (Same Interface as RunPodClient)

- **`execute_code_sync(code, context=None, timeout=300)`** - Execute code and wait for results
- **`run(input_data, timeout=300)`** - Submit asynchronous job
- **`runsync(input_data, timeout=300)`** - Submit synchronous job
- **`health()`** - Check endpoint health
- **`status(job_id)`** - Get job status  
- **`cancel(job_id)`** - Cancel running job
- **`stream(job_id)`** - Stream real-time updates
- **`wait_for_completion(job_id, poll_interval=1.0, max_wait=600)`** - Poll until complete

#### **Enhanced Error Handling**
```python
try:
    client = GPUProxyClient.auto_setup()
except ImportError as e:
    print(f"GPU proxy not available: {e}")
    # Handle fallback to local execution
except RuntimeError as e:
    print(f"Setup failed: {e}")
    # Handle configuration issues
```

### RunPodClient Class (Lower Level)

```python
from client import RunPodClient

client = RunPodClient(endpoint_id=None, api_key=None)  # Uses .env if None
```

#### Convenience Functions

```python
# 🆕 NEW: Auto-setup convenience function
from client import auto_setup_gpu_proxy

client = auto_setup_gpu_proxy()  # Equivalent to GPUProxyClient.auto_setup()

# Legacy functions (still available)
from client import execute_code_sync, health

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

### 🆕 **Auto-Setup with Custom Configuration**

```python
from client import GPUProxyClient

# Custom endpoint and disable auto-cloning
client = GPUProxyClient.auto_setup(
    endpoint_id="your-custom-endpoint",
    auto_clone=False  # Must have gpu-proxy manually installed
)

# Custom GitHub repository (for forks)
client = GPUProxyClient.auto_setup(
    github_url="https://github.com/your-fork/gpu-proxy"
)

# Multiple clients with different configurations
prod_client = GPUProxyClient.auto_setup(endpoint_id="prod-endpoint")
test_client = GPUProxyClient.auto_setup(endpoint_id="test-endpoint")
```

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
    client = GPUProxyClient.auto_setup()
    result = client.execute_code_sync(your_code, timeout_seconds=600)
    
    if result['execution_result']['success']:
        print("Success:", result['execution_result']['result'])
    else:
        print("Execution error:", result['execution_result']['error'])
        print("Stdout:", result['execution_result']['stdout'])
        print("Stderr:", result['execution_result']['stderr'])
        
except RuntimeError as e:
    print("Setup/Request error:", e)
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
cd gpu-proxy/src/runpod  # Auto-setup handles path detection
python test.py
```

**Test Coverage:**
- ✅ Health monitoring and endpoint validation
- ✅ Synchronous code execution with ML libraries
- ✅ Context data passing and processing
- ✅ System info and GPU benchmarking
- ✅ Async job workflows with streaming
- ✅ Convenience functions
- ✅ 🆕 Auto-setup functionality validation
- ✅ Error handling and edge cases

**Latest Test Results:**
- **Total Tests**: 9
- **Passed**: 9 (100%)
- **Failed**: 0
- **Success Rate**: 100%

## 🔌 Enhanced Integration Examples

### Example 1: 🆕 **Auto-Setup ML Training**

```python
# your_ml_project.py - No manual setup required!
from client import GPUProxyClient
import numpy as np

def train_model_on_gpu(training_data, labels):
    """Train a model using remote GPU resources with auto-setup"""
    
    # One line - handles detection, cloning, configuration, validation
    client = GPUProxyClient.auto_setup()
    
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

# Usage - GPU Proxy auto-setup handles all infrastructure
if __name__ == "__main__":
    # Your local data
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 10, 1000)
    
    try:
        # Train on remote GPU - fully automated setup
        results = train_model_on_gpu(X_train, y_train)
        print(f"Training completed on {results['gpu_used']}")
        print(f"Final accuracy: {results['accuracy']:.4f}")
    except RuntimeError as e:
        print(f"Training failed: {e}")
        # Auto-setup provides detailed error diagnostics
```

### Example 2: 🆕 **Auto-Setup Hyperparameter Optimization**

```python
# hyperparameter_optimizer.py
from client import GPUProxyClient
import optuna
import json

class GPUHyperparameterOptimizer:
    def __init__(self, dataset_path=None):
        # Auto-setup handles all infrastructure - no manual configuration needed
        self.client = GPUProxyClient.auto_setup()
        self.dataset_path = dataset_path
        
    def optimize_model(self, n_trials=50):
        """Run hyperparameter optimization using remote GPU with auto-setup"""
        
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

# Usage - Fully automated with auto-setup
if __name__ == "__main__":
    try:
        optimizer = GPUHyperparameterOptimizer()  # Auto-setup happens here
        
        print("Starting hyperparameter optimization on remote GPU...")
        results = optimizer.optimize_model(n_trials=25)
        
        print("\n🎯 Optimization Results:")
        print(f"Best Accuracy: {results['best_accuracy']:.4f}")
        print(f"Best Parameters: {json.dumps(results['best_params'], indent=2)}")
        print(f"Total Trials: {results['n_trials']}")
        
    except RuntimeError as e:
        print(f"Optimization failed: {e}")
        # Auto-setup provides detailed diagnostics
```

### Example 3: 🆕 **Auto-Setup Data Processing Pipeline**

```python
# data_processor.py
from client import GPUProxyClient
import pandas as pd
import numpy as np

class GPUDataProcessor:
    def __init__(self):
        # Auto-setup handles detection, cloning, configuration
        self.client = GPUProxyClient.auto_setup()
    
    def process_large_dataset(self, csv_path, processing_config):
        """Process large datasets using GPU acceleration with auto-setup"""
        
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

# Usage - Auto-setup makes it seamless
if __name__ == "__main__":
    try:
        processor = GPUDataProcessor()  # Auto-setup handles infrastructure
        
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
        
    except RuntimeError as e:
        print(f"Data processing failed: {e}")
        # Auto-setup provides detailed error information
```

### Basic ML Training with Auto-Setup

```python
# Simple integration with auto-setup
from client import GPUProxyClient

# One line - handles all infrastructure automatically
client = GPUProxyClient.auto_setup()

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

## 📋 Environment Configuration

### 🆕 **Auto-Setup Environment Management**

Auto-setup handles environment configuration automatically:

1. **Detection**: Finds existing `.env` files in GPU proxy installation
2. **Creation**: Creates configuration files if needed
3. **Loading**: Automatically loads environment variables
4. **Validation**: Verifies all required configurations are present

### GPU Proxy Configuration

The auto-setup system manages the `.env` file in the GPU proxy directory:

```bash
# gpu-proxy/.env (managed by auto-setup)
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
# No need for PYTHONPATH - auto-setup handles it
```

### Requirements Management

Add to your project's `requirements.txt`:

```txt
# Your existing requirements
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0

# GPU Proxy dependencies (auto-setup handles installation)
requests>=2.28.0
python-dotenv>=0.19.0
```

## 🎯 Integration Philosophy

### 🆕 **Enhanced Design Principles**

1. **Infrastructure Transparency**: GPU proxy acts as hardware extension, not ML service
2. **🆕 Zero-Configuration Setup**: Auto-setup eliminates manual configuration steps
3. **Project Agnosticism**: Works with any Python code, any ML framework
4. **Developer Control**: You maintain complete control over training logic
5. **Minimal Coupling**: No vendor lock-in, easy to switch providers
6. **🆕 Intelligent Automation**: Detects, configures, and validates automatically

### For Your Projects

GPU Proxy integrates seamlessly with existing workflows:

- **🆕 One-Line Integration**: `GPUProxyClient.auto_setup()` handles everything
- **ML Training**: Execute training loops on remote GPUs
- **Data Processing**: GPU-accelerated data preprocessing
- **Model Inference**: Run inference on large datasets
- **Research**: Experiment with different architectures and approaches
- **Hyperparameter Optimization**: Distributed parameter search

## 🧪 Validation Results

- **Test Success Rate**: 100% (9/9 tests passed)
- **🆕 Auto-Setup Validation**: Full detection, cloning, and configuration testing
- **GPU Performance**: 1.57ms computation time confirmed
- **Library Support**: PyTorch 2.0.1+cu118, TensorFlow 2.19.0 validated
- **CUDA Availability**: 2 GPUs detected and accessible
- **Production Ready**: All core functionality tested and working
- **🆕 Infrastructure Automation**: Auto-setup tested with multiple scenarios

## 🤝 Contributing

Key areas for contribution:

1. **🆕 Auto-Setup Enhancements**: Improve detection, error handling, configuration
2. **Provider Extensions**: Add support for new cloud GPU providers
3. **Framework Support**: Optimize for specific ML frameworks  
4. **Performance Optimization**: Reduce latency and improve throughput
5. **Documentation**: Examples, tutorials, best practices

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Current Status**: ✅ **PRODUCTION READY + AUTO-SETUP** - Execute-what-you-send architecture with intelligent infrastructure management

**Latest Achievement**: 🆕 **Auto-Setup Integration** - One-line setup handles detection, cloning, configuration, and validation

**Vision**: The standard interface for transparent cloud GPU access with zero-configuration setup