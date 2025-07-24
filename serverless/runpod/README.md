# GPU Proxy - RunPod Serverless Training Function

This directory contains the serverless function that executes ML training on RunPod's GPU infrastructure. This is the final component that makes the GPU proxy system fully operational.

## 🎯 What This Function Does

- **Receives training configurations** from the GPU proxy client
- **Executes ML training** on GPU using TensorFlow or PyTorch
- **Returns comprehensive results** including metrics, model info, and training history
- **Handles errors gracefully** with detailed logging and error reporting

## 📁 Files Overview

```
serverless/runpod/
├── handler.py          # Main serverless function handler
├── utils.py            # Training utilities and helpers
├── Dockerfile          # Container definition for RunPod
├── requirements.txt    # Python dependencies
├── deploy.sh          # Deployment automation script
└── README.md          # This file
```

## 🚀 Quick Deployment

### Prerequisites
- Docker installed and running
- Docker Hub account
- RunPod account

### Step 1: Build and Deploy
```bash
cd serverless/runpod

# Make deployment script executable
chmod +x deploy.sh

# Run deployment (will prompt for Docker Hub username)
./deploy.sh
```

### Step 2: Create RunPod Endpoint
1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Name**: `gpu-proxy-training`
   - **Docker Image**: `yourusername/gpu-proxy-training:latest`
   - **Container Start Command**: `python handler.py`
   - **GPU Type**: RTX 4090 (recommended) or A100
   - **Max Workers**: 1-3
   - **Timeout**: 300 seconds
   - **Max Requests per Worker**: 1

### Step 3: Test Your Deployment
```bash
# Update your .env with the new endpoint ID
echo "RUNPOD_ENDPOINT_ID=your_new_endpoint_id" >> ../../.env

# Test the complete system
python ../../examples/test_runpod.py
```

## 🔧 Function Capabilities

### Supported Frameworks
- **TensorFlow**: Full GPU training with Keras models
- **PyTorch**: Complete GPU training with custom models
- **Generic/Test**: Simulated training for development

### Model Types Supported
- **CNNs**: Convolutional neural networks for image tasks
- **MLPs**: Multi-layer perceptrons for tabular data
- **Custom architectures**: Defined via configuration

### Training Features
- **Real-time progress tracking**: Epoch-by-epoch metrics
- **GPU memory monitoring**: Usage and optimization
- **Early stopping**: Prevent overfitting
- **Synthetic data generation**: For testing and demos
- **Comprehensive error handling**: Detailed failure reporting

## 📊 Function Interface

### Input Format
```json
{
  "input": {
    "training_config": "base64_encoded_config",
    "framework": "tensorflow|pytorch|test",
    "max_epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

### Output Format
```json
{
  "success": true,
  "training_result": "base64_encoded_result",
  "execution_time": 45.2,
  "gpu_info": {
    "gpu_available": true,
    "gpu_count": 1,
    "memory_info": {...}
  },
  "framework_used": "tensorflow"
}
```

### Training Result Contents
```python
{
    'success': True,
    'framework': 'tensorflow',
    'final_loss': 0.234,
    'final_accuracy': 0.892,
    'training_history': {
        'loss': [2.1, 1.8, 1.2, 0.8, 0.234],
        'accuracy': [0.1, 0.3, 0.6, 0.8, 0.892],
        'val_loss': [...],
        'val_accuracy': [...]
    },
    'model_params': 1250000,
    'epochs_completed': 5
}
```

## 🧪 Local Testing

### Test the Handler Locally
```bash
# Create test configuration
python -c "
import pickle, base64
config = {
    'framework': 'tensorflow',
    'max_epochs': 2,
    'batch_size': 16,
    'model_config': {'type': 'simple_cnn'},
    'data_config': {'input_shape': [32, 32, 3], 'num_classes': 10}
}
encoded = base64.b64encode(pickle.dumps(config)).decode('utf-8')
print(f'Encoded config: {encoded[:50]}...')
"

# Test handler directly
python -c "
from handler import handler
result = handler({
    'input': {
        'training_config': 'YOUR_ENCODED_CONFIG_HERE',
        'framework': 'test'
    }
})
print(result)
"
```

### Test with Docker
```bash
# Build image
docker build -t gpu-proxy-training .

# Run test
docker run --rm gpu-proxy-training python -c "
from handler import handler
import json
result = handler({'input': {'framework': 'test', 'max_epochs': 1}})
print(json.dumps(result, indent=2))
"
```

## ⚡ Performance Optimization

### Cold Start Optimization
- **Base image**: Uses RunPod's optimized PyTorch image
- **Layer caching**: Requirements installed before code copy
- **Minimal dependencies**: Only essential packages included

### Training Optimization
- **GPU memory growth**: TensorFlow configured for dynamic allocation
- **Mixed precision**: Automatic optimization for supported GPUs
- **Efficient data loading**: Optimized batch processing

### Cost Optimization
- **Fast execution**: Minimal overhead, quick training
- **Resource cleanup**: Proper memory management
- **Timeout protection**: Prevents runaway jobs

## 🐛 Troubleshooting

### Common Issues

**1. "TensorFlow not available"**
- Check Dockerfile includes TensorFlow GPU
- Verify CUDA compatibility
- Ensure GPU drivers in base image

**2. "Training fails with memory error"**
- Reduce batch size in configuration
- Use smaller model architectures
- Check GPU memory limits

**3. "Serialization errors"**
- Verify training config format
- Check pickle compatibility
- Validate base64 encoding

**4. "Container fails to start"**
- Check Dockerfile syntax
- Verify all dependencies in requirements.txt
- Test locally with Docker

### Debug Mode
Enable verbose logging by setting environment variable:
```bash
export TF_CPP_MIN_LOG_LEVEL=0  # TensorFlow debug
export PYTORCH_LOGGING=DEBUG   # PyTorch debug
```

### Health Checks
The function includes built-in health monitoring:
- GPU availability detection
- Memory usage tracking
- Framework compatibility checks
- Training progress validation

## 📈 Monitoring & Metrics

### Built-in Metrics
- **Training progress**: Loss and accuracy per epoch
- **GPU utilization**: Memory and compute usage
- **Execution time**: Total and per-epoch timing
- **Model complexity**: Parameter counts and architecture

### Custom Metrics
Add custom metrics by modifying `utils.py`:
```python
def calculate_custom_metrics(history):
    # Your custom metric calculations
    return custom_metrics
```

## 🔄 Updates & Maintenance

### Updating the Function
1. Modify code in `handler.py` or `utils.py`
2. Run `./deploy.sh` to rebuild and redeploy
3. RunPod will automatically use the new version

### Version Management
- Tag Docker images with versions: `gpu-proxy-training:v1.1`
- Keep previous versions for rollback capability
- Test thoroughly before production deployment

### Framework Updates
Update ML frameworks in `Dockerfile`:
```dockerfile
RUN pip install --no-cache-dir \
    tensorflow-gpu==2.14.0 \  # Update version
    torch==2.1.0              # Update version
```

## 💰 Cost Optimization Tips

1. **Choose appropriate GPU**: RTX 4090 for most tasks, A100 for large models
2. **Optimize batch sizes**: Larger batches = better GPU utilization
3. **Use mixed precision**: Faster training with newer GPUs
4. **Set reasonable timeouts**: Prevent runaway costs
5. **Monitor usage**: Track costs in RunPod dashboard

## 🤝 Contributing

To improve the serverless function:

1. **Add new frameworks**: Extend handler for JAX, MLX, etc.
2. **Optimize performance**: Improve training speed and memory usage
3. **Add features**: Custom optimizers, advanced metrics, etc.
4. **Improve monitoring**: Better logging and diagnostics

## 📚 Additional Resources

- [RunPod Serverless Documentation](https://docs.runpod.io/serverless/endpoints)
- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [PyTorch CUDA Guide](https://pytorch.org/docs/stable/cuda.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

**Status**: ✅ Ready for deployment  
**GPU Support**: TensorFlow + PyTorch  
**Estimated Deploy Time**: 5-10 minutes