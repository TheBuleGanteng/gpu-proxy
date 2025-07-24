"""
Utility functions for RunPod serverless GPU training.
"""

import time
import logging
from typing import Dict, Any, List
import json

logger = logging.getLogger(__name__)


def monitor_gpu_memory():
    """Monitor GPU memory usage during training."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            return {
                "allocated_gb": round(allocated, 2),
                "cached_gb": round(cached, 2),
                "total_gb": round(total, 2),
                "utilization_percent": round((allocated / total) * 100, 1)
            }
    except Exception as e:
        logger.warning(f"Could not get GPU memory info: {e}")
    
    try:
        import tensorflow as tf # type: ignore
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # TensorFlow memory info is more limited
            return {
                "gpu_count": len(gpus),
                "framework": "tensorflow"
            }
    except Exception as e:
        logger.warning(f"Could not get TensorFlow GPU info: {e}")
    
    return {"gpu_available": False}


def create_synthetic_dataset(dataset_type: str, samples: int = 1000) -> Dict[str, Any]:
    """Create synthetic datasets for testing and demonstration."""
    
    if dataset_type == "cifar10":
        return {
            "input_shape": [32, 32, 3],
            "num_classes": 10,
            "samples": samples,
            "dataset_name": "cifar10_synthetic"
        }
    
    elif dataset_type == "mnist":
        return {
            "input_shape": [28, 28, 1],
            "num_classes": 10,
            "samples": samples,
            "dataset_name": "mnist_synthetic"
        }
    
    elif dataset_type == "imagenet":
        return {
            "input_shape": [224, 224, 3],
            "num_classes": 1000,
            "samples": min(samples, 10000),  # Limit for demo
            "dataset_name": "imagenet_synthetic"
        }
    
    else:
        # Default small dataset
        return {
            "input_shape": [32, 32, 3],
            "num_classes": 10,
            "samples": samples,
            "dataset_name": "default_synthetic"
        }


def validate_training_config(config: Dict[str, Any]) -> tuple[bool, str]:
    """Validate training configuration."""
    
    required_fields = ['framework', 'max_epochs']
    for field in required_fields:
        if field not in config:
            return False, f"Missing required field: {field}"
    
    # Validate framework
    framework = config.get('framework', '').lower()
    if framework not in ['tensorflow', 'pytorch', 'test', 'generic']:
        return False, f"Unsupported framework: {framework}"
    
    # Validate epochs
    max_epochs = config.get('max_epochs', 0)
    if not isinstance(max_epochs, int) or max_epochs <= 0:
        return False, "max_epochs must be a positive integer"
    
    if max_epochs > 100:
        return False, "max_epochs cannot exceed 100 for safety"
    
    # Validate batch size
    batch_size = config.get('batch_size', 32)
    if not isinstance(batch_size, int) or batch_size <= 0:
        return False, "batch_size must be a positive integer"
    
    if batch_size > 512:
        return False, "batch_size cannot exceed 512 for memory safety"
    
    # Validate learning rate
    learning_rate = config.get('learning_rate', 0.001)
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        return False, "learning_rate must be a positive number"
    
    if learning_rate > 1.0:
        return False, "learning_rate cannot exceed 1.0"
    
    return True, "Configuration valid"


def calculate_training_metrics(history: Dict[str, List[float]]) -> Dict[str, Any]:
    """Calculate additional training metrics from history."""
    
    metrics = {}
    
    if 'loss' in history and history['loss']:
        losses = history['loss']
        metrics['initial_loss'] = losses[0]
        metrics['final_loss'] = losses[-1]
        metrics['loss_improvement'] = losses[0] - losses[-1]
        metrics['loss_reduction_percent'] = ((losses[0] - losses[-1]) / losses[0]) * 100
    
    if 'accuracy' in history and history['accuracy']:
        accuracies = history['accuracy']
        metrics['initial_accuracy'] = accuracies[0]
        metrics['final_accuracy'] = accuracies[-1]
        metrics['accuracy_improvement'] = accuracies[-1] - accuracies[0]
    
    if 'val_accuracy' in history and history['val_accuracy']:
        val_accuracies = history['val_accuracy']
        metrics['final_val_accuracy'] = val_accuracies[-1]
        
        # Check for overfitting
        if 'accuracy' in history:
            train_acc = history['accuracy'][-1]
            val_acc = val_accuracies[-1]
            metrics['overfitting_gap'] = train_acc - val_acc
            metrics['likely_overfitting'] = (train_acc - val_acc) > 0.1
    
    # Training stability
    if 'loss' in history and len(history['loss']) > 3:
        losses = history['loss']
        # Calculate loss variance in last half of training
        mid_point = len(losses) // 2
        late_losses = losses[mid_point:]
        variance = sum((x - sum(late_losses)/len(late_losses))**2 for x in late_losses) / len(late_losses)
        metrics['loss_stability'] = variance
        metrics['training_stable'] = variance < 0.01
    
    return metrics


def format_training_summary(result: Dict[str, Any]) -> str:
    """Format training results into a readable summary."""
    
    if not result.get('success', False):
        return f"Training failed: {result.get('error_message', 'Unknown error')}"
    
    framework = result.get('framework', 'unknown')
    final_accuracy = result.get('final_accuracy', 0)
    epochs = result.get('epochs_completed', 0)
    params = result.get('model_params', 0)
    
    summary = f"""
Training Summary:
- Framework: {framework.title()}
- Final Accuracy: {final_accuracy:.1%}
- Epochs Completed: {epochs}
- Model Parameters: {params:,}
"""
    
    if 'training_history' in result:
        history = result['training_history']
        metrics = calculate_training_metrics(history)
        
        if 'accuracy_improvement' in metrics:
            summary += f"- Accuracy Improvement: +{metrics['accuracy_improvement']:.1%}\n"
        
        if 'loss_reduction_percent' in metrics:
            summary += f"- Loss Reduction: {metrics['loss_reduction_percent']:.1f}%\n"
        
        if 'likely_overfitting' in metrics:
            if metrics['likely_overfitting']:
                summary += f"- Overfitting Detected: {metrics['overfitting_gap']:.1%} gap\n"
            else:
                summary += "- No Overfitting Detected\n"
    
    return summary.strip()


class TrainingProgressTracker:
    """Track and log training progress in real-time."""
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.epoch_times = []
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log progress for an epoch."""
        current_time = time.time()
        
        if self.epoch_times:
            epoch_duration = current_time - self.epoch_times[-1]
        else:
            epoch_duration = current_time - self.start_time
        
        self.epoch_times.append(current_time)
        
        # Calculate estimated time remaining
        avg_epoch_time = (current_time - self.start_time) / (epoch + 1)
        remaining_epochs = self.total_epochs - (epoch + 1)
        eta_seconds = remaining_epochs * avg_epoch_time
        
        # Format metrics
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        logger.info(f"Epoch {epoch + 1}/{self.total_epochs} ({epoch_duration:.1f}s) - "
                   f"{metrics_str} - ETA: {eta_seconds:.0f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        total_time = time.time() - self.start_time
        
        return {
            "total_training_time": total_time,
            "average_epoch_time": total_time / len(self.epoch_times) if self.epoch_times else 0,
            "epochs_completed": len(self.epoch_times)
        }
        
def run_training_loop(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified training loop that handles validation, logging, and model dispatch.
    """
    import traceback

    try:
        logger.info("🚀 Starting training with config:")
        logger.info(json.dumps(input_data, indent=2))

        # Validate config
        is_valid, msg = validate_training_config(input_data)
        if not is_valid:
            raise ValueError(f"Invalid training config: {msg}")

        framework = input_data["framework"].lower()

        # Simulate a model training entry point
        if framework == "test":
            time.sleep(2)
            return {
                "success": True,
                "framework": "test",
                "final_accuracy": 1.0,
                "epochs_completed": 1,
                "model_params": 42
            }

        # Placeholder for other implementations
        raise NotImplementedError(f"Framework '{framework}' not yet implemented.")

    except Exception as e:
        logger.error("❌ Training failed:")
        logger.error(traceback.format_exc())
        return {"success": False, "error_message": str(e)}