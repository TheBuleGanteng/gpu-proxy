"""
Basic usage example for GPU Proxy.

This example demonstrates how to use the GPU proxy with different providers,
showing the core functionality and monitoring capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path so we can import from src
current_file = Path(__file__)
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from src.core.base import RemoteGPU, TrainingConfig, GPUProviderType
from src.utils.logger import logger


async def progress_callback(data):
    """Callback function for training progress updates."""
    logger.info(f"Training Progress - Epoch {data['epoch']}/{data['total_epochs']} "
               f"({data['progress']*100:.1f}%) - Loss: {data['loss']:.4f}, "
               f"Accuracy: {data['accuracy']:.4f}")


async def error_callback(data):
    """Callback function for error notifications."""
    logger.error(f"Training Error: {data}")


async def completion_callback(data):
    """Callback function for training completion."""
    if data['success']:
        logger.info(f"Training Completed Successfully! Final accuracy: {data['final_accuracy']:.4f}, "
                   f"Duration: {data['duration']:.1f}s")
    else:
        logger.error("Training Failed!")


async def basic_example():
    """Basic example showing simple GPU proxy usage."""
    logger.info("=== Basic GPU Proxy Example ===")
    
    # Create GPU proxy with mock provider
    gpu = RemoteGPU(provider=GPUProviderType.MOCK)
    
    try:
        # Connect to GPU provider
        logger.info("Connecting to GPU provider...")
        connected = await gpu.connect()
        
        if not connected:
            logger.error("Failed to connect to GPU provider")
            return
        
        # Get available resources
        logger.info("Checking available resources...")
        resources = await gpu.get_available_resources()
        
        logger.info(f"Found {len(resources)} GPU resources:")
        for resource in resources:
            logger.info(f"  - {resource.resource_type.value}: {resource.memory_gb}GB, "
                       f"${resource.cost_per_hour:.2f}/hour, Available: {resource.availability}")
        
        # Create training configuration
        training_config = TrainingConfig(
            model_config={
                "type": "cnn",
                "layers": [
                    {"type": "conv2d", "filters": 32, "kernel_size": 3},
                    {"type": "conv2d", "filters": 64, "kernel_size": 3},
                    {"type": "dense", "units": 128},
                    {"type": "dense", "units": 10, "activation": "softmax"}
                ]
            },
            training_params={
                "optimizer": "adam",
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy"]
            },
            data_config={
                "dataset": "cifar10",
                "input_shape": [32, 32, 3],
                "num_classes": 10
            },
            framework="tensorflow",
            max_epochs=5,
            batch_size=32,
            learning_rate=0.001
        )
        
        # Estimate cost
        logger.info("Estimating training cost...")
        cost_estimate = await gpu.estimate_cost(training_config)
        logger.info(f"Estimated cost: ${cost_estimate.total_estimated_cost:.4f} "
                   f"for {cost_estimate.estimated_duration_seconds:.1f} seconds")
        
        # Add callbacks for monitoring
        gpu.add_progress_callback(progress_callback)
        gpu.add_error_callback(error_callback)
        
        # Train model
        logger.info("Starting model training...")
        result = await gpu.train_model(training_config)
        
        # Process results
        if result.success:
            logger.info("Training completed successfully!")
            logger.info(f"Final metrics - Loss: {result.final_loss:.4f}, "
                       f"Accuracy: {result.final_accuracy:.4f}")
            logger.info(f"Training duration: {result.duration_seconds:.1f} seconds")
            
            if result.cost_info:
                logger.info(f"Actual cost: ${result.cost_info.total_estimated_cost:.4f}")
            
            if result.memory_usage:
                logger.info(f"Peak memory usage: {result.memory_usage.allocated_mb:.1f}MB "
                           f"({result.memory_usage.utilization_percent:.1f}%)")
        else:
            logger.error(f"Training failed: {result.error_message}")
    
    except Exception as e:
        logger.error(f"Example failed: {e}")
    
    finally:
        # Clean up
        await gpu.disconnect()
        logger.info("Disconnected from GPU provider")


async def context_manager_example():
    """Example showing async context manager usage."""
    logger.info("\n=== Context Manager Example ===")
    
    # Configure mock provider with faster training for demo
    mock_config = {
        'training_speed_multiplier': 0.2,  # 5x faster
        'simulate_failures': False
    }
    
    async with RemoteGPU(provider="mock", config=mock_config) as gpu:
        logger.info("Connected using context manager")
        
        # Simple training configuration
        config = TrainingConfig(
            model_config={"type": "simple_nn", "layers": 3},
            training_params={"optimizer": "sgd"},
            data_config={"dataset": "mnist"},
            framework="tensorflow",
            max_epochs=3,
            batch_size=64
        )
        
        # Add progress callback
        gpu.add_progress_callback(lambda data: logger.info(
            f"Quick training - Epoch {data['epoch']}: {data['accuracy']:.3f} accuracy"
        ))
        
        result = await gpu.train_model(config)
        
        if result.success:
            logger.info(f"Quick training completed: {result.final_accuracy:.3f} accuracy")
        else:
            logger.error(f"Quick training failed: {result.error_message}")
    
    logger.info("Context manager automatically disconnected")


async def failure_simulation_example():
    """Example showing failure handling."""
    logger.info("\n=== Failure Simulation Example ===")
    
    # Configure mock to simulate failures
    failure_config = {
        'simulate_failures': True,
        'failure_rate': 0.8,  # High failure rate for demo
        'training_speed_multiplier': 0.1  # Very fast for testing
    }
    
    gpu = RemoteGPU(provider="mock", config=failure_config)
    
    try:
        # Try multiple connection attempts
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Connection attempt {attempt}/{max_attempts}")
            
            if await gpu.connect():
                logger.info("Connection successful!")
                break
            else:
                logger.warning(f"Connection attempt {attempt} failed")
                
                if attempt < max_attempts:
                    logger.info("Retrying in 1 second...")
                    await asyncio.sleep(1)
                else:
                    logger.error("All connection attempts failed")
                    return
        
        # Try training with potential failures
        config = TrainingConfig(
            model_config={"type": "test_model"},
            training_params={},
            data_config={"dataset": "test"},
            framework="tensorflow",
            max_epochs=2
        )
        
        for attempt in range(1, 4):
            logger.info(f"Training attempt {attempt}")
            
            result = await gpu.train_model(config)
            
            if result.success:
                logger.info("Training succeeded despite failure simulation!")
                break
            else:
                logger.warning(f"Training attempt {attempt} failed: {result.error_message}")
                
                if attempt < 3:
                    await asyncio.sleep(0.5)
    
    except Exception as e:
        logger.error(f"Failure simulation example error: {e}")
    
    finally:
        await gpu.disconnect()


async def multiple_providers_example():
    """Example showing how to work with multiple providers."""
    logger.info("\n=== Multiple Providers Example ===")
    
    providers_to_test = [
        ("mock", {"training_speed_multiplier": 0.1}),
        # ("local", {}),  # Would be enabled when local provider is implemented
        # ("runpod_serverless", {"api_key": "your_key"})  # Future implementation
    ]
    
    training_config = TrainingConfig(
        model_config={"type": "benchmark_model", "complexity": "simple"},
        training_params={"optimizer": "adam"},
        data_config={"dataset": "synthetic", "size": "small"},
        framework="tensorflow",
        max_epochs=2,
        batch_size=32
    )
    
    results = {}
    
    for provider_name, provider_config in providers_to_test:
        logger.info(f"Testing provider: {provider_name}")
        
        try:
            gpu = RemoteGPU(provider=provider_name, config=provider_config)
            
            # Quick benchmark
            start_time = asyncio.get_event_loop().time()
            
            await gpu.connect()
            result = await gpu.train_model(training_config)
            await gpu.disconnect()
            
            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time
            
            results[provider_name] = {
                'success': result.success,
                'accuracy': result.final_accuracy if result.success else None,
                'total_time': total_time,
                'training_time': result.duration_seconds if result.success else None
            }
            
            logger.info(f"{provider_name} results: Success={result.success}, "
                       f"Accuracy={result.final_accuracy:.3f if result.success else 'N/A'}, "
                       f"Time={total_time:.1f}s")
        
        except Exception as e:
            logger.error(f"Provider {provider_name} failed: {e}")
            results[provider_name] = {'success': False, 'error': str(e)}
    
    # Summary
    logger.info("\n=== Provider Comparison Summary ===")
    for provider, result in results.items():
        if result['success']:
            logger.info(f"{provider}: ✓ Success - {result['accuracy']:.3f} accuracy in {result['total_time']:.1f}s")
        else:
            logger.info(f"{provider}: ✗ Failed - {result.get('error', 'Unknown error')}")


async def advanced_monitoring_example():
    """Example showing advanced monitoring capabilities."""
    logger.info("\n=== Advanced Monitoring Example ===")
    
    # Tracking variables
    epoch_times = []
    memory_snapshots = []
    
    def detailed_progress_callback(data):
        """Detailed progress tracking with timing."""
        epoch_times.append(asyncio.get_event_loop().time())
        logger.info(f"📊 Epoch {data['epoch']}/{data['total_epochs']} - "
                   f"Loss: {data['loss']:.4f} → {data['val_loss']:.4f} (val), "
                   f"Acc: {data['accuracy']:.4f} → {data['val_accuracy']:.4f} (val)")
    
    def memory_callback(data):
        """Track memory usage."""
        if 'memory_info' in data:
            memory_snapshots.append(data['memory_info'])
    
    # Configure GPU with monitoring
    gpu = RemoteGPU(provider="mock", config={'training_speed_multiplier': 0.3})
    
    try:
        await gpu.connect()
        
        # Add multiple callbacks
        gpu.add_progress_callback(detailed_progress_callback)
        gpu.add_progress_callback(memory_callback)
        
        # More complex training configuration
        config = TrainingConfig(
            model_config={
                "type": "resnet",
                "layers": [
                    {"type": "conv2d", "filters": 64, "kernel_size": 7},
                    {"type": "batch_norm"},
                    {"type": "residual_block", "filters": 64, "blocks": 2},
                    {"type": "residual_block", "filters": 128, "blocks": 2},
                    {"type": "global_avg_pool"},
                    {"type": "dense", "units": 1000, "activation": "softmax"}
                ]
            },
            training_params={
                "optimizer": "adam",
                "learning_rate_schedule": "cosine_decay",
                "weight_decay": 0.0001
            },
            data_config={
                "dataset": "imagenet_subset",
                "input_shape": [224, 224, 3],
                "num_classes": 1000,
                "augmentation": True
            },
            framework="tensorflow",
            max_epochs=6,
            batch_size=16,
            learning_rate=0.001
        )
        
        logger.info("Starting advanced training with monitoring...")
        result = await gpu.train_model(config)
        
        # Analyze results
        if result.success:
            logger.info("\n📈 Training Analysis:")
            logger.info(f"Final Performance: {result.final_accuracy:.4f} accuracy")
            
            if len(epoch_times) > 1:
                avg_epoch_time = (epoch_times[-1] - epoch_times[0]) / (len(epoch_times) - 1)
                logger.info(f"Average epoch time: {avg_epoch_time:.2f}s")
            
            # Training curve analysis
            if result.training_history:
                losses = result.training_history.get('loss', [])
                accuracies = result.training_history.get('accuracy', [])
                
                if len(losses) >= 2:
                    loss_improvement = losses[0] - losses[-1]
                    acc_improvement = accuracies[-1] - accuracies[0] if accuracies else 0
                    
                    logger.info(f"Loss improvement: {loss_improvement:.4f}")
                    logger.info(f"Accuracy improvement: {acc_improvement:.4f}")
            
            # Memory analysis
            if result.memory_usage:
                logger.info(f"Peak memory usage: {result.memory_usage.allocated_mb:.1f}MB")
                logger.info(f"Memory efficiency: {result.memory_usage.utilization_percent:.1f}%")
        
    except Exception as e:
        logger.error(f"Advanced monitoring example failed: {e}")
    
    finally:
        await gpu.disconnect()


async def main():
    """Run all examples."""
    logger.info("🚀 Starting GPU Proxy Examples")
    logger.info("=" * 50)
    
    try:
        # Run examples in sequence
        await basic_example()
        await context_manager_example()
        await failure_simulation_example()
        await multiple_providers_example()
        await advanced_monitoring_example()
        
        logger.info("\n✅ All examples completed successfully!")
    
    except KeyboardInterrupt:
        logger.info("\n⏹️  Examples interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Examples failed: {e}")
    
    logger.info("=" * 50)
    logger.info("🏁 GPU Proxy Examples Finished")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())