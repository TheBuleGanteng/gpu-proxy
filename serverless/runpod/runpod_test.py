"""
Test script for RunPod Serverless GPU integration
Validates basic functionality, GPU verification, and performance characteristics
"""

import numpy as np
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
current_file = Path(__file__)
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import logger

class RunPodServerlessClient:
    """Client for RunPod Serverless endpoints"""
    
    def __init__(self, endpoint_id: str, api_key: Optional[str] = None, timeout: int = 300):
        """
        Initialize RunPod Serverless client
        
        Args:
            endpoint_id: RunPod endpoint ID
            api_key: RunPod API key
            timeout: Request timeout in seconds
        """
        logger.debug("running __init__ ... initializing RunPod serverless client")
        
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def submit_job(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Submit a job to the serverless endpoint
        
        Args:
            payload: Job payload
            
        Returns:
            Job submission response or None if failed
        """
        logger.debug("running submit_job ... submitting job to serverless endpoint")
        
        try:
            response = requests.post(
                f"{self.base_url}/run",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.debug(f"running submit_job ... HTTP error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.debug(f"running submit_job ... error: {str(e)}")
            return None
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status
        
        Args:
            job_id: Job ID to check
            
        Returns:
            Job status response or None if failed
        """
        logger.debug(f"running get_job_status ... checking status for job {job_id}")
        
        try:
            response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.debug(f"running get_job_status ... HTTP error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.debug(f"running get_job_status ... error: {str(e)}")
            return None
    
    def wait_for_completion(self, job_id: str, max_wait: int = 300, poll_interval: int = 5) -> Optional[Dict[str, Any]]:
        """
        Wait for job completion
        
        Args:
            job_id: Job ID to wait for
            max_wait: Maximum wait time in seconds
            poll_interval: Polling interval in seconds
            
        Returns:
            Final job result or None if failed/timeout
        """
        logger.debug(f"running wait_for_completion ... waiting for job {job_id}")
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = self.get_job_status(job_id)
            
            if not status_response:
                logger.debug("running wait_for_completion ... failed to get status")
                return None
            
            status = status_response.get("status", "UNKNOWN")
            
            if status == "COMPLETED":
                logger.debug("running wait_for_completion ... job completed successfully")
                return status_response
            elif status == "FAILED":
                logger.debug("running wait_for_completion ... job failed")
                return status_response
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                # Still processing, continue waiting
                time.sleep(poll_interval)
            else:
                logger.debug(f"running wait_for_completion ... unknown status: {status}")
                return None
        
        logger.debug("running wait_for_completion ... timeout reached")
        return None

def generate_test_data(num_samples: int = 100, num_features: int = 4, num_classes: int = 2) -> tuple:
    """
    Generate synthetic test data for validation
    
    Args:
        num_samples: Number of training samples
        num_features: Number of input features
        num_classes: Number of output classes
        
    Returns:
        Tuple of (training_data, validation_data)
    """
    logger.debug(f"running generate_test_data ... generating {num_samples} samples")
    
    # Generate random features
    np.random.seed(42)
    features = np.random.randn(num_samples, num_features).astype(np.float32)
    
    # Generate labels (simple linear combination for consistency)
    weights = np.random.randn(num_features)
    linear_combo = np.dot(features, weights)
    labels = (linear_combo > np.median(linear_combo)).astype(int)
    
    # Split into train/validation
    split_idx = int(0.8 * num_samples)
    
    training_data = {
        "features": features[:split_idx].tolist(),
        "labels": labels[:split_idx].tolist()
    }
    
    validation_data = {
        "features": features[split_idx:].tolist(),
        "labels": labels[split_idx:].tolist()
    }
    
    return training_data, validation_data

def create_test_model_config() -> Dict[str, Any]:
    """
    Create a simple test model configuration
    
    Returns:
        Model configuration dictionary
    """
    logger.debug("running create_test_model_config ... creating test model config")
    
    return {
        "type": "sequential",
        "layers": [
            {"type": "linear", "in_features": 4, "out_features": 8},
            {"type": "relu"},
            {"type": "linear", "in_features": 8, "out_features": 2}
        ]
    }

def create_large_model_config() -> Dict[str, Any]:
    """
    Create a larger model configuration for performance testing
    
    Returns:
        Large model configuration dictionary
    """
    logger.debug("running create_large_model_config ... creating large model config")
    
    return {
        "type": "sequential",
        "layers": [
            {"type": "linear", "in_features": 100, "out_features": 512},
            {"type": "relu"},
            {"type": "dropout", "p": 0.3},
            {"type": "linear", "in_features": 512, "out_features": 256},
            {"type": "relu"},
            {"type": "dropout", "p": 0.3},
            {"type": "linear", "in_features": 256, "out_features": 128},
            {"type": "relu"},
            {"type": "linear", "in_features": 128, "out_features": 10}
        ]
    }

def create_test_hyperparameters() -> Dict[str, Any]:
    """
    Create test hyperparameters
    
    Returns:
        Hyperparameters dictionary
    """
    logger.debug("running create_test_hyperparameters ... creating test hyperparameters")
    
    return {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "loss_function": "cross_entropy"
    }

def test_gpu_information(client: RunPodServerlessClient) -> bool:
    """
    Test GPU information and availability
    
    Args:
        client: RunPod serverless client instance
        
    Returns:
        True if GPU is available and working
    """
    logger.debug("running test_gpu_information ... testing GPU information")
    
    print("Testing GPU information and availability...")
    
    try:
        # Submit GPU info request
        test_payload = {
            "input": {
                "operation": "gpu_info"
            }
        }
        
        start_time = time.time()
        response = client.submit_job(test_payload)
        
        if not response or "id" not in response:
            print("❌ GPU info test FAILED - job submission failed")
            return False
        
        job_id = response["id"]
        print(f"   GPU info job submitted: {job_id}")
        
        # Wait for completion
        final_result = client.wait_for_completion(job_id, max_wait=60)
        
        if not final_result:
            print("❌ GPU info test FAILED - job timeout or error")
            return False
        
        execution_time = time.time() - start_time
        
        # Check results
        if final_result.get("status") != "COMPLETED":
            print(f"❌ GPU info test FAILED - job status: {final_result.get('status')}")
            error = final_result.get("error", "Unknown error")
            print(f"   Error: {error}")
            return False
        
        output = final_result.get("output", {})
        gpu_info = output.get("gpu_info", {})
        benchmark_results = output.get("benchmark_results", {})
        
        print(f"✅ GPU information retrieved successfully!")
        print(f"   Execution time: {execution_time:.2f}s")
        
        # Display GPU information
        if gpu_info:
            print(f"\n🖥️  GPU Information:")
            print(f"   • CUDA Available: {gpu_info.get('cuda_available', 'Unknown')}")
            print(f"   • Device: {gpu_info.get('device_name', 'Unknown')}")
            print(f"   • PyTorch Version: {gpu_info.get('pytorch_version', 'Unknown')}")
            
            if gpu_info.get('cuda_available'):
                print(f"   • GPU Count: {gpu_info.get('gpu_count', 'Unknown')}")
                print(f"   • GPU Name: {gpu_info.get('gpu_name', 'Unknown')}")
                print(f"   • CUDA Version: {gpu_info.get('cuda_version', 'Unknown')}")
                print(f"   • cuDNN Version: {gpu_info.get('cudnn_version', 'Unknown')}")
                
                # Display test results
                tests = ['tensor_creation_test', 'computation_test', 'memory_test', 'neural_network_test']
                print(f"   • CUDA Tests:")
                for test in tests:
                    status = gpu_info.get(test, 'Unknown')
                    icon = "✅" if status == "PASSED" else "❌"
                    print(f"     {icon} {test.replace('_', ' ').title()}: {status}")
        
        # Display benchmark results
        if benchmark_results:
            print(f"\n⚡ Performance Benchmark:")
            print(f"   • Matrix Size: {benchmark_results.get('matrix_size', 'Unknown')}")
            print(f"   • Computation Time: {benchmark_results.get('computation_time_ms', 'Unknown'):.2f}ms")
            print(f"   • Operations/sec: {benchmark_results.get('operations_per_second', 'Unknown'):.2e}")
            performance = benchmark_results.get('performance_assessment', 'Unknown')
            if performance == "GPU_ACCELERATED":
                print(f"   🚀 Performance Assessment: GPU ACCELERATED")
            elif performance == "CPU_LIKELY":
                print(f"   🐌 Performance Assessment: CPU Processing")
            else:
                print(f"   ❓ Performance Assessment: {performance}")
        
        # Return True if GPU is available and working
        return gpu_info.get('cuda_available', False) and benchmark_results.get('benchmark_status') == 'COMPLETED'
        
    except Exception as e:
        print(f"❌ GPU info test FAILED - {str(e)}")
        logger.debug(f"running test_gpu_information ... error: {str(e)}")
        return False

def test_basic_connectivity(client: RunPodServerlessClient) -> bool:
    """
    Test basic connectivity to RunPod endpoint
    
    Args:
        client: RunPod serverless client instance
        
    Returns:
        True if connectivity test passes
    """
    logger.debug("running test_basic_connectivity ... testing endpoint connectivity")
    
    print("Testing basic connectivity...")
    
    try:
        # Submit a simple training job
        test_payload = {
            "input": {
                "operation": "train_epoch",
                "model_config": create_test_model_config(),
                "training_data": {
                    "features": [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
                    "labels": [0, 1]
                },
                "hyperparameters": create_test_hyperparameters()
            }
        }
        
        response = client.submit_job(test_payload)
        
        if response and "id" in response:
            print("✅ Connectivity test PASSED")
            print(f"   Job ID: {response['id']}")
            print(f"   Status: {response.get('status', 'Unknown')}")
            return True
        else:
            print("❌ Connectivity test FAILED - no valid response")
            return False
            
    except Exception as e:
        print(f"❌ Connectivity test FAILED - {str(e)}")
        logger.debug(f"running test_basic_connectivity ... error: {str(e)}")
        return False

def test_single_epoch_training(client: RunPodServerlessClient) -> bool:
    """
    Test single epoch training functionality
    
    Args:
        client: RunPod serverless client instance
        
    Returns:
        True if test passes
    """
    logger.debug("running test_single_epoch_training ... testing single epoch training")
    
    print("Testing single epoch training...")
    
    try:
        # Prepare test data
        training_data, _ = generate_test_data(num_samples=50)
        model_config = create_test_model_config()
        hyperparams = create_test_hyperparameters()
        
        # Create job payload
        payload = {
            "input": {
                "operation": "train_epoch",
                "model_config": model_config,
                "training_data": training_data,
                "hyperparameters": hyperparams
            }
        }
        
        # Submit job
        start_time = time.time()
        response = client.submit_job(payload)
        
        if not response or "id" not in response:
            print("❌ Single epoch training FAILED - job submission failed")
            return False
        
        job_id = response["id"]
        print(f"   Job submitted: {job_id}")
        
        # Wait for completion
        final_result = client.wait_for_completion(job_id, max_wait=180)
        
        if not final_result:
            print("❌ Single epoch training FAILED - job timeout or error")
            return False
        
        training_time = time.time() - start_time
        
        # Check results
        if final_result.get("status") != "COMPLETED":
            print(f"❌ Single epoch training FAILED - job status: {final_result.get('status')}")
            error = final_result.get("error", "Unknown error")
            print(f"   Error: {error}")
            return False
        
        output = final_result.get("output", {})
        metrics = output.get("metrics", {})
        model_state = output.get("model_state", "")
        
        if not metrics or "loss" not in metrics:
            print("❌ Single epoch training FAILED - no metrics returned")
            return False
        
        print(f"✅ Single epoch training PASSED")
        print(f"   Total time: {training_time:.2f}s")
        print(f"   Loss: {metrics.get('loss', 'N/A'):.4f}")
        print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"   Device used: {metrics.get('device_used', 'Unknown')}")
        print(f"   CUDA available: {metrics.get('cuda_available', 'Unknown')}")
        print(f"   Model state size: {len(model_state)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Single epoch training FAILED - {str(e)}")
        logger.debug(f"running test_single_epoch_training ... error: {str(e)}")
        return False

def test_gpu_performance(client: RunPodServerlessClient) -> bool:
    """
    Test GPU performance with a larger model
    
    Args:
        client: RunPod serverless client instance
        
    Returns:
        True if performance test passes
    """
    logger.debug("running test_gpu_performance ... testing GPU performance")
    
    print("Testing GPU performance with large model...")
    
    try:
        # Generate larger dataset
        large_training_data = {
            "features": [[float(i+j) for i in range(100)] for j in range(1000)],  # 1000 samples, 100 features
            "labels": [j % 10 for j in range(1000)]  # 10 classes
        }
        
        model_config = create_large_model_config()
        hyperparams = create_test_hyperparameters()
        
        # Create job payload
        payload = {
            "input": {
                "operation": "train_epoch",
                "model_config": model_config,
                "training_data": large_training_data,
                "hyperparameters": hyperparams
            }
        }
        
        # Submit job
        start_time = time.time()
        response = client.submit_job(payload)
        
        if not response or "id" not in response:
            print("❌ GPU performance test FAILED - job submission failed")
            return False
        
        job_id = response["id"]
        print(f"   Large model job submitted: {job_id}")
        
        # Wait for completion with longer timeout
        final_result = client.wait_for_completion(job_id, max_wait=300)
        
        if not final_result:
            print("❌ GPU performance test FAILED - job timeout or error")
            return False
        
        total_time = time.time() - start_time
        
        # Check results
        if final_result.get("status") != "COMPLETED":
            print(f"❌ GPU performance test FAILED - job status: {final_result.get('status')}")
            error = final_result.get("error", "Unknown error")
            print(f"   Error: {error}")
            return False
        
        output = final_result.get("output", {})
        metrics = output.get("metrics", {})
        
        if not metrics or "loss" not in metrics:
            print("❌ GPU performance test FAILED - no metrics returned")
            return False
        
        samples_processed = metrics.get('samples_processed', 0)
        samples_per_second = samples_processed / total_time if total_time > 0 else 0
        
        print(f"✅ GPU performance test PASSED")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Samples processed: {samples_processed}")
        print(f"   Processing rate: {samples_per_second:.1f} samples/second")
        print(f"   Loss: {metrics.get('loss', 'N/A'):.4f}")
        print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"   Device used: {metrics.get('device_used', 'Unknown')}")
        
        # Performance assessment
        if total_time < 10 and samples_per_second > 50:
            print("🚀 Performance indicates GPU acceleration!")
            return True
        elif total_time > 30 or samples_per_second < 20:
            print("🐌 Performance suggests CPU processing")
            return False
        else:
            print("❓ Performance assessment unclear")
            return True  # Still pass the test
        
    except Exception as e:
        print(f"❌ GPU performance test FAILED - {str(e)}")
        logger.debug(f"running test_gpu_performance ... error: {str(e)}")
        return False

def run_comprehensive_test(endpoint_id: str, api_key: Optional[str] = None) -> None:
    """
    Run comprehensive test suite including GPU verification
    
    Args:
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key (optional)
    """
    logger.debug("running run_comprehensive_test ... starting comprehensive test suite")
    
    print("=" * 60)
    print("RunPod Serverless GPU Integration Test Suite")
    print("=" * 60)
    
    # Initialize client
    try:
        client = RunPodServerlessClient(endpoint_id=endpoint_id, api_key=api_key)
        print(f"Initialized client for endpoint: {endpoint_id}")
    except Exception as e:
        print(f"❌ FAILED to initialize client: {str(e)}")
        return
    
    # Run tests
    tests = [
        ("GPU Information & Verification", lambda: test_gpu_information(client)),
        ("Basic Connectivity", lambda: test_basic_connectivity(client)),
        ("Single Epoch Training", lambda: test_single_epoch_training(client)),
        ("GPU Performance Test", lambda: test_gpu_performance(client)),
    ]
    
    passed = 0
    total = len(tests)
    gpu_confirmed = False
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            result = test_func()
            if result:
                passed += 1
                if test_name == "GPU Information & Verification":
                    gpu_confirmed = result
            time.sleep(2)  # Brief pause between tests
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {str(e)}")
            logger.debug(f"running run_comprehensive_test ... {test_name} failed: {str(e)}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if gpu_confirmed:
        print("🎉 GPU CONFIRMED: Your RunPod endpoint is using GPU acceleration!")
    else:
        print("⚠️  GPU Status: Could not confirm GPU acceleration")
    
    print(f"{'=' * 60}")
    
    if passed == total:
        print("🎉 All tests PASSED! RunPod serverless integration is working correctly.")
    else:
        print(f"⚠️  {total - passed} test(s) FAILED. Check logs for details.")

if __name__ == "__main__":
    # Get configuration from environment
    endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
    api_key = os.getenv('RUNPOD_API_KEY')
    
    if not endpoint_id:
        print("❌ Missing RUNPOD_ENDPOINT_ID environment variable!")
        print("Please set RUNPOD_ENDPOINT_ID in your .env file")
        sys.exit(1)
    
    run_comprehensive_test(endpoint_id, api_key)