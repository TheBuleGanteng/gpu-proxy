"""
Generic Execute-What-You-Send Handler for RunPod Serverless
Supports arbitrary Python code execution with GPU access and basic infrastructure operations
"""

import runpod
import sys
import io
import json
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Simple logging for serverless environment
import logging

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("runpod_handler")
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s | %(levelname)8s | %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger()

# Pre-import common libraries for fat image approach
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch available: {torch.__version__}, device: {device}")
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    device = None
    logger.warning("PyTorch not available")

try:
    import tensorflow as tf # type: ignore
    logger.info(f"TensorFlow available: {tf.__version__}")
except ImportError:
    tf = None  # type: ignore
    logger.warning("TensorFlow not available")

try:
    import numpy as np
    logger.info(f"NumPy available: {np.__version__}")
except ImportError:
    np = None  # type: ignore
    logger.warning("NumPy not available")

def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system and GPU information
    
    Returns:
        dict: System information including GPU specs, available libraries, and performance metrics
    """
    logger.debug("running get_system_info ... gathering system information")
    
    system_info = {
        "python_version": sys.version,
        "available_libraries": {},
        "gpu_info": {},
        "timestamp": time.time()
    }
    
    # Check available libraries
    libraries_to_check = {
        "torch": torch,
        "tensorflow": tf,
        "numpy": np
    }
    
    for lib_name, lib_module in libraries_to_check.items():
        if lib_module is not None:
            system_info["available_libraries"][lib_name] = {
                "available": True,
                "version": getattr(lib_module, "__version__", "unknown")
            }
        else:
            system_info["available_libraries"][lib_name] = {
                "available": False,
                "version": None
            }
    
    # GPU Information
    if torch is not None:
        system_info["gpu_info"].update({
            "cuda_available": torch.cuda.is_available(),
            "device_name": str(device),
        })
        
        if torch.cuda.is_available():
            system_info["gpu_info"].update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": getattr(torch.version, 'cuda', None), # type: ignore
                "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                "current_device": torch.cuda.current_device(),
                "memory_allocated_mb": torch.cuda.memory_allocated(0) / (1024 * 1024),
                "memory_reserved_mb": torch.cuda.memory_reserved(0) / (1024 * 1024)
            })
            
            # Run basic GPU functionality test
            try:
                test_tensor = torch.randn(100, 100).to(device)
                result = torch.matmul(test_tensor, test_tensor.T)
                del test_tensor, result
                torch.cuda.empty_cache()
                system_info["gpu_info"]["functionality_test"] = "PASSED"
            except Exception as e:
                system_info["gpu_info"]["functionality_test"] = f"FAILED: {str(e)}"
    
    return system_info

def run_performance_benchmark() -> Dict[str, Any]:
    """
    Run performance benchmark to test GPU acceleration
    
    Returns:
        dict: Benchmark results with timing and performance metrics
    """
    logger.debug("running run_performance_benchmark ... running performance tests")
    
    benchmark_results = {
        "timestamp": time.time(),
        "device": str(device) if device else "cpu",
        "benchmark_status": "COMPLETED"
    }
    
    if torch is not None and torch.cuda.is_available():
        try:
            matrix_size = 2000
            
            # Create test matrices
            start_time = time.time()
            a = torch.randn(matrix_size, matrix_size).to(device)
            b = torch.randn(matrix_size, matrix_size).to(device)
            creation_time = time.time() - start_time
            
            # Perform matrix multiplication
            start_time = time.time()
            result = torch.matmul(a, b)
            torch.cuda.synchronize()
            computation_time = time.time() - start_time
            
            benchmark_results.update({
                "matrix_size": matrix_size,
                "creation_time_ms": creation_time * 1000,
                "computation_time_ms": computation_time * 1000,
                "operations_per_second": (matrix_size ** 3) / computation_time,
                "performance_assessment": "GPU_ACCELERATED" if computation_time < 1.0 else "CPU_LIKELY"
            })
            
            # Cleanup
            del a, b, result
            torch.cuda.empty_cache()
            
        except Exception as e:
            benchmark_results.update({
                "benchmark_error": str(e),
                "benchmark_status": "FAILED"
            })
    else:
        benchmark_results["benchmark_status"] = "SKIPPED - No GPU available"
    
    return benchmark_results

def execute_code(code_string: str, context_data: Optional[Dict[str, Any]] = None, timeout_seconds: int = 300) -> Dict[str, Any]:
    """
    Execute arbitrary Python code with optional context data and safety measures
    
    Args:
        code_string (str): Python code to execute
        context_data (dict, optional): Data to make available to the code via 'context' variable
        timeout_seconds (int): Maximum execution time before timeout
        
    Returns:
        dict: Execution results including output, errors, and any returned values
    """
    logger.debug("running execute_code ... executing user-provided code")
    
    # Prepare execution environment with pre-imported libraries
    exec_globals = {
        '__builtins__': __builtins__,
        'json': json,
        'time': time,
        'sys': sys,
        'logger': logger
    }
    
    # Add pre-imported libraries if available
    if torch is not None:
        exec_globals['torch'] = torch
        exec_globals['nn'] = nn
        exec_globals['optim'] = optim
        exec_globals['device'] = device
    
    if tf is not None:
        exec_globals['tf'] = tf
        exec_globals['keras'] = tf.keras
    
    if np is not None:
        exec_globals['np'] = np
    
    # Add context data if provided
    if context_data:
        exec_globals['context'] = context_data
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    exec_locals = {}
    execution_error = None
    execution_start = time.time()
    
    try:
        # Execute the user code with output capture
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code_string, exec_globals, exec_locals)
        
        execution_time = time.time() - execution_start
        
        # Check for timeout
        if execution_time > timeout_seconds:
            raise TimeoutError(f"Execution exceeded {timeout_seconds} seconds")
        
        # Capture any returned result
        result = exec_locals.get('result', None)
        
        logger.debug(f"running execute_code ... code executed successfully in {execution_time:.2f}s")
        
    except Exception as e:
        execution_error = str(e)
        execution_time = time.time() - execution_start
        result = None  # Initialize result for error case
        logger.error(f"running execute_code ... code execution failed after {execution_time:.2f}s: {e}")
    
    # Return comprehensive results
    return {
        "success": execution_error is None,
        "result": result,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "error": execution_error,
        "execution_time_seconds": execution_time,
        "available_libraries": {
            "torch": torch is not None,
            "tensorflow": tf is not None,
            "numpy": np is not None
        },
        "device_used": str(device) if device else "cpu",
        "timestamp": time.time()
    }


"""
Final Optimized Handler - Minimal Response for RunPod Size Limits
"""

# Replace your current handler function with this ultra-optimized version:

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    TYPE-SAFE: NaN-safe minimal response handler with proper types
    """
    logger.debug("running handler ... processing job request")
    
    try:
        job_input = job.get("input", {})
        operation = job_input.get("operation", "execute_code")
        
        # Execute Code Operation (TYPE-SAFE NaN-SAFE RESPONSE)
        if operation == "execute_code":
            code = job_input.get("code", "")
            context = job_input.get("context", {})
            timeout = job_input.get("timeout_seconds", 300)
            
            if not code:
                return {"error": "No code", "status": "error"}
            
            execution_result = execute_code(code, context, timeout)
            
            # TYPE-SAFE NaN-safe minimal response
            if execution_result.get("success"):
                result_data = execution_result.get("result", {})
                
                # Start with bare minimum - properly typed
                minimal_response: Dict[str, Any] = {"success": True}
                
                # Add ONLY essential training results with NaN handling
                if isinstance(result_data, dict):
                    history = result_data.get('history', {})
                    if history:
                        # Helper function to safely convert metrics
                        def safe_metric(value: Any) -> float:
                            """Convert metric to safe float, replacing NaN/inf with 0.0"""
                            try:
                                float_val = float(value)
                                # Check for NaN (NaN != NaN in Python)
                                if float_val != float_val:  # This is the NaN check
                                    return 0.0
                                # Check for infinity
                                if float_val == float('inf') or float_val == float('-inf'):
                                    return 0.0
                                return round(float_val, 3)
                            except (ValueError, TypeError, OverflowError):
                                return 0.0
                        
                        # Only final metrics - NaN-safe with proper typing
                        if 'loss' in history and history['loss']:
                            minimal_response['loss'] = safe_metric(history['loss'][-1])
                        if 'accuracy' in history and history['accuracy']:
                            minimal_response['acc'] = safe_metric(history['accuracy'][-1])
                        if 'val_loss' in history and history['val_loss']:
                            minimal_response['val_loss'] = safe_metric(history['val_loss'][-1])
                        if 'val_accuracy' in history and history['val_accuracy']:
                            minimal_response['val_acc'] = safe_metric(history['val_accuracy'][-1])
                        
                        minimal_response['epochs'] = len(history.get('loss', []))
                    
                    # Optional: model params (also make safe)
                    if 'model_params' in result_data:
                        try:
                            params = int(result_data['model_params'])
                            minimal_response['params'] = params
                        except (ValueError, TypeError):
                            pass  # Skip if invalid
                
                # Log for monitoring
                response_size = len(str(minimal_response))
                logger.info(f"TYPE-SAFE RESPONSE SIZE: {response_size} chars")
                logger.info(f"TYPE-SAFE RESPONSE: {minimal_response}")
                
                return minimal_response
            
            else:
                # Ultra minimal error
                error_msg = execution_result.get("error", "Err")
                return {"success": False, "error": str(error_msg)[:20]}
        
        # Other operations - minimal
        elif operation == "health_check":
            return {"ok": True}
        
        elif operation == "system_info":
            return {"gpu": torch.cuda.is_available() if torch else False}
        
        else:
            return {"error": "Unknown op"}
            
    except Exception as e:
        logger.error(f"running handler ... error: {str(e)}")
        return {"error": str(e)[:20]}

# Start the serverless function
if __name__ == "__main__":
    logger.debug("running main ... starting serverless worker")
    logger.info(f"Python version: {sys.version}")
    
    if device:
        logger.info(f"Device initialized: {device}")
        if torch is not None:
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA available: False (PyTorch not available)")
    
    logger.info("Available libraries:")
    for lib_name, lib_module in [("torch", torch), ("tensorflow", tf), ("numpy", np)]:
        status = "✓" if lib_module else "✗"
        version = getattr(lib_module, "__version__", "unknown") if lib_module else "not available"
        logger.info(f"  {status} {lib_name}: {version}")
    
    logger.info("Handler ready for execute-what-you-send requests")
    runpod.serverless.start({"handler": handler}) # type: ignore