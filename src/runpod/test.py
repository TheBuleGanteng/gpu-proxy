#!/usr/bin/env python3
"""
Comprehensive test script for RunPodClient
Tests all endpoint operations and convenience functions
"""
from dotenv import load_dotenv
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
current_file = Path(__file__)
project_root = current_file.parent.parent.parent
print(f"running test.py ... Adding project root to sys.path: {project_root}")

sys.path.insert(0, str(project_root))

from src.runpod.client import RunPodClient
from src.utils.logger import logger
env_file = project_root / '.env'
load_dotenv(env_file)

class RunPodClientTester:
    """Comprehensive tester for RunPodClient functionality"""
    
    def __init__(self):
        """Initialize the tester"""
        logger.debug("running RunPodClientTester.__init__ ... initializing tester")
        
        # Check for required environment variables
        self.api_key = os.getenv('RUNPOD_API_KEY')
        self.endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
        self.docker_username = os.getenv('DOCKER_HUB_USERNAME')
        
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY environment variable not set")
        
        if not self.docker_username:
            raise ValueError("DOCKER_HUB_USERNAME environment variable not set")
        
        # Extract DOCKER_IMAGE_NAME from deploy.sh
        self.docker_image_name = self._extract_docker_image_name()
        
        # Construct full Docker image name
        self.docker_image = f"{self.docker_username}/{self.docker_image_name}:latest"
        
        logger.debug(f"running RunPodClientTester.__init__ ... API key loaded")
        logger.debug(f"running RunPodClientTester.__init__ ... Docker image: {self.docker_image}")
        
        # Initialize client and validate endpoint exists
        if not self.endpoint_id:
            self._show_manual_setup_instructions()
            raise ValueError("RUNPOD_ENDPOINT_ID not set - manual setup required")
        
        try:
            self.client = RunPodClient(self.endpoint_id, self.api_key)
            logger.debug(f"running RunPodClientTester.__init__ ... endpoint ID: {self.endpoint_id}")
            
            # Verify the endpoint actually exists
            if not self._verify_endpoint_exists():
                self._show_manual_setup_instructions()
                raise ValueError(f"Endpoint {self.endpoint_id} not found - please verify or recreate")
                
            logger.debug("running RunPodClientTester.__init__ ... client initialized successfully")
            
        except Exception as e:
            logger.error(f"running RunPodClientTester.__init__ ... failed to initialize client: {e}")
            self._show_manual_setup_instructions() 
            raise
        
        # Test results tracking
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }
    
    def _extract_docker_image_name(self) -> str:
        """Extract DOCKER_IMAGE_NAME from deploy.sh script"""
        logger.debug("running _extract_docker_image_name ... extracting docker image name from deploy.sh")
        
        deploy_script_path = current_file.parent / "deploy.sh"
        
        if not deploy_script_path.exists():
            logger.error("running _extract_docker_image_name ... deploy.sh not found")
            raise ValueError("deploy.sh not found in src/runpod directory")
        
        try:
            with open(deploy_script_path, 'r') as f:
                content = f.read()
            
            # Look for DOCKER_IMAGE_NAME="..." line
            for line in content.split('\n'):
                if line.strip().startswith('DOCKER_IMAGE_NAME='):
                    # Extract the value between quotes
                    image_name = line.split('=', 1)[1].strip().strip('"').strip("'")
                    logger.debug(f"running _extract_docker_image_name ... found docker image name: {image_name}")
                    return image_name
            
            raise ValueError("DOCKER_IMAGE_NAME not found in deploy.sh")
            
        except Exception as e:
            logger.error(f"running _extract_docker_image_name ... failed to read deploy.sh: {str(e)}")
            raise ValueError(f"Failed to extract docker image name from deploy.sh: {str(e)}")
    
    def _verify_endpoint_exists(self) -> bool:
        """Verify that the configured endpoint actually exists"""
        logger.debug(f"running _verify_endpoint_exists ... checking if endpoint {self.endpoint_id} exists")
        
        try:
            # Try a simple health check to verify endpoint exists
            health_result = self.client.health()
            logger.debug(f"running _verify_endpoint_exists ... endpoint {self.endpoint_id} exists and is accessible")
            return True
            
        except Exception as e:
            logger.error(f"running _verify_endpoint_exists ... endpoint {self.endpoint_id} not accessible: {str(e)}")
            return False
    
    def _show_manual_setup_instructions(self):
        """Show detailed instructions for manual RunPod serverless setup"""
        print(f"\n{'='*80}")
        print(f"🛠️  MANUAL RUNPOD SERVERLESS SETUP REQUIRED")
        print(f"{'='*80}")
        print(f"\n📋 Your Docker Image: {self.docker_image}")
        print(f"\n🔧 Setup Instructions:")
        print(f"   1. Go to: https://www.runpod.io/console/serverless/user/templates")
        print(f"   2. Click 'New Template'")
        print(f"   3. Enter these details:")
        print(f"      • Template Name: gpu-proxy-template")
        print(f"      • Container Image: {self.docker_image}")
        print(f"      • Container Disk: 5 GB")
        print(f"      • Docker Command: python handler.py")
        print(f"      • Volume Size: 0 GB")
        print(f"   4. Click 'Save Template'")
        print(f"")
        print(f"   5. Go to: https://www.runpod.io/console/serverless/user/endpoints")
        print(f"   6. Click 'New Endpoint'")
        print(f"   7. Configure the endpoint:")
        print(f"      • Endpoint Name: gpu-proxy-endpoint")
        print(f"      • Select your template: gpu-proxy-template")
        print(f"      • GPU Type: RTX 4090 or similar")
        print(f"      • Max Workers: 1-3")
        print(f"      • Min Workers: 0")
        print(f"      • Idle Timeout: 5 seconds")
        print(f"   8. Click 'Create Endpoint'")
        print(f"")
        print(f"   9. Copy the Endpoint ID from the created endpoint")
        print(f"  10. Add to your .env file:")
        print(f"      RUNPOD_ENDPOINT_ID=your_endpoint_id_here")
        print(f"")
        print(f"⚠️  IMPORTANT: Serverless endpoints must be created manually via the RunPod console.")
        print(f"   Programmatic creation via API has known limitations and reliability issues.")
        print(f"")
        print(f"💡 After setup, re-run this test to validate your endpoint configuration.")
        print(f"{'='*80}\n")
    
    def print_test_header(self, test_name: str):
        """Print a formatted test header"""
        print(f"\n{'='*60}")
        print(f"🧪 TESTING: {test_name}")
        print(f"{'='*60}")
    
    def print_test_result(self, test_name: str, success: bool, message: str = "", data: Any = None):
        """Print and record test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if message:
            print(f"   {message}")
        if data and isinstance(data, dict):
            print(f"   Data preview: {str(data)[:100]}...")
        
        # Record result
        if success:
            self.test_results["passed"] += 1
        else:
            self.test_results["failed"] += 1
        
        self.test_results["details"].append({
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": time.time()
        })
    
    def skip_test(self, test_name: str, reason: str):
        """Skip a test with reason"""
        print(f"⏭️  SKIP {test_name}")
        print(f"   Reason: {reason}")
        self.test_results["skipped"] += 1
        self.test_results["details"].append({
            "test": test_name,
            "success": None,
            "message": f"Skipped: {reason}",
            "timestamp": time.time()
        })
    
    def test_endpoint_operations(self):
        """Test all endpoint operation methods"""
        self.print_test_header("Endpoint Operations")
        
        # Test 1: Health check
        try:
            logger.debug("running test_endpoint_operations ... testing health")
            
            health_result = self.client.health()
            ready_workers = health_result.get('workers', {}).get('ready', 0)
            idle_workers = health_result.get('workers', {}).get('idle', 0)
            
            self.print_test_result(
                "health()", 
                True, 
                f"Health check completed - Ready: {ready_workers}, Idle: {idle_workers}",
                health_result
            )
            
        except Exception as e:
            self.print_test_result("health()", False, f"Error: {str(e)}")
        
        # Test 2: Execute code synchronously
        try:
            logger.debug("running test_endpoint_operations ... testing execute_code_sync")
            
            test_code = """
import torch
import numpy as np
import tensorflow as tf
result = {
    "torch_version": torch.__version__,
    "numpy_version": np.__version__,
    "tensorflow_version": tf.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "test_calculation": 2 ** 10
}
"""
            
            execution_result = self.client.execute_code_sync(test_code, timeout_seconds=60)
            
            if execution_result.get('execution_result', {}).get('success', False):
                result_data = execution_result['execution_result']['result']
                self.print_test_result(
                    "execute_code_sync()", 
                    True, 
                    f"Code executed successfully",
                    result_data
                )
                print(f"   PyTorch: {result_data.get('torch_version', 'N/A')}")
                print(f"   TensorFlow: {result_data.get('tensorflow_version', 'N/A')}")
                print(f"   CUDA Available: {result_data.get('cuda_available', 'N/A')}")
                print(f"   GPU Count: {result_data.get('device_count', 'N/A')}")
            else:
                error_msg = execution_result.get('execution_result', {}).get('error', 'Unknown error')
                self.print_test_result("execute_code_sync()", False, f"Execution failed: {error_msg}")
                
        except Exception as e:
            self.print_test_result("execute_code_sync()", False, f"Error: {str(e)}")
        
        # Test 3: Execute code with context
        try:
            logger.debug("running test_endpoint_operations ... testing execute_code_sync with context")
            
            context_code = """
x = context['input_a']
y = context['input_b']
result = {
    "sum": x + y,
    "product": x * y,
    "context_received": True,
    "context_keys": list(context.keys())
}
"""
            
            context_data = {"input_a": 15, "input_b": 25}
            
            execution_result = self.client.execute_code_sync(
                context_code, 
                context=context_data, 
                timeout_seconds=30
            )
            
            if execution_result.get('execution_result', {}).get('success', False):
                result_data = execution_result['execution_result']['result']
                self.print_test_result(
                    "execute_code_sync() with context", 
                    True, 
                    f"Context code executed successfully",
                    result_data
                )
                print(f"   Sum (15 + 25): {result_data.get('sum', 'N/A')}")
                print(f"   Product (15 * 25): {result_data.get('product', 'N/A')}")
            else:
                error_msg = execution_result.get('execution_result', {}).get('error', 'Unknown error')
                self.print_test_result("execute_code_sync() with context", False, f"Execution failed: {error_msg}")
                
        except Exception as e:
            self.print_test_result("execute_code_sync() with context", False, f"Error: {str(e)}")
        
        # Test 4: System info operation
        try:
            logger.debug("running test_endpoint_operations ... testing system_info operation")
            
            system_info_result = self.client.runsync({
                "operation": "system_info"
            }, timeout=30)
            
            if system_info_result.get('status') != 'error':
                output = system_info_result.get('output', {})
                self.print_test_result(
                    "system_info operation", 
                    True, 
                    f"System info retrieved",
                    output
                )
                
                # Show key system info if available
                if 'system_info' in output:
                    sys_info = output['system_info']
                    if 'gpu_info' in sys_info:
                        gpu_info = sys_info['gpu_info']
                        print(f"   GPU: {gpu_info.get('name', 'N/A')}")
                        print(f"   GPU Memory: {gpu_info.get('memory_total', 'N/A')}")
            else:
                self.print_test_result("system_info operation", False, f"Operation failed")
                
        except Exception as e:
            self.print_test_result("system_info operation", False, f"Error: {str(e)}")
        
        # Test 5: Benchmark operation
        try:
            logger.debug("running test_endpoint_operations ... testing benchmark operation")
            
            benchmark_result = self.client.runsync({
                "operation": "benchmark"
            }, timeout=60)
            
            if benchmark_result.get('status') != 'error':
                output = benchmark_result.get('output', {})
                self.print_test_result(
                    "benchmark operation", 
                    True, 
                    f"Benchmark completed",
                    output
                )
                
                # Show benchmark results if available
                if 'benchmark_results' in output:
                    bench = output['benchmark_results']
                    comp_time = bench.get('computation_time_ms')
                    if comp_time:
                        print(f"   GPU Computation Time: {comp_time:.2f}ms")
            else:
                self.print_test_result("benchmark operation", False, f"Benchmark failed")
                
        except Exception as e:
            self.print_test_result("benchmark operation", False, f"Error: {str(e)}")
        
        # Test 6: Async job workflow
        try:
            logger.debug("running test_endpoint_operations ... testing async job workflow")
            
            # Submit async job
            async_job = self.client.run({
                "operation": "execute_code",
                "code": "import time; time.sleep(2); result = 'async_completed'",
                "timeout_seconds": 30
            })
            
            job_id = async_job.get('id')
            if job_id:
                print(f"   Submitted async job: {job_id}")
                
                # Wait for completion
                final_result = self.client.wait_for_completion(job_id, poll_interval=0.5, max_wait=60)
                
                if final_result.get('status') == 'COMPLETED':
                    self.print_test_result(
                        "async job workflow", 
                        True, 
                        f"Async job completed successfully"
                    )
                else:
                    self.print_test_result("async job workflow", False, f"Job status: {final_result.get('status')}")
            else:
                self.print_test_result("async job workflow", False, "Failed to get job ID")
                
        except Exception as e:
            self.print_test_result("async job workflow", False, f"Error: {str(e)}")
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        self.print_test_header("Convenience Functions")
        
        # Test convenience functions (these just import and call the class methods)
        try:
            logger.debug("running test_convenience_functions ... testing convenience function imports")
            
            from src.runpod.client import execute_code_sync, health
            
            self.print_test_result(
                "convenience function imports", 
                True, 
                f"Convenience functions imported successfully"
            )
            
            # Test execute_code_sync convenience function
            result = execute_code_sync(
                "result = 'convenience_function_test'", 
                endpoint_id=self.endpoint_id,
                api_key=self.api_key
            )
            
            if result.get('execution_result', {}).get('success', False):
                self.print_test_result(
                    "execute_code_sync() convenience function", 
                    True, 
                    f"Convenience function executed successfully"
                )
            else:
                self.print_test_result("execute_code_sync() convenience function", False, "Convenience function failed")
            
            # Test health convenience function
            health_result = health(endpoint_id=self.endpoint_id, api_key=self.api_key)
            ready_workers = health_result.get('workers', {}).get('ready', 0)
            
            self.print_test_result(
                "health() convenience function", 
                True, 
                f"Health convenience function works - Ready workers: {ready_workers}"
            )
            
        except Exception as e:
            self.print_test_result("convenience functions", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all tests and print summary"""
        print(f"\n🚀 Starting RunPodClient Tests")
        print(f"API Key: {'✅ Set' if self.api_key else '❌ Not Set'}")
        print(f"Endpoint ID: {'✅ Set' if self.endpoint_id else '❌ Not Set'}")
        print(f"Docker Hub Username: {'✅ Set' if self.docker_username else '❌ Not Set'}")
        print(f"Docker Image: {'✅ ' + self.docker_image if hasattr(self, 'docker_image') and self.docker_image else '❌ Not Set'}")
        
        try:
            # Run test suites
            self.test_endpoint_operations()
            self.test_convenience_functions()
            
        except KeyboardInterrupt:
            print(f"\n🛑 Tests interrupted by user")
            raise
        except Exception as e:
            logger.error(f"running run_all_tests ... unexpected error: {str(e)}")
            raise
        
        # Print final summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print final test summary"""
        total_tests = self.test_results["passed"] + self.test_results["failed"] + self.test_results["skipped"]
        
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {self.test_results['passed']}")
        print(f"❌ Failed: {self.test_results['failed']}")
        print(f"⏭️  Skipped: {self.test_results['skipped']}")
        
        if self.test_results["failed"] > 0:
            print(f"\n🔍 Failed Tests:")
            for detail in self.test_results["details"]:
                if not detail["success"] and detail["success"] is not None:
                    print(f"   - {detail['test']}: {detail['message']}")
        
        success_rate = (self.test_results['passed'] / (self.test_results['passed'] + self.test_results['failed'])) * 100 if (self.test_results['passed'] + self.test_results['failed']) > 0 else 0
        
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        if self.test_results["failed"] == 0:
            print(f"🎉 All tests passed!")
        else:
            print(f"⚠️  Some tests failed - check configuration and endpoint status")

def main():
    """Main function to run tests"""
    print("GPU Proxy - RunPodClient Test Suite")
    print("=" * 60)
    
    try:
        tester = RunPodClientTester()
        tester.run_all_tests()
        
    except KeyboardInterrupt:
        print(f"\n🛑 Tests interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"running main ... test suite failed to initialize: {str(e)}")
        print(f"❌ Test suite failed to initialize: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())