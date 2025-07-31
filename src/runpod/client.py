"""
RunPod Serverless API Client
Provides easy-to-use functions for all RunPod serverless API endpoints
Focuses on job execution - instance management must be done manually via RunPod console
"""
from dotenv import load_dotenv
import os
import requests
import time
import json
import sys
from typing import Dict, Any, Optional, Iterator
from pathlib import Path

# Import logger from utils
current_file = Path(__file__)
project_root = current_file.parent.parent.parent
print(f"running client.py ... adding project root to sys.path: {project_root}")
sys.path.insert(0, str(project_root))

from src.utils.logger import logger
env_file = project_root / '.env'
load_dotenv(env_file)

class RunPodClient:
    """
    Client for RunPod Serverless API with support for all job execution endpoints
    
    Note: Instance/endpoint management must be done manually via RunPod console.
    This client focuses on job execution operations only.
    """
    
    def __init__(self, endpoint_id: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize RunPod client with credentials from environment or parameters
        
        Args:
            endpoint_id (str, optional): RunPod serverless endpoint ID. 
                                       If None, reads from RUNPOD_ENDPOINT_ID env var
            api_key (str, optional): RunPod API key. 
                                   If None, reads from RUNPOD_API_KEY env var
        
        Raises:
            ValueError: If endpoint_id or api_key cannot be found, or if endpoint doesn't exist
        """
        logger.debug("running RunPodClient.__init__ ... initializing RunPod client")
        
        # Get credentials from parameters or environment
        self.endpoint_id = endpoint_id or os.getenv('RUNPOD_ENDPOINT_ID')
        self.api_key = api_key or os.getenv('RUNPOD_API_KEY')
        
        if not self.endpoint_id:
            self._show_setup_instructions()
            raise ValueError("endpoint_id not provided and RUNPOD_ENDPOINT_ID env var not set")
        if not self.api_key:
            raise ValueError("api_key not provided and RUNPOD_API_KEY env var not set")
        
        # Set up base URL and headers
        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        # Verify endpoint exists by attempting a health check
        try:
            health_result = self.health()
            logger.debug(f"running RunPodClient.__init__ ... initialized with endpoint {self.endpoint_id}")
        except Exception as e:
            logger.error(f"running RunPodClient.__init__ ... endpoint {self.endpoint_id} not accessible: {str(e)}")
            self._show_setup_instructions()
            raise ValueError(f"Endpoint {self.endpoint_id} not accessible - please verify it exists and is active")
    
    def _show_setup_instructions(self):
        """Show detailed instructions for manual RunPod serverless setup"""
        # Extract docker image info if possible
        docker_username = os.getenv('DOCKER_HUB_USERNAME', 'your_username')
        docker_image = f"{docker_username}/gpu-proxy-fat-image:latest"
        
        print(f"\n{'='*80}")
        print(f"🛠️  MANUAL RUNPOD SERVERLESS SETUP REQUIRED")
        print(f"{'='*80}")
        print(f"\n📋 Your Docker Image: {docker_image}")
        print(f"\n🔧 Setup Instructions:")
        print(f"   1. Go to: https://www.runpod.io/console/serverless/user/templates")
        print(f"   2. Click 'New Template'")
        print(f"   3. Enter these details:")
        print(f"      • Template Name: gpu-proxy-template")
        print(f"      • Container Image: {docker_image}")
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
        print(f"{'='*80}\n")
    
    def run(self, input_data: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """
        Submit asynchronous job to RunPod serverless endpoint
        
        Args:
            input_data (Dict[str, Any]): Input data for the serverless function
            timeout (int): Request timeout in seconds (default: 300)
        
        Returns:
            Dict[str, Any]: Response containing job ID and initial status
            
        Example:
            ```python
            client = RunPodClient()
            response = client.run({
                "operation": "execute_code",
                "code": "import torch; result = torch.cuda.is_available()",
                "timeout_seconds": 60
            })
            job_id = response['id']
            ```
        
        Raises:
            requests.RequestException: If the HTTP request fails
            ValueError: If the response indicates an error
        """
        logger.debug("running RunPodClient.run ... submitting asynchronous job")
        
        url = f"{self.base_url}/run"
        payload = {'input': input_data}
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('status') == 'error':
                raise ValueError(f"RunPod error: {result.get('error', 'Unknown error')}")
            
            logger.debug(f"running RunPodClient.run ... job submitted successfully, ID: {result.get('id')}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"running RunPodClient.run ... request failed: {str(e)}")
            raise
    
    def runsync(self, input_data: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """
        Submit synchronous job to RunPod serverless endpoint (waits for completion)
        
        Args:
            input_data (Dict[str, Any]): Input data for the serverless function
            timeout (int): Request timeout in seconds (default: 300)
        
        Returns:
            Dict[str, Any]: Complete response with job results
            
        Example:
            ```python
            client = RunPodClient()
            response = client.runsync({
                "operation": "execute_code",
                "code": "result = 2 + 2"
            })
            print(response['output']['execution_result']['result'])  # 4
            ```
        
        Raises:
            requests.RequestException: If the HTTP request fails
            ValueError: If the response indicates an error
        """
        logger.debug("running RunPodClient.runsync ... submitting synchronous job")
        
        url = f"{self.base_url}/runsync"
        payload = {'input': input_data}
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('status') == 'error':
                raise ValueError(f"RunPod error: {result.get('error', 'Unknown error')}")
            
            logger.debug("running RunPodClient.runsync ... synchronous job completed successfully")
            return result
            
        except requests.RequestException as e:
            logger.error(f"running RunPodClient.runsync ... request failed: {str(e)}")
            raise
    
    def health(self) -> Dict[str, Any]:
        """
        Check health status of RunPod serverless endpoint
        
        Returns:
            Dict[str, Any]: Health status information including worker availability
            
        Example:
            ```python
            client = RunPodClient()
            health = client.health()
            print(f"Workers ready: {health.get('workers', {}).get('ready', 0)}")
            ```
        
        Raises:
            requests.RequestException: If the HTTP request fails
        """
        logger.debug("running RunPodClient.health ... checking endpoint health")
        
        url = f"{self.base_url}/health"
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.debug("running RunPodClient.health ... health check completed")
            return result
            
        except requests.RequestException as e:
            logger.error(f"running RunPodClient.health ... health check failed: {str(e)}")
            raise
    
    def cancel(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running job
        
        Args:
            job_id (str): ID of the job to cancel
        
        Returns:
            Dict[str, Any]: Cancellation confirmation
            
        Example:
            ```python
            client = RunPodClient()
            response = client.run({"operation": "execute_code", "code": "time.sleep(100)"})
            job_id = response['id']
            
            # Cancel the long-running job
            cancel_result = client.cancel(job_id)
            print(f"Job cancelled: {cancel_result}")
            ```
        
        Raises:
            requests.RequestException: If the HTTP request fails
        """
        logger.debug(f"running RunPodClient.cancel ... cancelling job {job_id}")
        
        url = f"{self.base_url}/cancel/{job_id}"
        
        try:
            response = requests.post(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"running RunPodClient.cancel ... job {job_id} cancelled successfully")
            return result
            
        except requests.RequestException as e:
            logger.error(f"running RunPodClient.cancel ... cancel request failed: {str(e)}")
            raise
    
    def purge_queue(self) -> Dict[str, Any]:
        """
        Purge all pending jobs from the queue
        
        Returns:
            Dict[str, Any]: Purge operation result
            
        Example:
            ```python
            client = RunPodClient()
            result = client.purge_queue()
            print(f"Queue purged: {result}")
            ```
        
        Warning:
            This will cancel ALL pending jobs in the queue. Use with caution.
        
        Raises:
            requests.RequestException: If the HTTP request fails
        """
        logger.debug("running RunPodClient.purge_queue ... purging job queue")
        
        url = f"{self.base_url}/purge-queue"
        
        try:
            response = requests.post(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.debug("running RunPodClient.purge_queue ... queue purged successfully")
            return result
            
        except requests.RequestException as e:
            logger.error(f"running RunPodClient.purge_queue ... purge request failed: {str(e)}")
            raise
    
    def status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a specific job
        
        Args:
            job_id (str): ID of the job to check
        
        Returns:
            Dict[str, Any]: Job status and results (if completed)
            
        Example:
            ```python
            client = RunPodClient()
            response = client.run({"operation": "system_info"})
            job_id = response['id']
            
            # Poll for completion
            while True:
                status = client.status(job_id)
                if status['status'] in ['COMPLETED', 'FAILED']:
                    break
                time.sleep(1)
            
            print(status['output'])
            ```
        
        Raises:
            requests.RequestException: If the HTTP request fails
        """
        logger.debug(f"running RunPodClient.status ... checking status of job {job_id}")
        
        url = f"{self.base_url}/status/{job_id}"
        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"running RunPodClient.status ... status retrieved for job {job_id}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"running RunPodClient.status ... status request failed: {str(e)}")
            raise
    
    def stream(self, job_id: str) -> Iterator[Dict[str, Any]]:
        """
        Stream real-time updates for a job
        
        Args:
            job_id (str): ID of the job to stream
        
        Yields:
            Dict[str, Any]: Real-time job updates
            
        Example:
            ```python
            client = RunPodClient()
            response = client.run({
                "operation": "execute_code",
                "code": '''
                for i in range(5):
                    print(f"Step {i}")
                    time.sleep(1)
                result = "completed"
                '''
            })
            job_id = response['id']
            
            # Stream real-time updates
            for update in client.stream(job_id):
                print(f"Update: {update}")
                if update.get('status') in ['COMPLETED', 'FAILED']:
                    break
            ```
        
        Raises:
            requests.RequestException: If the HTTP request fails
        """
        logger.debug(f"running RunPodClient.stream ... starting stream for job {job_id}")
        
        url = f"{self.base_url}/stream/{job_id}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        update = json.loads(line)
                        yield update
                    except json.JSONDecodeError:
                        logger.warning(f"running RunPodClient.stream ... invalid JSON in stream: {line}")
                        continue
            
            logger.debug(f"running RunPodClient.stream ... stream ended for job {job_id}")
            
        except requests.RequestException as e:
            logger.error(f"running RunPodClient.stream ... stream request failed: {str(e)}")
            raise
    
    def wait_for_completion(self, job_id: str, poll_interval: float = 1.0, max_wait: int = 600) -> Dict[str, Any]:
        """
        Convenience method to wait for job completion with polling
        
        Args:
            job_id (str): ID of the job to wait for
            poll_interval (float): Seconds between status checks (default: 1.0)
            max_wait (int): Maximum seconds to wait before timeout (default: 600)
        
        Returns:
            Dict[str, Any]: Final job status and results
            
        Example:
            ```python
            client = RunPodClient()
            response = client.run({"operation": "benchmark"})
            job_id = response['id']
            
            # Wait for completion with custom polling
            result = client.wait_for_completion(job_id, poll_interval=0.5, max_wait=300)
            print(result['output'])
            ```
        
        Raises:
            TimeoutError: If job doesn't complete within max_wait seconds
            requests.RequestException: If status requests fail
        """
        logger.debug(f"running RunPodClient.wait_for_completion ... waiting for job {job_id}")
        
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Job {job_id} did not complete within {max_wait} seconds")
            
            status = self.status(job_id)
            job_status = status.get('status', 'UNKNOWN')
            
            if job_status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                logger.debug(f"running RunPodClient.wait_for_completion ... job {job_id} finished with status {job_status}")
                return status
            
            time.sleep(poll_interval)
    
    def execute_code_sync(self, code: str, context: Optional[Dict[str, Any]] = None, 
                         timeout_seconds: int = 300) -> Dict[str, Any]:
        """
        Convenience method to execute code synchronously and return results
        
        Args:
            code (str): Python code to execute
            context (Dict[str, Any], optional): Context data to pass to the code
            timeout_seconds (int): Execution timeout (default: 300)
        
        Returns:
            Dict[str, Any]: Execution results including output and any returned values
            
        Example:
            ```python
            client = RunPodClient()
            
            # Simple calculation
            result = client.execute_code_sync("result = 2 ** 10")
            print(result['execution_result']['result'])  # 1024
            
            # With context data
            result = client.execute_code_sync(
                "result = context['x'] + context['y']",
                context={'x': 10, 'y': 20}
            )
            print(result['execution_result']['result'])  # 30
            ```
        
        Raises:
            requests.RequestException: If the request fails
            ValueError: If the execution fails
        """
        logger.debug("running RunPodClient.execute_code_sync ... executing code synchronously")
        
        input_data = {
            "operation": "execute_code",
            "code": code,
            "timeout_seconds": timeout_seconds
        }
        
        if context:
            input_data["context"] = context
        
        response = self.runsync(input_data, timeout=timeout_seconds + 30)
        
        if response.get('status') == 'error':
            raise ValueError(f"Code execution failed: {response.get('error', 'Unknown error')}")
        
        return response.get('output', {})


# Convenience functions for direct usage without class instantiation
def run(input_data: Dict[str, Any], endpoint_id: Optional[str] = None, 
        api_key: Optional[str] = None, timeout: int = 300) -> Dict[str, Any]:
    """Convenience function for RunPodClient.run()"""
    client = RunPodClient(endpoint_id, api_key)
    return client.run(input_data, timeout)

def runsync(input_data: Dict[str, Any], endpoint_id: Optional[str] = None, 
            api_key: Optional[str] = None, timeout: int = 300) -> Dict[str, Any]:
    """Convenience function for RunPodClient.runsync()"""
    client = RunPodClient(endpoint_id, api_key)
    return client.runsync(input_data, timeout)

def health(endpoint_id: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for RunPodClient.health()"""
    client = RunPodClient(endpoint_id, api_key)
    return client.health()

def execute_code_sync(code: str, context: Optional[Dict[str, Any]] = None,
                     timeout_seconds: int = 300, endpoint_id: Optional[str] = None,
                     api_key: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for RunPodClient.execute_code_sync()"""
    client = RunPodClient(endpoint_id, api_key)
    return client.execute_code_sync(code, context, timeout_seconds)