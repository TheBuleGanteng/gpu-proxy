"""
RunPod Serverless API Client - ENHANCED VERSION WITH GPU PROXY INTEGRATION
Provides easy-to-use functions for all RunPod serverless API endpoints
Now includes GPUProxyClient wrapper with auto-setup capabilities

"""
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
import os
import requests
from requests import Session, Response
import time
import json
import sys
import atexit
import signal
import subprocess
import shutil
from typing import Dict, Any, Optional, Iterator, Union
from pathlib import Path

# Import logger from utils
current_file = Path(__file__)
project_root = current_file.parent
print(f"running client.py ... GPU proxy project root: {project_root}")

# Try to import logger from hyperparameter project if available
try:
    # Check if we're in a hyperparameter project context
    hyperparameter_project_root = project_root.parent
    sys.path.insert(0, str(hyperparameter_project_root))
    from src.utils.logger import logger
    logger.debug("running client.py ... imported logger from hyperparameter project")
except ImportError:
    # Fallback to basic logging for standalone GPU proxy usage
    import logging
    logger = logging.getLogger("gpu_proxy_client")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s | %(levelname)8s | %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.debug("running client.py ... using fallback logger for GPU proxy")

env_file = project_root / '.env'
load_dotenv(env_file)

class RunPodClient:
    """
    Client for RunPod Serverless API with support for all job execution endpoints
    
    FIXED: Now includes proper session management and connection cleanup
    
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
        logger.debug("running RunPodClient.__init__ ... initializing RunPod client with connection management")
        
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
            'Authorization': f'Bearer {self.api_key}',
            'Connection': 'close'  # Prevent keep-alive connections
        }
        
        # Create a managed session for connection pooling and cleanup
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Configure session for proper connection management
        self._configure_session()
        
        # Register cleanup handlers
        self._cleanup_registered = False
        self._register_cleanup()
        
        # Verify endpoint exists by attempting a health check
        try:
            health_result = self.health()
            logger.debug(f"running RunPodClient.__init__ ... initialized with endpoint {self.endpoint_id}")
        except Exception as e:
            logger.error(f"running RunPodClient.__init__ ... endpoint {self.endpoint_id} not accessible: {str(e)}")
            self._show_setup_instructions()
            raise ValueError(f"Endpoint {self.endpoint_id} not accessible - please verify it exists and is active")
    
    def _configure_session(self) -> None:
        """Configure the session for optimal connection management"""
        # Set adapter configuration for connection pooling
        adapter = HTTPAdapter(
            pool_connections=1,      # Limit connection pool size
            pool_maxsize=1,          # Maximum pool size
            max_retries=3,           # Retry failed requests
            pool_block=False         # Don't block when pool is full
        )
        
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        
        # Set reasonable timeouts
        self.default_timeout = (10, 300)  # (connect_timeout, read_timeout)
        
        logger.debug("running _configure_session ... session configured for optimal connection management")
    
    def _register_cleanup(self) -> None:
        """Register cleanup handlers to ensure connections are closed"""
        if self._cleanup_registered:
            return
        
        # Register cleanup on normal exit
        atexit.register(self.cleanup)
        
        # Register cleanup on signal termination
        def signal_handler(signum, frame):
            logger.debug(f"running signal_handler ... Received signal {signum}, cleaning up connections...")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        self._cleanup_registered = True
        logger.debug("running _register_cleanup ... cleanup handlers registered")
    
    def cleanup(self) -> None:
        """Clean up session and close all connections"""
        try:
            if hasattr(self, 'session') and self.session:
                logger.debug("running cleanup ... closing RunPod client session...")
                self.session.close()
                logger.debug("running cleanup ... RunPod client session closed")
        except Exception as e:
            logger.warning(f"running cleanup ... error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        return False
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except Exception:
            pass  # Avoid errors during garbage collection
    
    def _make_request(self, method: str, url: str, timeout: Optional[int] = None, **kwargs) -> requests.Response:
        """
        FIX: Centralized request method with proper error handling and connection management
        """
        request_timeout = timeout or self.default_timeout
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=request_timeout, **kwargs)
            elif method.upper() == 'POST':
                response = self.session.post(url, timeout=request_timeout, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response
            
        except requests.RequestException as e:
            logger.error(f"running _make_request ... {method} request to {url} failed: {str(e)}")
            raise
    
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
        FIXED: Now uses managed session with proper connection cleanup
        """
        logger.debug("running RunPodClient.run ... submitting asynchronous job")
        
        url = f"{self.base_url}/run"
        payload = {'input': input_data}
        
        response = self._make_request('POST', url, timeout=timeout, json=payload)
        result = response.json()
        
        if result.get('status') == 'error':
            raise ValueError(f"RunPod error: {result.get('error', 'Unknown error')}")
        
        logger.debug(f"running RunPodClient.run ... job submitted successfully, ID: {result.get('id')}")
        return result
    
    def runsync(self, input_data: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """
        Submit synchronous job to RunPod serverless endpoint (waits for completion)
        FIXED: Now uses managed session with proper connection cleanup
        """
        logger.debug("running RunPodClient.runsync ... submitting synchronous job")
        
        url = f"{self.base_url}/runsync"
        payload = {'input': input_data}
        
        response = self._make_request('POST', url, timeout=timeout, json=payload)
        result = response.json()
        
        if result.get('status') == 'error':
            raise ValueError(f"RunPod error: {result.get('error', 'Unknown error')}")
        
        logger.debug("running RunPodClient.runsync ... synchronous job completed successfully")
        return result
    
    def health(self) -> Dict[str, Any]:
        """
        Check health status of RunPod serverless endpoint
        FIXED: Now uses managed session with proper connection cleanup
        """
        logger.debug("running RunPodClient.health ... checking endpoint health")
        
        url = f"{self.base_url}/health"
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'Connection': 'close'  # Ensure connection is closed
        }
        
        response = self._make_request('GET', url, timeout=30, headers=headers)
        result = response.json()
        
        logger.debug("running RunPodClient.health ... health check completed")
        return result
    
    def cancel(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running job
        FIXED: Now uses managed session with proper connection cleanup
        """
        logger.debug(f"running RunPodClient.cancel ... cancelling job {job_id}")
        
        url = f"{self.base_url}/cancel/{job_id}"
        
        response = self._make_request('POST', url, timeout=30)
        result = response.json()
        
        logger.debug(f"running RunPodClient.cancel ... job {job_id} cancelled successfully")
        return result
    
    def purge_queue(self) -> Dict[str, Any]:
        """
        Purge all pending jobs from the queue
        FIXED: Now uses managed session with proper connection cleanup
        """
        logger.debug("running RunPodClient.purge_queue ... purging job queue")
        
        url = f"{self.base_url}/purge-queue"
        
        response = self._make_request('POST', url, timeout=30)
        result = response.json()
        
        logger.debug("running RunPodClient.purge_queue ... queue purged successfully")
        return result
    
    def status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a specific job
        FIXED: Now uses managed session with proper connection cleanup
        """
        logger.debug(f"running RunPodClient.status ... checking status of job {job_id}")
        
        url = f"{self.base_url}/status/{job_id}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Connection': 'close'  # Ensure connection is closed
        }
        
        response = self._make_request('GET', url, timeout=30, headers=headers)
        result = response.json()
        
        logger.debug(f"running RunPodClient.status ... status retrieved for job {job_id}")
        return result
    
    def stream(self, job_id: str) -> Iterator[Dict[str, Any]]:
        """Stream real-time updates for a job"""
        logger.debug(f"running RunPodClient.stream ... starting stream for job {job_id}")
        
        url = f"{self.base_url}/stream/{job_id}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'Connection': 'close'
        }
        
        response: Optional[Response] = None
        try:
            response = self.session.get(url, headers=headers, stream=True, timeout=30)
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
        finally:
            # Ensure the streaming response is properly closed
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
    
    def wait_for_completion(self, job_id: str, poll_interval: float = 1.0, max_wait: int = 600) -> Dict[str, Any]:
        """
        Convenience method to wait for job completion with polling
        Uses managed session with proper connection cleanup
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
        Uses managed session with proper connection cleanup
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
        
        # Handle both 'output' and 'execution_result' format
        if 'execution_result' in response:
            return response
        elif 'output' in response:
            return response.get('output', {})
        else:
            return response


class GPUProxyClient:
    """
    GPU Proxy Client with auto-setup capabilities
    
    This class handles all GPU proxy infrastructure concerns:
    - Detection of gpu-proxy installation
    - Auto-cloning from GitHub if needed
    - Environment setup and validation
    - Wrapping RunPod client with GPU-proxy-specific enhancements
    """
    
    def __init__(self, runpod_client: RunPodClient):
        """
        Initialize GPUProxyClient with a configured RunPodClient
        
        Args:
            runpod_client: Configured RunPodClient instance
        """
        self.runpod_client = runpod_client
        self.endpoint_id = runpod_client.endpoint_id
        
        logger.debug(f"running GPUProxyClient.__init__ ... initialized with endpoint {self.endpoint_id}")
    
    @classmethod
    def auto_setup(
        cls, 
        endpoint_id: Optional[str] = None,
        api_key: Optional[str] = None,
        auto_clone: bool = True,
        github_url: str = "https://github.com/TheBuleGanteng/gpu-proxy"
    ) -> 'GPUProxyClient':
        """
        Auto-setup GPU proxy with detection, cloning, and configuration
        
        Args:
            endpoint_id: RunPod endpoint ID (optional, reads from env if None)
            api_key: RunPod API key (optional, reads from env if None)
            auto_clone: Whether to auto-clone gpu-proxy if not found
            github_url: GitHub URL for gpu-proxy repository
            
        Returns:
            Configured GPUProxyClient instance
            
        Raises:
            RuntimeError: If GPU proxy setup fails and no fallback available
        """
        logger.debug("running GPUProxyClient.auto_setup ... starting auto-setup process")
        
        try:
            # Step 1: Detect or clone GPU proxy installation
            gpu_proxy_path = cls._detect_or_clone_gpu_proxy(auto_clone, github_url)
            logger.debug(f"running GPUProxyClient.auto_setup ... GPU proxy available at: {gpu_proxy_path}")
            
            # Step 2: Setup environment and load configuration
            cls._setup_gpu_proxy_environment(gpu_proxy_path)
            
            # Step 3: Create and validate RunPod client
            runpod_client = cls._create_runpod_client(endpoint_id, api_key)
            
            # Step 4: Validate GPU proxy functionality
            cls._validate_gpu_proxy_setup(runpod_client)
            
            # Step 5: Create GPUProxyClient wrapper
            gpu_proxy_client = cls(runpod_client)
            
            logger.debug("running GPUProxyClient.auto_setup ... auto-setup completed successfully")
            return gpu_proxy_client
            
        except Exception as e:
            logger.error(f"running GPUProxyClient.auto_setup ... auto-setup failed: {e}")
            raise RuntimeError(f"GPU proxy auto-setup failed: {e}")
    
    @staticmethod
    def _detect_or_clone_gpu_proxy(auto_clone: bool, github_url: str) -> Path:
        """
        Detect GPU proxy installation or clone if needed
        
        Args:
            auto_clone: Whether to auto-clone if not found
            github_url: GitHub URL for cloning
            
        Returns:
            Path to gpu-proxy directory
            
        Raises:
            RuntimeError: If gpu-proxy not found and auto_clone is False
        """
        logger.debug("running _detect_or_clone_gpu_proxy ... detecting GPU proxy installation")
        
        # Get current working directory (should be hyperparameter project root)
        current_dir = Path.cwd()
        
        # Check location 1: ./gpu-proxy (subdirectory of hyperparameter project)
        local_gpu_proxy = current_dir / "gpu-proxy"
        if local_gpu_proxy.exists() and local_gpu_proxy.is_dir():
            logger.debug(f"running _detect_or_clone_gpu_proxy ... found GPU proxy at: {local_gpu_proxy}")
            return local_gpu_proxy
        
        # Check location 2: ../gpu-proxy (sibling directory)
        sibling_gpu_proxy = current_dir.parent / "gpu-proxy"
        if sibling_gpu_proxy.exists() and sibling_gpu_proxy.is_dir():
            logger.debug(f"running _detect_or_clone_gpu_proxy ... found GPU proxy at: {sibling_gpu_proxy}")
            return sibling_gpu_proxy
        
        # Not found - try to clone if auto_clone enabled
        if auto_clone:
            logger.debug("running _detect_or_clone_gpu_proxy ... GPU proxy not found, attempting to clone")
            return GPUProxyClient._clone_gpu_proxy(github_url, sibling_gpu_proxy)
        else:
            raise RuntimeError(
                "GPU proxy not found in ./gpu-proxy or ../gpu-proxy and auto_clone=False. "
                f"Please clone manually: git clone {github_url}"
            )
    
    @staticmethod
    def _clone_gpu_proxy(github_url: str, target_path: Path) -> Path:
        """
        Clone GPU proxy repository from GitHub
        
        Args:
            github_url: GitHub repository URL
            target_path: Target directory for cloning
            
        Returns:
            Path to cloned gpu-proxy directory
            
        Raises:
            RuntimeError: If cloning fails
        """
        logger.debug(f"running _clone_gpu_proxy ... cloning GPU proxy from {github_url}")
        
        try:
            # Ensure parent directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Clone repository
            result = subprocess.run(
                ['git', 'clone', github_url, str(target_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")
            
            # Verify clone was successful
            if not target_path.exists() or not (target_path / '.git').exists():
                raise RuntimeError("Clone appeared successful but directory structure is invalid")
            
            logger.debug(f"running _clone_gpu_proxy ... GPU proxy cloned successfully to: {target_path}")
            return target_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Git clone timed out after 60 seconds")
        except FileNotFoundError:
            raise RuntimeError("Git not found - please install Git to enable auto-cloning")
        except Exception as e:
            raise RuntimeError(f"Failed to clone GPU proxy: {e}")
    
    @staticmethod
    def _setup_gpu_proxy_environment(gpu_proxy_path: Path) -> None:
        """
        Setup GPU proxy environment and load configuration
        
        Args:
            gpu_proxy_path: Path to gpu-proxy directory
        """
        logger.debug("running _setup_gpu_proxy_environment ... setting up GPU proxy environment")
        
        # Add gpu-proxy to Python path if not already there
        gpu_proxy_str = str(gpu_proxy_path)
        if gpu_proxy_str not in sys.path:
            sys.path.insert(0, gpu_proxy_str)
            logger.debug(f"running _setup_gpu_proxy_environment ... added to Python path: {gpu_proxy_str}")
        
        # Load GPU proxy environment file if it exists
        gpu_proxy_env_file = gpu_proxy_path / '.env'
        if gpu_proxy_env_file.exists():
            logger.debug(f"running _setup_gpu_proxy_environment ... loading GPU proxy .env file: {gpu_proxy_env_file}")
            load_dotenv(gpu_proxy_env_file)
        else:
            logger.debug("running _setup_gpu_proxy_environment ... no GPU proxy .env file found")
        
        logger.debug("running _setup_gpu_proxy_environment ... GPU proxy environment setup completed")
    
    @staticmethod
    def _create_runpod_client(endpoint_id: Optional[str], api_key: Optional[str]) -> RunPodClient:
        """
        Create and validate RunPod client
        
        Args:
            endpoint_id: RunPod endpoint ID
            api_key: RunPod API key
            
        Returns:
            Configured RunPodClient instance
        """
        logger.debug("running _create_runpod_client ... creating RunPod client")
        
        try:
            runpod_client = RunPodClient(endpoint_id=endpoint_id, api_key=api_key)
            logger.debug("running _create_runpod_client ... RunPod client created successfully")
            return runpod_client
            
        except Exception as e:
            logger.error(f"running _create_runpod_client ... failed to create RunPod client: {e}")
            raise RuntimeError(f"Failed to create RunPod client: {e}")
    
    @staticmethod
    def _validate_gpu_proxy_setup(runpod_client: RunPodClient) -> None:
        """
        Validate GPU proxy setup by running health check
        
        Args:
            runpod_client: RunPod client to validate
        """
        logger.debug("running _validate_gpu_proxy_setup ... validating GPU proxy setup")
        
        try:
            health_result = runpod_client.health()
            
            # Check for basic health indicators
            if not isinstance(health_result, dict):
                raise RuntimeError("Health check returned invalid response format")
            
            # Log health status for debugging
            workers = health_result.get('workers', {})
            ready_workers = workers.get('ready', 0)
            
            if ready_workers == 0:
                logger.warning("running _validate_gpu_proxy_setup ... no ready workers available")
            else:
                logger.debug(f"running _validate_gpu_proxy_setup ... {ready_workers} ready workers available")
            
            logger.debug("running _validate_gpu_proxy_setup ... GPU proxy validation completed")
            
        except Exception as e:
            logger.error(f"running _validate_gpu_proxy_setup ... validation failed: {e}")
            raise RuntimeError(f"GPU proxy validation failed: {e}")
    
    # Delegate all RunPod methods to the wrapped client
    def run(self, input_data: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """Submit asynchronous job (delegated to RunPodClient)"""
        return self.runpod_client.run(input_data, timeout)
    
    def runsync(self, input_data: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """Submit synchronous job (delegated to RunPodClient)"""
        return self.runpod_client.runsync(input_data, timeout)
    
    def health(self) -> Dict[str, Any]:
        """Check endpoint health (delegated to RunPodClient)"""
        return self.runpod_client.health()
    
    def cancel(self, job_id: str) -> Dict[str, Any]:
        """Cancel job (delegated to RunPodClient)"""
        return self.runpod_client.cancel(job_id)
    
    def purge_queue(self) -> Dict[str, Any]:
        """Purge job queue (delegated to RunPodClient)"""
        return self.runpod_client.purge_queue()
    
    def status(self, job_id: str) -> Dict[str, Any]:
        """Get job status (delegated to RunPodClient)"""
        return self.runpod_client.status(job_id)
    
    def stream(self, job_id: str) -> Iterator[Dict[str, Any]]:
        """Stream job updates (delegated to RunPodClient)"""
        return self.runpod_client.stream(job_id)
    
    def wait_for_completion(self, job_id: str, poll_interval: float = 1.0, max_wait: int = 600) -> Dict[str, Any]:
        """Wait for job completion (delegated to RunPodClient)"""
        return self.runpod_client.wait_for_completion(job_id, poll_interval, max_wait)
    
    def execute_code_sync(self, code: str, context: Optional[Dict[str, Any]] = None, 
                         timeout_seconds: int = 300) -> Dict[str, Any]:
        """Execute code synchronously (delegated to RunPodClient)"""
        return self.runpod_client.execute_code_sync(code, context, timeout_seconds)
    
    def cleanup(self) -> None:
        """Clean up resources (delegated to RunPodClient)"""
        self.runpod_client.cleanup()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        return False


# FIXED: Update convenience functions to use GPUProxyClient with auto-setup
def auto_setup_gpu_proxy(endpoint_id: Optional[str] = None, api_key: Optional[str] = None,
                         auto_clone: bool = True) -> GPUProxyClient:
    """Convenience function for GPUProxyClient.auto_setup()"""
    return GPUProxyClient.auto_setup(endpoint_id=endpoint_id, api_key=api_key, auto_clone=auto_clone)

# Legacy convenience functions (maintained for backward compatibility)
def run(input_data: Dict[str, Any], endpoint_id: Optional[str] = None, 
        api_key: Optional[str] = None, timeout: int = 300) -> Dict[str, Any]:
    """Convenience function for RunPodClient.run() - FIXED with auto cleanup"""
    with RunPodClient(endpoint_id, api_key) as client:
        return client.run(input_data, timeout)

def runsync(input_data: Dict[str, Any], endpoint_id: Optional[str] = None, 
            api_key: Optional[str] = None, timeout: int = 300) -> Dict[str, Any]:
    """Convenience function for RunPodClient.runsync() - FIXED with auto cleanup"""
    with RunPodClient(endpoint_id, api_key) as client:
        return client.runsync(input_data, timeout)

def health(endpoint_id: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for RunPodClient.health() - FIXED with auto cleanup"""
    with RunPodClient(endpoint_id, api_key) as client:
        return client.health()

def execute_code_sync(code: str, context: Optional[Dict[str, Any]] = None,
                     timeout_seconds: int = 300, endpoint_id: Optional[str] = None,
                     api_key: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for RunPodClient.execute_code_sync() - FIXED with auto cleanup"""
    with RunPodClient(endpoint_id, api_key) as client:
        return client.execute_code_sync(code, context, timeout_seconds)