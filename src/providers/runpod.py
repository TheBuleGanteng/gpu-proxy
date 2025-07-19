"""
RunPod Serverless GPU provider implementation.

This provider integrates with RunPod's serverless GPU infrastructure to provide
on-demand GPU training capabilities with automatic scaling and cost optimization.
"""

import asyncio
import aiohttp
import json
import time
import uuid
import base64
import pickle
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from pathlib import Path

from src.core.base import (
    GPUProviderBase, GPUProviderType, GPUResourceType, SessionStatus,
    GPUResourceInfo, CostEstimate, GPUMemoryInfo, TrainingConfig, TrainingResult
)
from src.utils.logger import logger


class RunPodServerlessProvider(GPUProviderBase):
    """
    RunPod Serverless implementation for on-demand GPU training.
    
    This provider uses RunPod's serverless infrastructure to execute training
    jobs on demand, providing cost-effective GPU access with automatic scaling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RunPod Serverless provider.
        
        Args:
            config: Configuration dictionary with:
                - api_key: str - RunPod API key (required)
                - endpoint_id: str - RunPod serverless endpoint ID (required)
                - gpu_type: str - Preferred GPU type (optional, default: "NVIDIA RTX A6000")
                - max_workers: int - Maximum concurrent workers (optional, default: 1)
                - timeout_seconds: int - Request timeout (optional, default: 3600)
                - polling_interval: float - Status polling interval (optional, default: 5.0)
                - region: str - Preferred region (optional)
        """
        super().__init__(config)
        
        # Required configuration
        self.api_key = config.get('api_key')
        self.endpoint_id = config.get('endpoint_id')
        
        if not self.api_key:
            raise ValueError("RunPod API key is required")
        if not self.endpoint_id:
            raise ValueError("RunPod endpoint ID is required")
        
        # Optional configuration
        self.gpu_type = config.get('gpu_type', "NVIDIA RTX A6000")
        self.max_workers = config.get('max_workers', 1)
        self.timeout_seconds = config.get('timeout_seconds', 3600)
        self.polling_interval = config.get('polling_interval', 5.0)
        self.region = config.get('region')
        
        # RunPod API configuration
        self.base_url = "https://api.runpod.ai/v2"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Session state
        self._session: Optional[aiohttp.ClientSession] = None
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        
        logger.debug(f"running __init__ ... initialized RunPodServerlessProvider with endpoint: {self.endpoint_id}")
    
    @property
    def provider_type(self) -> GPUProviderType:
        """Return RunPod Serverless provider type."""
        return GPUProviderType.RUNPOD_SERVERLESS
    
    async def connect(self) -> bool:
        """
        Establish connection to RunPod API and validate credentials.
        
        Returns:
            bool: True if connection successful
        """
        logger.debug("running connect ... connecting to RunPod Serverless API")
        
        try:
            self.status = SessionStatus.CONNECTING
            
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
            
            # Validate API key and endpoint
            if not await self._validate_endpoint():
                await self._cleanup_session()
                self.status = SessionStatus.ERROR
                return False
            
            # Generate session ID
            self.session_id = f"runpod-{uuid.uuid4().hex[:8]}"
            self.status = SessionStatus.CONNECTED
            
            logger.info(f"running connect ... connected to RunPod with session ID: {self.session_id}")
            self._trigger_callback('connection_established', {
                'session_id': self.session_id,
                'endpoint_id': self.endpoint_id
            })
            
            return True
            
        except Exception as e:
            logger.error(f"running connect ... connection failed: {e}")
            await self._cleanup_session()
            self.status = SessionStatus.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from RunPod API and clean up resources.
        
        Returns:
            bool: True if disconnection successful
        """
        logger.debug("running disconnect ... disconnecting from RunPod Serverless")
        
        try:
            # Cancel any active jobs
            if self._active_jobs:
                logger.warning(f"running disconnect ... cancelling {len(self._active_jobs)} active jobs")
                await self._cancel_all_jobs()
            
            # Clean up session
            await self._cleanup_session()
            
            self.status = SessionStatus.DISCONNECTED
            self.session_id = None
            
            logger.info("running disconnect ... disconnected from RunPod successfully")
            self._trigger_callback('disconnected', {})
            
            return True
            
        except Exception as e:
            logger.error(f"running disconnect ... disconnection error: {e}")
            return False
    
    async def get_available_resources(self) -> List[GPUResourceInfo]:
        """
        Get available GPU resources from RunPod.
        
        Note: RunPod serverless endpoints don't expose GPU type info directly,
        so we return estimated resource info based on typical serverless offerings.
        
        Returns:
            List[GPUResourceInfo]: Available GPU resources
        """
        logger.debug("running get_available_resources ... returning estimated serverless resources")
        
        if not await self.is_connected():
            logger.error("running get_available_resources ... not connected")
            return []
        
        # For serverless, we don't have direct access to GPU inventory
        # Return typical serverless GPU options
        estimated_resources = [
            GPUResourceInfo(
                resource_type=GPUResourceType.RTX_4090,
                memory_gb=24.0,
                compute_units=128,
                cost_per_hour=1.20,
                availability=True,
                location="RunPod Serverless",
                estimated_performance=100.0
            ),
            GPUResourceInfo(
                resource_type=GPUResourceType.A100_40GB,
                memory_gb=40.0,
                compute_units=108,
                cost_per_hour=2.50,
                availability=True,
                location="RunPod Serverless",
                estimated_performance=120.0
            ),
            GPUResourceInfo(
                resource_type=GPUResourceType.RTX_3090,
                memory_gb=24.0,
                compute_units=82,
                cost_per_hour=0.80,
                availability=True,
                location="RunPod Serverless",
                estimated_performance=85.0
            )
        ]
        
        logger.info(f"running get_available_resources ... returning {len(estimated_resources)} estimated serverless resources")
        
        return estimated_resources
    
    async def estimate_cost(self, config: TrainingConfig) -> CostEstimate:
        """
        Estimate cost for training operation on RunPod.
        
        Args:
            config: Training configuration
            
        Returns:
            CostEstimate: Cost estimation
        """
        logger.debug("running estimate_cost ... calculating RunPod cost estimate")
        
        if not await self.validate_config(config):
            raise ValueError("Invalid training configuration")
        
        try:
            # Base cost calculation
            # These are estimates - actual costs depend on GPU type and region
            base_cost_per_second = 0.0002  # ~$0.72/hour for mid-tier GPU
            
            # Estimate training time based on model complexity
            model_complexity = self._estimate_model_complexity(config)
            base_time_per_epoch = 45.0  # seconds for reference model
            
            estimated_duration = config.max_epochs * base_time_per_epoch * model_complexity
            
            # Add overhead for cold starts and data transfer
            overhead_seconds = 60.0  # 1 minute overhead
            total_duration = estimated_duration + overhead_seconds
            
            # Calculate costs
            compute_cost = total_duration * base_cost_per_second
            data_transfer_cost = compute_cost * 0.05  # 5% for data transfer
            total_cost = compute_cost + data_transfer_cost
            
            cost_estimate = CostEstimate(
                estimated_duration_seconds=total_duration,
                cost_per_second=base_cost_per_second,
                total_estimated_cost=total_cost,
                currency="USD",
                breakdown={
                    "compute_cost": compute_cost,
                    "data_transfer": data_transfer_cost,
                    "cold_start_overhead": overhead_seconds * base_cost_per_second
                }
            )
            
            logger.info(f"running estimate_cost ... estimated ${total_cost:.4f} for {total_duration:.1f}s")
            
            return cost_estimate
            
        except Exception as e:
            logger.error(f"running estimate_cost ... error: {e}")
            raise
    
    async def train_model(self, config: TrainingConfig) -> TrainingResult:
        """
        Execute model training on RunPod serverless infrastructure.
        
        Args:
            config: Training configuration
            
        Returns:
            TrainingResult: Training results
        """
        logger.debug("running train_model ... starting RunPod serverless training")
        
        if not await self.validate_config(config):
            return TrainingResult(
                success=False,
                error_message="Invalid training configuration"
            )
        
        if not await self.is_connected():
            return TrainingResult(
                success=False,
                error_message="Not connected to RunPod"
            )
        
        self.status = SessionStatus.BUSY
        start_time = time.time()
        job_id = None
        
        try:
            # Submit training job to RunPod
            job_id = await self._submit_training_job(config)
            if not job_id:
                return TrainingResult(
                    success=False,
                    error_message="Failed to submit training job",
                    duration_seconds=time.time() - start_time
                )
            
            # Monitor job progress
            result = await self._monitor_training_job(job_id, config, start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"running train_model ... training failed: {e}")
            
            # Cleanup failed job
            if job_id and job_id in self._active_jobs:
                await self._cancel_job(job_id)
            
            return TrainingResult(
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
        
        finally:
            self.status = SessionStatus.CONNECTED
    
    async def get_memory_info(self) -> Optional[GPUMemoryInfo]:
        """
        Get GPU memory information from RunPod.
        
        Note: RunPod serverless doesn't provide real-time memory info,
        so this returns estimated usage based on active jobs.
        
        Returns:
            Optional[GPUMemoryInfo]: Estimated memory usage
        """
        if not await self.is_connected():
            return None
        
        # Estimate memory usage based on active jobs
        active_job_count = len(self._active_jobs)
        
        # Typical GPU memory for serverless instances
        total_memory = 24576.0  # 24GB typical for A6000
        estimated_usage = min(active_job_count * 8192.0, total_memory)  # 8GB per job estimate
        
        memory_info = GPUMemoryInfo(
            allocated_mb=estimated_usage,
            cached_mb=estimated_usage * 0.1,
            total_mb=total_memory,
            utilization_percent=(estimated_usage / total_memory) * 100
        )
        
        logger.debug(f"running get_memory_info ... estimated memory usage: {memory_info.utilization_percent:.1f}%")
        
        return memory_info
    
    async def run_sync(self, config: TrainingConfig) -> TrainingResult:
        """
        Execute synchronous training job (blocks until completion).
        
        Args:
            config: Training configuration
            
        Returns:
            TrainingResult: Training results
        """
        logger.debug("running run_sync ... starting synchronous RunPod training")
        
        if not await self.validate_config(config):
            return TrainingResult(
                success=False,
                error_message="Invalid training configuration"
            )
        
        if not await self.is_connected():
            return TrainingResult(
                success=False,
                error_message="Not connected to RunPod"
            )
        
        try:
            job_payload = {
                "input": {
                    "training_config": self._serialize_config(config),
                    "framework": config.framework,
                    "max_epochs": config.max_epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate
                }
            }
            
            url = f"{self.base_url}/{self.endpoint_id}/runsync"
            
            logger.info("running run_sync ... submitting synchronous job")
            
            async with self._session.post(url, json=job_payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Process synchronous result
                    if 'output' in data:
                        result_data = self._deserialize_result(data['output'].get('training_result', ''))
                        return TrainingResult(**result_data)
                    else:
                        return TrainingResult(
                            success=False,
                            error_message="No output received from synchronous job"
                        )
                else:
                    error_text = await response.text()
                    logger.error(f"running run_sync ... sync job failed: {response.status} - {error_text}")
                    return TrainingResult(
                        success=False,
                        error_message=f"Sync job failed: {response.status}"
                    )
                    
        except Exception as e:
            logger.error(f"running run_sync ... error: {e}")
            return TrainingResult(
                success=False,
                error_message=f"Sync job error: {str(e)}"
            )
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check endpoint health status.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        logger.debug("running check_health ... checking endpoint health")
        
        if not await self.is_connected():
            return {"status": "disconnected", "healthy": False}
        
        try:
            url = f"{self.base_url}/{self.endpoint_id}/health"
            
            async with self._session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug("running check_health ... endpoint is healthy")
                    return {"status": "healthy", "healthy": True, "details": data}
                else:
                    logger.warning(f"running check_health ... health check failed: {response.status}")
                    return {"status": "unhealthy", "healthy": False, "status_code": response.status}
                    
        except Exception as e:
            logger.error(f"running check_health ... health check error: {e}")
            return {"status": "error", "healthy": False, "error": str(e)}
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a specific running job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            bool: True if cancellation successful
        """
        logger.info(f"running cancel_job ... cancelling job {job_id}")
        
        # Check if we have an active session, not just connected status
        if not self._session or self._session.closed:
            logger.error("running cancel_job ... no active session")
            return False
        
        try:
            url = f"{self.base_url}/{self.endpoint_id}/cancel/{job_id}"
            
            async with self._session.post(url) as response:
                if response.status == 200:
                    logger.info(f"running cancel_job ... job {job_id} cancelled successfully")
                    
                    # Remove from active jobs tracking
                    if job_id in self._active_jobs:
                        del self._active_jobs[job_id]
                    
                    self._trigger_callback('job_cancelled', {'job_id': job_id})
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"running cancel_job ... cancellation failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"running cancel_job ... error: {e}")
            return False
    
    async def purge_queue(self) -> bool:
        """
        Purge all jobs in the queue for this endpoint.
        
        Returns:
            bool: True if purge successful
        """
        logger.info("running purge_queue ... purging endpoint queue")
        
        if not await self.is_connected():
            logger.error("running purge_queue ... not connected")
            return False
        
        try:
            url = f"{self.base_url}/{self.endpoint_id}/purge-queue"
            
            async with self._session.post(url) as response:
                if response.status == 200:
                    logger.info("running purge_queue ... queue purged successfully")
                    
                    # Clear all active jobs
                    purged_jobs = list(self._active_jobs.keys())
                    self._active_jobs.clear()
                    
                    self._trigger_callback('queue_purged', {'purged_jobs': purged_jobs})
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"running purge_queue ... purge failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"running purge_queue ... error: {e}")
            return False
    
    async def stream_job(self, job_id: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream job output/progress in real-time.
        
        Args:
            job_id: Job ID to stream
            
        Yields:
            Dict[str, Any]: Streaming job data
        """
        logger.info(f"running stream_job ... starting stream for job {job_id}")
        
        if not await self.is_connected():
            yield {"error": "Not connected to RunPod"}
            return
        
        try:
            url = f"{self.base_url}/{self.endpoint_id}/stream/{job_id}"
            
            async with self._session.get(url) as response:
                if response.status == 200:
                    async for line in response.content:
                        try:
                            # Parse streaming JSON data
                            data = json.loads(line.decode('utf-8'))
                            logger.debug(f"running stream_job ... received stream data: {data}")
                            
                            self._trigger_callback('stream_data', {
                                'job_id': job_id,
                                'data': data
                            })
                            
                            yield data
                            
                        except json.JSONDecodeError:
                            # Skip non-JSON lines
                            continue
                        except Exception as e:
                            logger.error(f"running stream_job ... stream parsing error: {e}")
                            yield {"error": f"Stream parsing error: {str(e)}"}
                            break
                else:
                    error_text = await response.text()
                    logger.error(f"running stream_job ... stream failed: {response.status} - {error_text}")
                    yield {"error": f"Stream failed: {response.status}"}
                    
        except Exception as e:
            logger.error(f"running stream_job ... error: {e}")
            yield {"error": f"Stream error: {str(e)}"}
    
    # Update the existing _cancel_job method to use the new cancel_job method
    async def _cancel_job(self, job_id: str) -> bool:
        """Cancel a specific job (internal method)."""
        return await self.cancel_job(job_id)
    
    async def _validate_endpoint(self) -> bool:
        """Validate API key and endpoint."""
        try:
            # For RunPod serverless, we can't easily validate without submitting a job
            # So we'll do a simple test - try to submit a minimal test job
            test_payload = {"input": {"test": "validation"}}
            url = f"{self.base_url}/{self.endpoint_id}/run"
            
            async with self._session.post(url, json=test_payload) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    logger.debug(f"running _validate_endpoint ... endpoint validation successful")
                    return True
                elif response.status == 401:
                    logger.error(f"running _validate_endpoint ... invalid API key: {response.status}")
                    return False
                elif response.status == 404:
                    logger.error(f"running _validate_endpoint ... endpoint not found: {self.endpoint_id}")
                    return False
                else:
                    # Other errors might be due to serverless function issues, but endpoint exists
                    logger.warning(f"running _validate_endpoint ... endpoint exists but returned: {response.status}")
                    logger.debug(f"running _validate_endpoint ... response: {response_text}")
                    return True  # Assume endpoint is valid, issue is with serverless function
                    
        except Exception as e:
            logger.error(f"running _validate_endpoint ... validation error: {e}")
            return False
    
    async def _submit_training_job(self, config: TrainingConfig) -> Optional[str]:
        """Submit training job to RunPod serverless."""
        try:
            # Prepare job payload
            job_payload = {
                "input": {
                    "training_config": self._serialize_config(config),
                    "framework": config.framework,
                    "max_epochs": config.max_epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate
                }
            }
            
            # Use the correct RunPod serverless API endpoint
            url = f"{self.base_url}/{self.endpoint_id}/run"
            
            logger.info(f"running _submit_training_job ... submitting job to RunPod endpoint")
            
            async with self._session.post(url, json=job_payload) as response:
                response_text = await response.text()
                logger.debug(f"running _submit_training_job ... response status: {response.status}")
                logger.debug(f"running _submit_training_job ... response body: {response_text}")
                
                if response.status == 200:
                    data = await response.json() if response_text else {}
                    job_id = data.get('id')
                    
                    if job_id:
                        self._active_jobs[job_id] = {
                            'config': config,
                            'start_time': time.time(),
                            'status': 'SUBMITTED'
                        }
                        logger.info(f"running _submit_training_job ... job submitted with ID: {job_id}")
                        
                        self._trigger_callback('job_submitted', {
                            'job_id': job_id,
                            'endpoint_id': self.endpoint_id
                        })
                    
                    return job_id
                else:
                    logger.error(f"running _submit_training_job ... submission failed: {response.status} - {response_text}")
                    return None
                
        except Exception as e:
            logger.error(f"running _submit_training_job ... error: {e}")
            return None
    
    async def _monitor_training_job(self, job_id: str, config: TrainingConfig, start_time: float) -> TrainingResult:
        """Monitor training job progress and return results."""
        logger.info(f"running _monitor_training_job ... monitoring job {job_id}")
        
        last_progress_time = time.time()
        timeout_time = start_time + self.timeout_seconds
        
        try:
            while time.time() < timeout_time:
                # Check job status
                status_data = await self._get_job_status(job_id)
                
                if not status_data:
                    await asyncio.sleep(self.polling_interval)
                    continue
                
                job_status = status_data.get('status', 'UNKNOWN')
                
                # Update active job status
                if job_id in self._active_jobs:
                    self._active_jobs[job_id]['status'] = job_status
                
                # Handle different job states
                if job_status == 'COMPLETED':
                    logger.info(f"running _monitor_training_job ... job {job_id} completed")
                    return await self._process_completed_job(job_id, status_data, start_time)
                
                elif job_status == 'FAILED':
                    error_msg = status_data.get('output', {}).get('error', 'Unknown error')
                    logger.error(f"running _monitor_training_job ... job {job_id} failed: {error_msg}")
                    return TrainingResult(
                        success=False,
                        error_message=f"RunPod job failed: {error_msg}",
                        duration_seconds=time.time() - start_time
                    )
                
                elif job_status in ['IN_PROGRESS', 'IN_QUEUE']:
                    # Check for progress updates
                    if time.time() - last_progress_time > 30:  # Log progress every 30 seconds
                        logger.info(f"running _monitor_training_job ... job {job_id} status: {job_status}")
                        last_progress_time = time.time()
                        
                        self._trigger_callback('training_progress', {
                            'job_id': job_id,
                            'status': job_status,
                            'elapsed_time': time.time() - start_time
                        })
                
                await asyncio.sleep(self.polling_interval)
            
            # Timeout reached
            logger.error(f"running _monitor_training_job ... job {job_id} timed out after {self.timeout_seconds}s")
            await self._cancel_job(job_id)
            
            return TrainingResult(
                success=False,
                error_message=f"Training job timed out after {self.timeout_seconds} seconds",
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"running _monitor_training_job ... monitoring error: {e}")
            return TrainingResult(
                success=False,
                error_message=f"Job monitoring failed: {str(e)}",
                duration_seconds=time.time() - start_time
            )
        
        finally:
            # Clean up job tracking
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]
    
    async def _get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status from RunPod API."""
        try:
            url = f"{self.base_url}/{self.endpoint_id}/status/{job_id}"
            async with self._session.get(url) as response:
                response_text = await response.text()
                logger.debug(f"running _get_job_status ... status check response: {response.status}")
                
                if response.status == 200:
                    return await response.json() if response_text else {}
                else:
                    logger.warning(f"running _get_job_status ... status check failed: {response.status} - {response_text}")
                    return None
        except Exception as e:
            logger.error(f"running _get_job_status ... error: {e}")
            return None
    
    async def _process_completed_job(self, job_id: str, status_data: Dict[str, Any], start_time: float) -> TrainingResult:
        """Process completed job results."""
        try:
            output = status_data.get('output', {})
            
            # Deserialize training results
            if 'training_result' in output:
                result_data = self._deserialize_result(output['training_result'])
                
                # Update timing
                result_data['duration_seconds'] = time.time() - start_time
                
                # Add cost information
                if 'execution_time' in output:
                    execution_time = output['execution_time']
                    estimated_cost = execution_time * 0.0002  # Rough estimate
                    
                    result_data['cost_info'] = CostEstimate(
                        estimated_duration_seconds=execution_time,
                        cost_per_second=0.0002,
                        total_estimated_cost=estimated_cost,
                        currency="USD"
                    )
                
                logger.info(f"running _process_completed_job ... job {job_id} results processed successfully")
                
                self._trigger_callback('training_complete', {
                    'job_id': job_id,
                    'success': result_data.get('success', False),
                    'final_accuracy': result_data.get('final_accuracy'),
                    'duration': result_data.get('duration_seconds')
                })
                
                return TrainingResult(**result_data)
            
            else:
                logger.error(f"running _process_completed_job ... no training results in output")
                return TrainingResult(
                    success=False,
                    error_message="No training results returned from RunPod",
                    duration_seconds=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"running _process_completed_job ... error processing results: {e}")
            return TrainingResult(
                success=False,
                error_message=f"Failed to process job results: {str(e)}",
                duration_seconds=time.time() - start_time
            )
    
    async def _cancel_job(self, job_id: str) -> bool:
        """Cancel a specific job."""
        try:
            url = f"{self.base_url}/{self.endpoint_id}/cancel/{job_id}"
            async with self._session.post(url) as response:
                if response.status == 200:
                    logger.info(f"running _cancel_job ... cancelled job {job_id}")
                    return True
                else:
                    logger.warning(f"running _cancel_job ... cancel failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"running _cancel_job ... error: {e}")
            return False
    
    async def _cancel_all_jobs(self) -> None:
        """Cancel all active jobs."""
        jobs_to_cancel = list(self._active_jobs.keys())
        for job_id in jobs_to_cancel:
            await self._cancel_job(job_id)
        self._active_jobs.clear()
    
    async def _cleanup_session(self) -> None:
        """Clean up HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("running _cleanup_session ... HTTP session cleaned up")
    
    def _serialize_config(self, config: TrainingConfig) -> str:
        """Serialize training configuration for transmission."""
        config_dict = {
            'model_config': config.model_config,
            'training_params': config.training_params,
            'data_config': config.data_config,
            'framework': config.framework,
            'max_epochs': config.max_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'early_stopping': config.early_stopping,
            'save_checkpoints': config.save_checkpoints
        }
        
        # Base64 encode pickled config for safe transmission
        pickled_data = pickle.dumps(config_dict)
        encoded_data = base64.b64encode(pickled_data).decode('utf-8')
        
        return encoded_data
    
    def _deserialize_result(self, encoded_result: str) -> Dict[str, Any]:
        """Deserialize training result from transmission."""
        try:
            pickled_data = base64.b64decode(encoded_result.encode('utf-8'))
            result_dict = pickle.loads(pickled_data)
            return result_dict
        except Exception as e:
            logger.error(f"running _deserialize_result ... deserialization error: {e}")
            return {'success': False, 'error_message': 'Failed to deserialize results'}
    
    def _map_gpu_type(self, gpu_name: str) -> GPUResourceType:
        """Map RunPod GPU names to our enum types."""
        gpu_mapping = {
            'RTX A6000': GPUResourceType.RTX_4090,  # Close equivalent
            'RTX 3090': GPUResourceType.RTX_3090,
            'RTX 3080': GPUResourceType.RTX_3080,
            'RTX 4090': GPUResourceType.RTX_4090,
            'A100 40GB': GPUResourceType.A100_40GB,
            'A100 80GB': GPUResourceType.A100_80GB,
            'V100': GPUResourceType.V100,
            'T4': GPUResourceType.T4
        }
        
        for name_pattern, gpu_type in gpu_mapping.items():
            if name_pattern.lower() in gpu_name.lower():
                return gpu_type
        
        return GPUResourceType.AUTO  # Default fallback
    
    def _estimate_performance(self, gpu_name: str) -> float:
        """Estimate relative performance score for GPU."""
        performance_scores = {
            'A100': 100.0,
            'RTX 4090': 85.0,
            'RTX A6000': 75.0,
            'RTX 3090': 70.0,
            'RTX 3080': 60.0,
            'V100': 50.0,
            'T4': 30.0
        }
        
        for gpu_pattern, score in performance_scores.items():
            if gpu_pattern.lower() in gpu_name.lower():
                return score
        
        return 50.0  # Default score
    
    def _estimate_model_complexity(self, config: TrainingConfig) -> float:
        """Estimate model complexity multiplier for cost calculation."""
        complexity = 1.0
        
        # Analyze model configuration
        if config.model_config:
            # Count layers/parameters
            layer_count = len(config.model_config.get('layers', []))
            complexity *= (1.0 + layer_count * 0.1)
            
            # Check for complex architectures
            model_type = config.model_config.get('type', '').lower()
            if 'resnet' in model_type or 'transformer' in model_type:
                complexity *= 2.0
            elif 'vgg' in model_type or 'inception' in model_type:
                complexity *= 1.5
        
        # Factor in batch size
        if config.batch_size > 64:
            complexity *= 1.2
        elif config.batch_size < 16:
            complexity *= 0.8
        
        return min(complexity, 5.0)  # Cap at 5x complexity