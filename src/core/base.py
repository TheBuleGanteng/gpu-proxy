"""
Abstract base classes for GPU proxy providers.

This module defines the core interfaces that all GPU providers must implement,
ensuring consistent behavior across different cloud providers and local GPU resources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import time

from src.utils.logger import logger


class GPUProviderType(Enum):
    """Supported GPU provider types."""
    RUNPOD_SERVERLESS = "runpod_serverless"
    RUNPOD_POD = "runpod_pod"
    LOCAL = "local"
    GCP_VERTEX = "gcp_vertex"
    AWS_SAGEMAKER = "aws_sagemaker"
    AZURE_ML = "azure_ml"
    MOCK = "mock"


class GPUResourceType(Enum):
    """Available GPU resource types."""
    RTX_3070 = "RTX3070"
    RTX_3080 = "RTX3080"
    RTX_3090 = "RTX3090"
    RTX_4070 = "RTX4070"
    RTX_4080 = "RTX4080"
    RTX_4090 = "RTX4090"
    A100_40GB = "A100_40GB"
    A100_80GB = "A100_80GB"
    V100 = "V100"
    T4 = "T4"
    AUTO = "AUTO"  # Let provider choose best available


class SessionStatus(Enum):
    """GPU session status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class GPUResourceInfo:
    """Information about available GPU resources."""
    resource_type: GPUResourceType
    memory_gb: float
    compute_units: int
    cost_per_hour: Optional[float] = None
    availability: bool = True
    location: Optional[str] = None
    estimated_performance: Optional[float] = None  # Relative performance score


@dataclass
class CostEstimate:
    """Cost estimation for GPU operations."""
    estimated_duration_seconds: float
    cost_per_second: float
    total_estimated_cost: float
    currency: str = "USD"
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class GPUMemoryInfo:
    """GPU memory usage information."""
    allocated_mb: float
    cached_mb: float
    total_mb: float
    utilization_percent: float


@dataclass
class TrainingConfig:
    """Configuration for training operations."""
    model_config: Dict[str, Any]
    training_params: Dict[str, Any]
    data_config: Dict[str, Any]
    framework: str  # "tensorflow", "pytorch", "jax"
    max_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping: bool = True
    save_checkpoints: bool = False


@dataclass
class TrainingResult:
    """Results from training operations."""
    success: bool
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    model_state: Optional[Any] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    memory_usage: Optional[GPUMemoryInfo] = None
    cost_info: Optional[CostEstimate] = None


class GPUProviderBase(ABC):
    """
    Abstract base class for all GPU providers.
    
    This defines the core interface that enables transparent GPU access
    across different cloud providers and local resources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GPU provider.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        logger.debug(f"running __init__ ... initializing {self.__class__.__name__}")
        self.config = config
        self.session_id: Optional[str] = None
        self.status = SessionStatus.DISCONNECTED
        self._callbacks: Dict[str, List[Callable]] = {}
        
    @property
    @abstractmethod
    def provider_type(self) -> GPUProviderType:
        """Return the provider type."""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to GPU resources.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from GPU resources and clean up.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_available_resources(self) -> List[GPUResourceInfo]:
        """
        Get list of available GPU resources.
        
        Returns:
            List[GPUResourceInfo]: Available GPU resources
        """
        pass
    
    @abstractmethod
    async def estimate_cost(self, config: TrainingConfig) -> CostEstimate:
        """
        Estimate cost for training operation.
        
        Args:
            config: Training configuration
            
        Returns:
            CostEstimate: Estimated cost information
        """
        pass
    
    @abstractmethod
    async def train_model(self, config: TrainingConfig) -> TrainingResult:
        """
        Execute training on GPU resources.
        
        Args:
            config: Training configuration
            
        Returns:
            TrainingResult: Training results and metrics
        """
        pass
    
    @abstractmethod
    async def get_memory_info(self) -> Optional[GPUMemoryInfo]:
        """
        Get current GPU memory usage information.
        
        Returns:
            Optional[GPUMemoryInfo]: Memory usage info if available
        """
        pass
    
    # Session management methods
    async def is_connected(self) -> bool:
        """Check if provider is connected and ready."""
        return self.status == SessionStatus.CONNECTED
    
    async def health_check(self) -> bool:
        """
        Perform health check on GPU resources.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        logger.debug("running health_check ... performing basic health check")
        try:
            if not await self.is_connected():
                return False
            
            # Basic memory check
            memory_info = await self.get_memory_info()
            if memory_info is None:
                return False
                
            return memory_info.utilization_percent < 95.0  # Not completely maxed out
            
        except Exception as e:
            logger.error(f"running health_check ... health check failed: {e}")
            return False
    
    # Callback system for monitoring
    def add_callback(self, event: str, callback: Callable) -> None:
        """
        Add callback for specific events.
        
        Args:
            event: Event name (e.g., 'training_progress', 'error', 'completion')
            callback: Callback function to execute
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
        logger.debug(f"running add_callback ... added callback for event: {event}")
    
    def _trigger_callback(self, event: str, data: Any = None) -> None:
        """Trigger callbacks for specific event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"running _trigger_callback ... callback error for {event}: {e}")
    
    # Utility methods
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        return {
            "provider_type": self.provider_type.value,
            "session_id": self.session_id,
            "status": self.status.value,
            "config": self.config
        }
    
    async def validate_config(self, config: TrainingConfig) -> bool:
        """
        Validate training configuration for this provider.
        
        Args:
            config: Training configuration to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        logger.debug("running validate_config ... validating training configuration")
        
        # Basic validation
        if not config.model_config:
            logger.error("running validate_config ... model_config is empty")
            return False
        
        if not config.framework in ["tensorflow", "pytorch", "jax"]:
            logger.error(f"running validate_config ... unsupported framework: {config.framework}")
            return False
        
        if config.max_epochs <= 0:
            logger.error(f"running validate_config ... invalid max_epochs: {config.max_epochs}")
            return False
        
        if config.batch_size <= 0:
            logger.error(f"running validate_config ... invalid batch_size: {config.batch_size}")
            return False
        
        return True


class RemoteGPU:
    """
    Main interface for transparent GPU access.
    
    This is the primary class that users will interact with to access
    remote GPU resources as if they were local.
    """
    
    def __init__(self, 
                 provider: Union[str, GPUProviderType] = GPUProviderType.RUNPOD_SERVERLESS,
                 config: Optional[Dict[str, Any]] = None,
                 auto_connect: bool = True):
        """
        Initialize RemoteGPU interface.
        
        Args:
            provider: GPU provider type
            config: Provider-specific configuration
            auto_connect: Whether to automatically connect on initialization
        """
        logger.debug(f"running __init__ ... initializing RemoteGPU with provider: {provider}")
        
        if isinstance(provider, str):
            provider = GPUProviderType(provider)
        
        self.provider_type = provider
        self.config = config or {}
        self._provider: Optional[GPUProviderBase] = None
        self._auto_connect = auto_connect
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> bool:
        """
        Connect to GPU provider.
        
        Returns:
            bool: True if connection successful
        """
        logger.debug("running connect ... connecting to GPU provider")
        
        # Import and instantiate provider (dynamic import to avoid circular dependencies)
        if self.provider_type == GPUProviderType.RUNPOD_SERVERLESS:
            from src.providers.runpod import RunPodServerlessProvider
            self._provider = RunPodServerlessProvider(self.config)
        elif self.provider_type == GPUProviderType.LOCAL:
            from src.providers.local import LocalGPUProvider
            self._provider = LocalGPUProvider(self.config)
        elif self.provider_type == GPUProviderType.MOCK:
            from src.providers.mock import MockGPUProvider
            self._provider = MockGPUProvider(self.config)
        else:
            logger.error(f"running connect ... unsupported provider: {self.provider_type}")
            return False
        
        return await self._provider.connect()
    
    async def disconnect(self) -> bool:
        """Disconnect from GPU provider."""
        if self._provider:
            logger.debug("running disconnect ... disconnecting from GPU provider")
            return await self._provider.disconnect()
        return True
    
    async def train_model(self, config: TrainingConfig) -> TrainingResult:
        """
        Train model on remote GPU.
        
        Args:
            config: Training configuration
            
        Returns:
            TrainingResult: Training results
        """
        if not self._provider:
            if self._auto_connect:
                await self.connect()
            else:
                raise RuntimeError("Not connected to GPU provider. Call connect() first.")
        
        logger.debug("running train_model ... starting model training")
        return await self._provider.train_model(config)
    
    async def get_available_resources(self) -> List[GPUResourceInfo]:
        """Get available GPU resources."""
        if not self._provider:
            if self._auto_connect:
                await self.connect()
            else:
                raise RuntimeError("Not connected to GPU provider. Call connect() first.")
        
        return await self._provider.get_available_resources()
    
    async def estimate_cost(self, config: TrainingConfig) -> CostEstimate:
        """Estimate cost for training operation."""
        if not self._provider:
            if self._auto_connect:
                await self.connect()
            else:
                raise RuntimeError("Not connected to GPU provider. Call connect() first.")
        
        return await self._provider.estimate_cost(config)
    
    def add_progress_callback(self, callback: Callable) -> None:
        """Add callback for training progress updates."""
        if self._provider:
            self._provider.add_callback('training_progress', callback)
    
    def add_error_callback(self, callback: Callable) -> None:
        """Add callback for error notifications."""
        if self._provider:
            self._provider.add_callback('error', callback)
    
    async def health_check(self) -> bool:
        """Perform health check on GPU provider."""
        if self._provider:
            return await self._provider.health_check()
        return False
    
    async def run_sync(self, config: TrainingConfig) -> TrainingResult:
        """
        Execute synchronous training (blocks until completion).
        
        Args:
            config: Training configuration
            
        Returns:
            TrainingResult: Training results
        """
        if not self._provider:
            if self._auto_connect:
                await self.connect()
            else:
                raise RuntimeError("Not connected to GPU provider. Call connect() first.")
        
        if hasattr(self._provider, 'run_sync'):
            return await self._provider.run_sync(config)
        else:
            # Fallback to regular training for providers that don't support sync
            return await self._provider.train_model(config)
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            bool: True if cancellation successful
        """
        if not self._provider:
            return False
        
        if hasattr(self._provider, 'cancel_job'):
            return await self._provider.cancel_job(job_id)
        return False
    
    async def purge_queue(self) -> bool:
        """
        Purge all jobs in the queue.
        
        Returns:
            bool: True if purge successful
        """
        if not self._provider:
            return False
        
        if hasattr(self._provider, 'purge_queue'):
            return await self._provider.purge_queue()
        return False
    
    async def check_endpoint_health(self) -> Dict[str, Any]:
        """
        Check endpoint health status.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        if not self._provider:
            return {"status": "disconnected", "healthy": False}
        
        if hasattr(self._provider, 'check_health'):
            return await self._provider.check_health()
        return {"status": "not_supported", "healthy": True}
    
    async def stream_job_output(self, job_id: str):
        """
        Stream job output in real-time.
        
        Args:
            job_id: Job ID to stream
            
        Yields:
            Dict[str, Any]: Streaming job data
        """
        if not self._provider:
            yield {"error": "Not connected to GPU provider"}
            return
        
        if hasattr(self._provider, 'stream_job'):
            async for data in self._provider.stream_job(job_id):
                yield data
        else:
            yield {"error": "Streaming not supported by this provider"}