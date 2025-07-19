"""
Mock GPU provider implementation for testing and development.

This provider simulates GPU operations with realistic delays and responses,
allowing for development and testing without actual GPU resources.
"""
import asyncio
import random
import time
import uuid
from typing import Any, Dict, List, Optional

from src.core.base import (
    GPUProviderBase, GPUProviderType, GPUResourceType, SessionStatus,
    GPUResourceInfo, CostEstimate, GPUMemoryInfo, TrainingConfig, TrainingResult
)
from src.utils.logger import logger


class MockGPUProvider(GPUProviderBase):
    """
    Mock implementation of GPU provider for testing and development.
    
    Simulates realistic GPU operations including:
    - Connection delays
    - Training progress with callbacks
    - Memory usage simulation
    - Cost estimation
    - Random failures for error testing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mock GPU provider.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - simulate_failures: bool (default False) - Whether to randomly fail operations
                - failure_rate: float (default 0.1) - Probability of random failures
                - connection_delay: float (default 1.0) - Simulated connection delay in seconds
                - training_speed_multiplier: float (default 1.0) - Speed up/slow down training simulation
        """
        super().__init__(config)
        
        # Mock configuration parameters
        self.simulate_failures = config.get('simulate_failures', False)
        self.failure_rate = config.get('failure_rate', 0.1)
        self.connection_delay = config.get('connection_delay', 1.0)
        self.training_speed_multiplier = config.get('training_speed_multiplier', 1.0)
        
        # Mock state
        self._mock_memory = GPUMemoryInfo(
            allocated_mb=0.0,
            cached_mb=0.0,
            total_mb=11264.0,  # Simulate RTX 3080 with 11GB
            utilization_percent=0.0
        )
        self._is_training = False
        self._training_progress = 0.0
        
        logger.debug("running __init__ ... initialized MockGPUProvider")
    
    @property
    def provider_type(self) -> GPUProviderType:
        """Return mock provider type."""
        return GPUProviderType.MOCK
    
    async def connect(self) -> bool:
        """
        Simulate connection to GPU resources.
        
        Returns:
            bool: True if connection successful
        """
        logger.debug("running connect ... starting mock connection")
        
        # Check for simulated failure
        if self._should_simulate_failure():
            logger.error("running connect ... simulated connection failure")
            self.status = SessionStatus.ERROR
            return False
        
        self.status = SessionStatus.CONNECTING
        
        # Simulate connection delay
        await asyncio.sleep(self.connection_delay)
        
        # Generate mock session ID
        self.session_id = f"mock-{uuid.uuid4().hex[:8]}"
        self.status = SessionStatus.CONNECTED
        
        logger.info(f"running connect ... connected with session ID: {self.session_id}")
        self._trigger_callback('connection_established', {'session_id': self.session_id})
        
        return True
    
    async def disconnect(self) -> bool:
        """
        Simulate disconnection from GPU resources.
        
        Returns:
            bool: True if disconnection successful
        """
        logger.debug("running disconnect ... starting mock disconnection")
        
        if self._is_training:
            logger.warning("running disconnect ... terminating active training session")
            self._is_training = False
        
        # Reset mock state
        self._mock_memory.allocated_mb = 0.0
        self._mock_memory.cached_mb = 0.0
        self._mock_memory.utilization_percent = 0.0
        
        self.status = SessionStatus.DISCONNECTED
        self.session_id = None
        
        logger.info("running disconnect ... disconnected successfully")
        self._trigger_callback('disconnected', {})
        
        return True
    
    async def get_available_resources(self) -> List[GPUResourceInfo]:
        """
        Simulate available GPU resources.
        
        Returns:
            List[GPUResourceInfo]: Mock available resources
        """
        logger.debug("running get_available_resources ... fetching mock resources")
        
        if not await self.is_connected():
            logger.error("running get_available_resources ... not connected")
            return []
        
        # Simulate variety of available GPUs
        mock_resources = [
            GPUResourceInfo(
                resource_type=GPUResourceType.RTX_3080,
                memory_gb=10.0,
                compute_units=68,
                cost_per_hour=0.50,
                availability=True,
                location="mock-datacenter-1",
                estimated_performance=85.0
            ),
            GPUResourceInfo(
                resource_type=GPUResourceType.RTX_4090,
                memory_gb=24.0,
                compute_units=128,
                cost_per_hour=1.20,
                availability=random.choice([True, False]),  # Random availability
                location="mock-datacenter-2",
                estimated_performance=100.0
            ),
            GPUResourceInfo(
                resource_type=GPUResourceType.A100_40GB,
                memory_gb=40.0,
                compute_units=108,
                cost_per_hour=2.50,
                availability=True,
                location="mock-datacenter-3",
                estimated_performance=120.0
            )
        ]
        
        available_count = len([r for r in mock_resources if r.availability])
        logger.info(f"running get_available_resources ... found {available_count} available resources")
        
        return mock_resources
    
    async def estimate_cost(self, config: TrainingConfig) -> CostEstimate:
        """
        Simulate cost estimation for training operation.
        
        Args:
            config: Training configuration
            
        Returns:
            CostEstimate: Mock cost estimation
        """
        logger.debug("running estimate_cost ... calculating mock cost estimate")
        
        if not await self.validate_config(config):
            raise ValueError("Invalid training configuration")
        
        # Mock cost calculation based on epochs and model complexity
        base_time_per_epoch = 30.0  # seconds
        complexity_multiplier = len(str(config.model_config)) / 100.0  # Rough complexity estimate
        
        estimated_duration = (config.max_epochs * base_time_per_epoch * complexity_multiplier) / self.training_speed_multiplier
        cost_per_second = 0.50 / 3600  # $0.50 per hour
        total_cost = estimated_duration * cost_per_second
        
        cost_estimate = CostEstimate(
            estimated_duration_seconds=estimated_duration,
            cost_per_second=cost_per_second,
            total_estimated_cost=total_cost,
            currency="USD",
            breakdown={
                "compute_cost": total_cost * 0.8,
                "data_transfer": total_cost * 0.1,
                "storage": total_cost * 0.1
            }
        )
        
        logger.info(f"running estimate_cost ... estimated ${total_cost:.4f} for {estimated_duration:.1f}s")
        
        return cost_estimate
    
    async def train_model(self, config: TrainingConfig) -> TrainingResult:
        """
        Simulate model training with realistic progress updates.
        
        Args:
            config: Training configuration
            
        Returns:
            TrainingResult: Mock training results
        """
        logger.debug("running train_model ... starting mock training")
        
        if not await self.validate_config(config):
            return TrainingResult(
                success=False,
                error_message="Invalid training configuration"
            )
        
        if self._is_training:
            return TrainingResult(
                success=False,
                error_message="Another training session is already active"
            )
        
        # Check for simulated failure
        if self._should_simulate_failure():
            logger.error("running train_model ... simulated training failure")
            return TrainingResult(
                success=False,
                error_message="Simulated training failure"
            )
        
        self._is_training = True
        self.status = SessionStatus.BUSY
        start_time = time.time()
        
        try:
            return await self._simulate_training(config, start_time)
        
        except Exception as e:
            logger.error(f"running train_model ... training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )
        
        finally:
            self._is_training = False
            self.status = SessionStatus.CONNECTED
    
    async def get_memory_info(self) -> Optional[GPUMemoryInfo]:
        """
        Get mock GPU memory information.
        
        Returns:
            Optional[GPUMemoryInfo]: Mock memory usage info
        """
        if not await self.is_connected():
            return None
        
        # Simulate memory usage during training
        if self._is_training:
            # Gradually increase memory usage during training
            base_allocation = 2048.0  # 2GB base
            training_allocation = self._training_progress * 6144.0  # Up to 6GB additional
            self._mock_memory.allocated_mb = base_allocation + training_allocation
            self._mock_memory.cached_mb = min(1024.0, self._mock_memory.allocated_mb * 0.2)
        
        self._mock_memory.utilization_percent = (self._mock_memory.allocated_mb / self._mock_memory.total_mb) * 100
        
        logger.debug(f"running get_memory_info ... memory usage: {self._mock_memory.utilization_percent:.1f}%")
        
        return self._mock_memory
    
    async def _simulate_training(self, config: TrainingConfig, start_time: float) -> TrainingResult:
        """
        Simulate the training process with realistic progress.
        
        Args:
            config: Training configuration
            start_time: Training start timestamp
            
        Returns:
            TrainingResult: Training results
        """
        logger.info(f"running _simulate_training ... training for {config.max_epochs} epochs")
        
        # Initialize training history
        training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Simulate training epochs
        for epoch in range(config.max_epochs):
            if not self._is_training:  # Check if training was stopped
                break
            
            self._training_progress = (epoch + 1) / config.max_epochs
            
            # Simulate epoch duration
            epoch_duration = (2.0 + random.uniform(-0.5, 0.5)) / self.training_speed_multiplier
            await asyncio.sleep(epoch_duration)
            
            # Generate realistic training metrics
            loss = self._generate_mock_loss(epoch, config.max_epochs)
            accuracy = self._generate_mock_accuracy(epoch, config.max_epochs)
            val_loss = loss + random.uniform(0.0, 0.1)
            val_accuracy = accuracy - random.uniform(0.0, 0.05)
            
            # Store metrics
            training_history['loss'].append(loss)
            training_history['accuracy'].append(accuracy)
            training_history['val_loss'].append(val_loss)
            training_history['val_accuracy'].append(val_accuracy)
            
            # Trigger progress callback
            progress_data = {
                'epoch': epoch + 1,
                'total_epochs': config.max_epochs,
                'loss': loss,
                'accuracy': accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'progress': self._training_progress
            }
            self._trigger_callback('training_progress', progress_data)
            
            logger.debug(f"running _simulate_training ... epoch {epoch + 1}/{config.max_epochs} - "
                        f"loss: {loss:.4f}, accuracy: {accuracy:.4f}")
        
        # Calculate final metrics
        duration = time.time() - start_time
        final_loss = training_history['loss'][-1] if training_history['loss'] else 1.0
        final_accuracy = training_history['accuracy'][-1] if training_history['accuracy'] else 0.1
        
        # Simulate cost calculation
        cost_estimate = await self.estimate_cost(config)
        actual_cost = CostEstimate(
            estimated_duration_seconds=duration,
            cost_per_second=cost_estimate.cost_per_second,
            total_estimated_cost=duration * cost_estimate.cost_per_second,
            currency="USD"
        )
        
        logger.info(f"running _simulate_training ... training completed - "
                   f"final accuracy: {final_accuracy:.4f}, duration: {duration:.1f}s")
        
        # Trigger completion callback
        self._trigger_callback('training_complete', {
            'success': True,
            'final_accuracy': final_accuracy,
            'duration': duration
        })
        
        return TrainingResult(
            success=True,
            final_loss=final_loss,
            final_accuracy=final_accuracy,
            training_history=training_history,
            model_state={'mock_model': 'trained'},  # Mock model state
            duration_seconds=duration,
            memory_usage=await self.get_memory_info(),
            cost_info=actual_cost
        )
    
    def _generate_mock_loss(self, epoch: int, total_epochs: int) -> float:
        """Generate realistic decreasing loss values."""
        # Start high, decrease with some noise
        base_loss = 2.0 * (1 - epoch / total_epochs) ** 0.5
        noise = random.uniform(-0.1, 0.1)
        return max(0.01, base_loss + noise)
    
    def _generate_mock_accuracy(self, epoch: int, total_epochs: int) -> float:
        """Generate realistic increasing accuracy values."""
        # Start low, increase with some noise and plateau
        progress = epoch / total_epochs
        base_accuracy = 0.1 + 0.85 * (1 - (1 - progress) ** 2)  # Curve that plateaus
        noise = random.uniform(-0.02, 0.02)
        return min(0.99, max(0.05, base_accuracy + noise))
    
    def _should_simulate_failure(self) -> bool:
        """Determine if a simulated failure should occur."""
        return self.simulate_failures and random.random() < self.failure_rate
    
    # Additional mock-specific methods for testing
    def set_failure_simulation(self, enabled: bool, rate: float = 0.1) -> None:
        """
        Configure failure simulation for testing.
        
        Args:
            enabled: Whether to enable failure simulation
            rate: Probability of failures (0.0 to 1.0)
        """
        self.simulate_failures = enabled
        self.failure_rate = max(0.0, min(1.0, rate))
        logger.debug(f"running set_failure_simulation ... failure simulation: {enabled}, rate: {rate}")
    
    def set_training_speed(self, multiplier: float) -> None:
        """
        Set training speed multiplier for faster/slower simulation.
        
        Args:
            multiplier: Speed multiplier (1.0 = normal, 0.1 = 10x faster, 2.0 = 2x slower)
        """
        self.training_speed_multiplier = max(0.01, multiplier)
        logger.debug(f"running set_training_speed ... speed multiplier set to {multiplier}")
    
    def force_stop_training(self) -> None:
        """Force stop current training session (for testing)."""
        if self._is_training:
            logger.warning("running force_stop_training ... forcibly stopping training")
            self._is_training = False
            self._trigger_callback('training_stopped', {'reason': 'forced_stop'})