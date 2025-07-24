"""
RunPod GPU Client for Local Orchestration
Provides simple interface for GPU-accelerated ML operations via RunPod serverless
"""

import requests
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add the project root to Python path for local usage
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import logger

class RunPodGPUClient:
    """
    Client for communicating with RunPod serverless GPU endpoints
    """
    
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None, timeout: int = 300):
        """
        Initialize RunPod GPU client
        
        Args:
            endpoint_url: RunPod serverless endpoint URL
            api_key: RunPod API key (if required)
            timeout: Request timeout in seconds
        """
        logger.debug("running __init__ ... initializing RunPod GPU client")
        
        self.endpoint_url = endpoint_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            })
    
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make request to RunPod endpoint with error handling
        
        Args:
            payload: Request payload
            
        Returns:
            Response data
            
        Raises:
            Exception: If request fails or times out
        """
        logger.debug(f"running _make_request ... sending request to {self.endpoint_url}")
        
        try:
            # For RunPod /run endpoint (synchronous)
            url = f"{self.endpoint_url}/run"
            
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('status') == 'error':
                raise Exception(f"RunPod error: {result.get('error', 'Unknown error')}")
            
            logger.debug("running _make_request ... request completed successfully")
            return result
            
        except requests.exceptions.Timeout:
            logger.error("running _make_request ... request timed out")
            raise Exception(f"Request timed out after {self.timeout} seconds")
        except requests.exceptions.RequestException as e:
            logger.error(f"running _make_request ... request failed: {str(e)}")
            raise Exception(f"Request failed: {str(e)}")
    
    def train_epoch(self, 
                   model_config: Dict[str, Any], 
                   training_data: Dict[str, Any], 
                   hyperparameters: Dict[str, Any],
                   model_state: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
        """
        Train model for one epoch on GPU
        
        Args:
            model_config: Model architecture configuration
            training_data: Training data with features and labels
            hyperparameters: Training hyperparameters
            model_state: Existing model state (base64 encoded)
            
        Returns:
            Tuple of (metrics, updated_model_state)
        """
        logger.debug("running train_epoch ... training single epoch on GPU")
        
        payload = {
            "input": {
                "operation": "train_epoch",
                "model_config": model_config,
                "training_data": training_data,
                "hyperparameters": hyperparameters
            }
        }
        
        if model_state:
            payload["input"]["model_state"] = model_state
        
        result = self._make_request(payload)
        output = result.get("output", {})
        
        return output.get("metrics", {}), output.get("model_state", "")
    
    def evaluate_model(self, 
                      model_config: Dict[str, Any], 
                      evaluation_data: Dict[str, Any],
                      model_state: str) -> Dict[str, Any]:
        """
        Evaluate model on GPU
        
        Args:
            model_config: Model architecture configuration
            evaluation_data: Evaluation data with features and labels
            model_state: Model state (base64 encoded)
            
        Returns:
            Evaluation metrics
        """
        logger.debug("running evaluate_model ... evaluating model on GPU")
        
        payload = {
            "input": {
                "operation": "evaluate",
                "model_config": model_config,
                "evaluation_data": evaluation_data,
                "model_state": model_state
            }
        }
        
        result = self._make_request(payload)
        return result.get("output", {}).get("metrics", {})
    
    def predict(self, 
               model_config: Dict[str, Any], 
               input_data: Dict[str, Any],
               model_state: str) -> List[Any]:
        """
        Make predictions using model on GPU
        
        Args:
            model_config: Model architecture configuration
            input_data: Input data for prediction
            model_state: Model state (base64 encoded)
            
        Returns:
            List of predictions
        """
        logger.debug("running predict ... making predictions on GPU")
        
        payload = {
            "input": {
                "operation": "predict",
                "model_config": model_config,
                "input_data": input_data,
                "model_state": model_state
            }
        }
        
        result = self._make_request(payload)
        return result.get("output", {}).get("predictions", [])
    
    def health_check(self) -> bool:
        """
        Check if RunPod endpoint is healthy
        
        Returns:
            True if endpoint is responding, False otherwise
        """
        logger.debug("running health_check ... checking endpoint health")
        
        try:
            # Simple test request
            test_payload = {
                "input": {
                    "operation": "train_epoch",
                    "model_config": {
                        "type": "sequential",
                        "layers": [
                            {"type": "linear", "in_features": 2, "out_features": 1}
                        ]
                    },
                    "training_data": {
                        "features": [[1.0, 2.0], [3.0, 4.0]],
                        "labels": [0, 1]
                    },
                    "hyperparameters": {
                        "optimizer": "adam",
                        "learning_rate": 0.001,
                        "loss_function": "cross_entropy"
                    }
                }
            }
            
            result = self._make_request(test_payload)
            return result.get("output", {}).get("status") == "success"
            
        except Exception as e:
            logger.error(f"running health_check ... health check failed: {str(e)}")
            return False


class GPUTrainingOrchestrator:
    """
    Higher-level orchestrator for multi-epoch training workflows
    """
    
    def __init__(self, runpod_client: RunPodGPUClient):
        """
        Initialize training orchestrator
        
        Args:
            runpod_client: RunPod GPU client instance
        """
        logger.debug("running __init__ ... initializing GPU training orchestrator")
        self.client = runpod_client
        self.training_history = []
    
    def train_model(self, 
                   model_config: Dict[str, Any],
                   training_data: Dict[str, Any],
                   validation_data: Optional[Dict[str, Any]],
                   hyperparameters: Dict[str, Any],
                   epochs: int) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Train model for multiple epochs with validation
        
        Args:
            model_config: Model architecture configuration
            training_data: Training data
            validation_data: Validation data (optional)
            hyperparameters: Training hyperparameters
            epochs: Number of training epochs
            
        Returns:
            Tuple of (final_model_state, training_history)
        """
        logger.debug(f"running train_model ... training model for {epochs} epochs")
        
        model_state = None
        history = []
        
        for epoch in range(epochs):
            logger.debug(f"running train_model ... starting epoch {epoch + 1}/{epochs}")
            
            # Train one epoch
            train_metrics, model_state = self.client.train_epoch(
                model_config=model_config,
                training_data=training_data,
                hyperparameters=hyperparameters,
                model_state=model_state
            )
            
            epoch_data = {
                "epoch": epoch + 1,
                "train_metrics": train_metrics
            }
            
            # Validate if validation data provided
            if validation_data:
                val_metrics = self.client.evaluate_model(
                    model_config=model_config,
                    evaluation_data=validation_data,
                    model_state=model_state
                )
                epoch_data["val_metrics"] = val_metrics
                
                logger.debug(f"running train_model ... epoch {epoch + 1} complete - "
                           f"train_loss: {train_metrics.get('loss', 'N/A'):.4f}, "
                           f"val_loss: {val_metrics.get('loss', 'N/A'):.4f}")
            else:
                logger.debug(f"running train_model ... epoch {epoch + 1} complete - "
                           f"train_loss: {train_metrics.get('loss', 'N/A'):.4f}")
            
            history.append(epoch_data)
        
        self.training_history = history
        logger.debug("running train_model ... training complete")
        
        return model_state, history
    
    def hyperparameter_trial(self,
                           model_config: Dict[str, Any],
                           training_data: Dict[str, Any],
                           validation_data: Dict[str, Any],
                           hyperparameters: Dict[str, Any],
                           epochs: int) -> Dict[str, Any]:
        """
        Run a single hyperparameter optimization trial
        
        Args:
            model_config: Model architecture configuration
            training_data: Training data
            validation_data: Validation data
            hyperparameters: Hyperparameters to test
            epochs: Number of training epochs
            
        Returns:
            Trial results with final metrics
        """
        logger.debug("running hyperparameter_trial ... running optimization trial")
        
        start_time = time.time()
        
        model_state, history = self.train_model(
            model_config=model_config,
            training_data=training_data,
            validation_data=validation_data,
            hyperparameters=hyperparameters,
            epochs=epochs
        )
        
        training_time = time.time() - start_time
        
        # Extract final metrics
        final_epoch = history[-1] if history else {}
        
        trial_result = {
            "hyperparameters": hyperparameters,
            "final_train_metrics": final_epoch.get("train_metrics", {}),
            "final_val_metrics": final_epoch.get("val_metrics", {}),
            "training_history": history,
            "training_time": training_time,
            "model_state": model_state
        }
        
        logger.debug(f"running hyperparameter_trial ... trial complete in {training_time:.2f}s")
        
        return trial_result