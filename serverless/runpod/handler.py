"""
Generic PyTorch Handler for RunPod Serverless
Supports basic model training, evaluation, and inference operations
"""

import runpod
import torch
import torch.nn as nn
import torch.optim as optim
import base64
import pickle
import json
from pathlib import Path

# Simple logging for serverless environment
import logging
import sys

def setup_logger():
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

# Global device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model_from_config(model_config):
    """
    Create a PyTorch model from configuration dictionary
    
    Args:
        model_config (dict): Model configuration with architecture details
        
    Returns:
        torch.nn.Module: Configured PyTorch model
    """
    logger.debug("running create_model_from_config ... creating model from config")
    
    model_type = model_config.get("type", "sequential")
    
    if model_type == "sequential":
        layers = []
        layer_configs = model_config.get("layers", [])
        
        for layer_config in layer_configs:
            layer_type = layer_config.get("type")
            
            if layer_type == "linear":
                layers.append(nn.Linear(
                    layer_config["in_features"],
                    layer_config["out_features"]
                ))
            elif layer_type == "relu":
                layers.append(nn.ReLU())
            elif layer_type == "dropout":
                layers.append(nn.Dropout(layer_config.get("p", 0.5)))
            elif layer_type == "conv2d":
                layers.append(nn.Conv2d(
                    layer_config["in_channels"],
                    layer_config["out_channels"],
                    layer_config["kernel_size"],
                    stride=layer_config.get("stride", 1),
                    padding=layer_config.get("padding", 0)
                ))
            elif layer_type == "maxpool2d":
                layers.append(nn.MaxPool2d(
                    layer_config["kernel_size"],
                    stride=layer_config.get("stride", None)
                ))
            elif layer_type == "flatten":
                layers.append(nn.Flatten())
            
        model = nn.Sequential(*layers)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model.to(device)

def serialize_model_state(model):
    """
    Serialize model state dict to base64 string
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        str: Base64 encoded model state
    """
    logger.debug("running serialize_model_state ... serializing model state")
    
    model_bytes = pickle.dumps(model.state_dict())
    return base64.b64encode(model_bytes).decode('utf-8')

def deserialize_model_state(model, state_str):
    """
    Deserialize base64 model state and load into model
    
    Args:
        model (torch.nn.Module): PyTorch model
        state_str (str): Base64 encoded model state
        
    Returns:
        torch.nn.Module: Model with loaded state
    """
    logger.debug("running deserialize_model_state ... deserializing model state")
    
    if state_str:
        state_bytes = base64.b64decode(state_str.encode('utf-8'))
        state_dict = pickle.loads(state_bytes)
        model.load_state_dict(state_dict)
    
    return model

def prepare_data_tensors(data_config):
    """
    Convert data configuration to PyTorch tensors
    
    Args:
        data_config (dict): Data configuration with features and labels
        
    Returns:
        tuple: (features_tensor, labels_tensor)
    """
    logger.debug("running prepare_data_tensors ... preparing data tensors")
    
    features = torch.tensor(data_config["features"], dtype=torch.float32).to(device)
    labels = None
    
    if "labels" in data_config:
        labels = torch.tensor(data_config["labels"], dtype=torch.long).to(device)
    
    return features, labels

def train_single_epoch(model, data_config, hyperparams):
    """
    Train model for a single epoch
    
    Args:
        model (torch.nn.Module): PyTorch model
        data_config (dict): Training data configuration
        hyperparams (dict): Training hyperparameters
        
    Returns:
        dict: Training metrics (loss, accuracy, etc.)
    """
    logger.debug("running train_single_epoch ... training for one epoch")
    
    model.train()
    
    # Prepare data
    features, labels = prepare_data_tensors(data_config)
    
    # Setup optimizer and loss function
    optimizer_name = hyperparams.get("optimizer", "adam")
    learning_rate = hyperparams.get("learning_rate", 0.001)
    
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    loss_function_name = hyperparams.get("loss_function", "cross_entropy")
    if loss_function_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_function_name == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_function_name}")
    
    # Training step
    optimizer.zero_grad()
    outputs = model(features)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # Calculate metrics
    with torch.no_grad():
        predicted = torch.argmax(outputs, dim=1)
        accuracy = (predicted == labels).float().mean().item()
    
    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy,
        "samples_processed": len(features)
    }
    
    logger.debug(f"running train_single_epoch ... epoch complete, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}")
    
    return metrics

def evaluate_model(model, data_config):
    """
    Evaluate model on provided data
    
    Args:
        model (torch.nn.Module): PyTorch model
        data_config (dict): Evaluation data configuration
        
    Returns:
        dict: Evaluation metrics
    """
    logger.debug("running evaluate_model ... evaluating model")
    
    model.eval()
    features, labels = prepare_data_tensors(data_config)
    
    with torch.no_grad():
        outputs = model(features)
        
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels).item()
            predicted = torch.argmax(outputs, dim=1)
            accuracy = (predicted == labels).float().mean().item()
            
            metrics = {
                "loss": loss,
                "accuracy": accuracy,
                "samples_processed": len(features)
            }
        else:
            # Inference only
            predicted = torch.argmax(outputs, dim=1)
            metrics = {
                "predictions": predicted.cpu().tolist(),
                "samples_processed": len(features)
            }
    
    logger.debug(f"running evaluate_model ... evaluation complete")
    
    return metrics

def handler(job):
    """
    Main handler function for RunPod serverless requests
    
    Args:
        job (dict): Job request containing input parameters
        
    Returns:
        dict: Response with results or error information
    """
    logger.debug("running handler ... processing job request")
    
    try:
        job_input = job["input"]
        operation = job_input.get("operation", "train_epoch")
        
        # Create or load model
        model_config = job_input["model_config"]
        model = create_model_from_config(model_config)
        
        # Load existing model state if provided
        if "model_state" in job_input:
            model = deserialize_model_state(model, job_input["model_state"])
        
        if operation == "train_epoch":
            # Train for one epoch
            data_config = job_input["training_data"]
            hyperparams = job_input["hyperparameters"]
            
            metrics = train_single_epoch(model, data_config, hyperparams)
            model_state = serialize_model_state(model)
            
            return {
                "operation": "train_epoch",
                "metrics": metrics,
                "model_state": model_state,
                "status": "success"
            }
            
        elif operation == "evaluate":
            # Evaluate model
            data_config = job_input["evaluation_data"]
            metrics = evaluate_model(model, data_config)
            
            return {
                "operation": "evaluate",
                "metrics": metrics,
                "status": "success"
            }
            
        elif operation == "predict":
            # Make predictions
            data_config = job_input["input_data"]
            results = evaluate_model(model, data_config)
            
            return {
                "operation": "predict",
                "predictions": results.get("predictions", []),
                "status": "success"
            }
            
        else:
            return {
                "error": f"Unsupported operation: {operation}",
                "status": "error"
            }
            
    except Exception as e:
        logger.error(f"running handler ... error occurred: {str(e)}")
        return {
            "error": str(e),
            "status": "error"
        }

# Start the serverless function
if __name__ == "__main__":
    logger.debug("running main ... starting serverless worker")
    runpod.serverless.start({"handler": handler})