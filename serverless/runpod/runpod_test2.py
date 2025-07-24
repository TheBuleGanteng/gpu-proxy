#!/usr/bin/env python3
"""
Fixed RunPod serverless test script
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pathlib import Path
import requests
import json
import time
from typing import Optional, Dict, Any

# Add the project root to Python path
current_file = Path(__file__)
project_root = current_file.parent.parent.parent  # Go up 3 levels to project root
sys.path.insert(0, str(project_root))

from src.utils.logger import logger

def test_runpod_endpoint(endpoint_id: str, api_key: Optional[str] = None) -> bool:
    """Test the RunPod serverless endpoint with a simple request"""
    
    logger.debug("running test_runpod_endpoint ... testing serverless endpoint")
    
    # Simple test payload for serverless endpoint
    test_payload = {
        "input": {
            "operation": "train_epoch",
            "model_config": {
                "type": "sequential",
                "layers": [
                    {"type": "linear", "in_features": 4, "out_features": 8},
                    {"type": "relu"},
                    {"type": "linear", "in_features": 8, "out_features": 2}
                ]
            },
            "training_data": {
                "features": [
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 3.0, 4.0, 5.0],
                    [3.0, 4.0, 5.0, 6.0],
                    [4.0, 5.0, 6.0, 7.0]
                ],
                "labels": [0, 1, 0, 1]
            },
            "hyperparameters": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "loss_function": "cross_entropy"
            }
        }
    }
    
    print("🧪 Testing RunPod serverless endpoint...")
    print(f"Endpoint ID: {endpoint_id}")
    if api_key:
        print(f"API Key: {api_key[:10]}...")
    print("=" * 60)
    
    try:
        # Correct URL format for RunPod serverless endpoints
        url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        
        # Set up headers with API key
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        else:
            logger.debug("running test_runpod_endpoint ... no API key provided")
            print("⚠️  Warning: No API key provided")
        
        print("📤 Sending test request...")
        start_time = time.time()
        
        response = requests.post(
            url,
            json=test_payload,
            headers=headers,
            timeout=120
        )
        
        request_time = time.time() - start_time
        
        print(f"⏱️  Request completed in {request_time:.2f} seconds")
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("✅ Response received successfully!")
            logger.debug(f"running test_runpod_endpoint ... received response: {result}")
            
            # Handle RunPod serverless response format
            if "id" in result:
                job_id = result["id"]
                status = result.get("status", "UNKNOWN")
                
                print(f"🎯 Job submitted successfully!")
                print(f"  • Job ID: {job_id}")
                print(f"  • Status: {status}")
                
                if status == "COMPLETED":
                    output = result.get("output", {})
                    print(f"  • Output: {output}")
                    return True
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    print("  • Job is processing... you may need to check status separately")
                    return True
                else:
                    print(f"  • Unexpected status: {status}")
                    return False
            else:
                print("❌ Unexpected response structure")
                print(f"Response: {json.dumps(result, indent=2)}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            logger.debug(f"running test_runpod_endpoint ... HTTP error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (this might be a cold start - try again)")
        logger.debug("running test_runpod_endpoint ... request timeout")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
        logger.debug(f"running test_runpod_endpoint ... request failed: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        logger.debug(f"running test_runpod_endpoint ... unexpected error: {str(e)}")
        return False

def main() -> bool:
    """Main test function"""
    
    print("🚀 RunPod Serverless Deployment Test")
    print("=" * 60)
    
    # Get configuration from environment
    runpod_api_key = os.getenv('RUNPOD_API_KEY')
    runpod_endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
    
    logger.debug(f"running main ... endpoint_id: {runpod_endpoint_id}")
    logger.debug(f"running main ... api_key available: {bool(runpod_api_key)}")
    
    if not runpod_endpoint_id:
        print("❌ Missing RUNPOD_ENDPOINT_ID environment variable!")
        print("Please set RUNPOD_ENDPOINT_ID in your .env file")
        return False
    
    if not runpod_api_key:
        print("⚠️  Warning: RUNPOD_API_KEY not found in environment")
        print("This may cause authentication issues")
    
    success = test_runpod_endpoint(runpod_endpoint_id, runpod_api_key)
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Your serverless endpoint is working correctly!")
        print("2. You can now integrate this with your hyperparameter optimization")
        print("3. Consider implementing job status polling for longer-running tasks")
    else:
        print("\n🔧 Troubleshooting:")
        print("1. Check RunPod console for endpoint status")
        print("2. Verify the endpoint is active and has workers available")
        print("3. Check the logs in RunPod console for errors")
        print("4. Ensure your API key has correct permissions")
        print("5. Try again in a few minutes (cold starts can be slow)")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)