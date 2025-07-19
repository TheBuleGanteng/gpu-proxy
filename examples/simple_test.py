"""
Simple test to validate RunPod API connectivity using the correct endpoint format.
"""

import asyncio
import aiohttp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_runpod_api():
    """Test RunPod API using the correct endpoint format."""
    
    api_key = os.getenv('RUNPOD_API_KEY')
    endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
    
    if not api_key:
        print("❌ RUNPOD_API_KEY not found in .env file")
        return
    
    if not endpoint_id:
        print("❌ RUNPOD_ENDPOINT_ID not found in .env file")
        return
    
    print(f"✅ API Key: {api_key[:10]}..." if api_key else "❌ No API key")
    print(f"✅ Endpoint ID: {endpoint_id}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Use the correct RunPod API structure
    base_url = "https://api.runpod.ai/v2"
    job_url = f"{base_url}/{endpoint_id}/run"
    
    print(f"\n🔍 Testing RunPod endpoint: {job_url}")
    
    async with aiohttp.ClientSession(headers=headers) as session:
        
        # Test with the exact format from your curl example
        test_payload = {
            "input": {
                "prompt": "Hello from GPU Proxy test"
            }
        }
        
        try:
            print(f"📤 Submitting test job...")
            print(f"URL: {job_url}")
            print(f"Payload: {test_payload}")
            
            async with session.post(job_url, json=test_payload) as response:
                status = response.status
                text = await response.text()
                
                print(f"\n📥 Response:")
                print(f"Status: {status}")
                print(f"Body: {text}")
                
                if status == 200:
                    print("✅ SUCCESS! RunPod endpoint is working correctly!")
                    try:
                        data = await response.json() if text else {}
                        job_id = data.get('id')
                        if job_id:
                            print(f"🎯 Job ID: {job_id}")
                            
                            # Test job status endpoint
                            print(f"\n🔍 Testing job status endpoint...")
                            status_url = f"{base_url}/{endpoint_id}/status/{job_id}"
                            
                            async with session.get(status_url) as status_response:
                                status_text = await status_response.text()
                                print(f"Status endpoint: {status_url}")
                                print(f"Status response: {status_response.status}")
                                print(f"Status body: {status_text}")
                        
                    except Exception as e:
                        print(f"⚠️ Could not parse response as JSON: {e}")
                        
                elif status == 401:
                    print("❌ AUTHENTICATION FAILED")
                    print("🔧 Check your API key in .env file")
                    
                elif status == 404:
                    print("❌ ENDPOINT NOT FOUND")
                    print("🔧 Check your endpoint ID in .env file")
                    print(f"🔧 Current endpoint ID: {endpoint_id}")
                    
                elif status == 400:
                    print("❌ BAD REQUEST")
                    print("🔧 The request format might be incorrect")
                    print("🔧 This could mean your serverless function expects different input")
                    
                else:
                    print(f"⚠️ UNEXPECTED STATUS: {status}")
                    print("🔧 This might indicate an issue with your serverless function")
                    
        except Exception as e:
            print(f"❌ REQUEST FAILED: {e}")
            print("🔧 Check your internet connection and API endpoint")
    
    print("\n" + "="*60)
    print("📋 NEXT STEPS:")
    print("1. ✅ If status=200: Your RunPod integration is working!")
    print("2. ❌ If status=401: Update your API key in .env")
    print("3. ❌ If status=404: Check your endpoint ID in .env")
    print("4. ❌ If status=400: You need to deploy a serverless function")
    print("5. 🔗 RunPod Docs: https://docs.runpod.io/serverless/endpoints")

if __name__ == "__main__":
    asyncio.run(test_runpod_api())