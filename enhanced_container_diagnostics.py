#!/usr/bin/env python3
"""
Enhanced Container Diagnostics - Check RunPod library installation specifically
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from unittest import result

# Add project root to path for imports
current_file = Path(__file__)
project_root = current_file.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import logger

try:
    import aiohttp
except ImportError:
    logger.error("aiohttp not installed. Install with: pip install aiohttp")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed. Ensure environment variables are set manually.")


class EnhancedRunPodDiagnostics:
    """Enhanced diagnostics focusing on RunPod library installation."""
    
    def __init__(self):
        self.api_key = os.getenv('RUNPOD_API_KEY')
        self.endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
        
        if not self.api_key or not self.endpoint_id:
            logger.error("Missing RunPod credentials. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in .env")
            raise ValueError("Missing RunPod credentials")
        
        logger.info(f"RunPod Endpoint: {self.endpoint_id}")
    
    async def run_enhanced_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics with special focus on RunPod library."""
        logger.info("🔍 Running Enhanced Container Diagnostics")
        logger.info("Focus: RunPod library installation and container state")
        
        # Create a comprehensive diagnostic payload
        # Note: Since your handler has a health check mode, we'll use that
        diagnostic_payload = {
            "input": {
                "test": "health_check"  # This should work even without RunPod if handler runs
            }
        }
        
        result = await self._submit_and_monitor(diagnostic_payload, "enhanced_diagnostics", timeout=90)
        
        if result:
            await self._analyze_diagnostic_results(result)
        
        # Instead of returning potentially None result:
        if result is None:
            return {
                "error": "Operation failed - no result available",
                "success": False,
                "timestamp": time.time()
            }
        
        return result        
    
    async def test_library_installation(self) -> Dict[str, Any]:
        """Test a minimal job to see what libraries are actually available."""
        logger.info("🧪 Testing Library Installation")
        
        # Send minimal training config to trigger library imports
        library_test_payload = {
            "input": {
                "training_config": "",  # Empty config to trigger deserialization error (expected)
                "framework": "test",
                "max_epochs": 1,
                "sync_mode": True
            }
        }
        
        result = await self._submit_and_monitor(library_test_payload, "library_test", timeout=60)
        
        if result is None:
            return {
                "error": "No result returned from monitoring",
                "status": "failed",
                "timestamp": time.time()
            }
        
        return result
    
    async def _submit_and_monitor(self, payload: Dict[str, Any], test_name: str, timeout: int = 120) -> Optional[Dict[str, Any]]:
        """Submit job and monitor with enhanced error analysis."""
        
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"📤 Submitting {test_name} to RunPod...")
        logger.debug(f"running enhanced_container_diagnostics ... Payload: {json.dumps(payload, indent=2)}")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Submit the job
                async with session.post(url, json=payload, headers=headers) as response:
                    submit_result = await response.json()
                    
                    logger.debug(f"running enhanced_container_diagnostics ... Submit response status: {response.status}")
                    logger.debug(f"running enhanced_container_diagnostics ... Submit response: {json.dumps(submit_result, indent=2)}")
                    
                    if response.status != 200:
                        logger.error(f"❌ Submit failed with status {response.status}")
                        logger.error(f"Response: {submit_result}")
                        return None
                    
                    if 'id' not in submit_result:
                        logger.error(f"❌ No job ID returned: {submit_result}")
                        return None
                    
                    job_id = submit_result['id']
                    logger.info(f"✅ Job submitted successfully. ID: {job_id}")
                    
                    # Monitor job with detailed status tracking
                    return await self._monitor_job_with_analysis(session, headers, job_id, timeout)
                    
            except Exception as e:
                logger.error(f"❌ Failed to submit {test_name}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None
    
    async def _monitor_job_with_analysis(self, session: aiohttp.ClientSession, headers: Dict[str, str], 
                                       job_id: str, timeout: int) -> Optional[Dict[str, Any]]:
        """Monitor job with detailed analysis of different failure modes."""
        
        status_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/status/{job_id}"
        start_time = time.time()
        
        logger.info(f"📊 Monitoring job {job_id} (timeout: {timeout}s)...")
        
        last_status = None
        status_changes = []
        
        while time.time() - start_time < timeout:
            try:
                async with session.get(status_url, headers=headers) as response:
                    status_result = await response.json()
                    
                    current_status = status_result.get('status', 'UNKNOWN')
                    elapsed = time.time() - start_time
                    
                    # Track status changes
                    if current_status != last_status:
                        status_changes.append({
                            'status': current_status,
                            'timestamp': elapsed,
                            'details': status_result
                        })
                        last_status = current_status
                        
                        logger.info(f"📋 Status: {current_status} (elapsed: {elapsed:.1f}s)")
                    
                    if current_status == 'COMPLETED':
                        output = status_result.get('output', {})
                        logger.info("✅ Job completed successfully!")
                        
                        return {
                            "success": True,
                            "status": current_status,
                            "output": output,
                            "elapsed_seconds": elapsed,
                            "status_changes": status_changes
                        }
                    
                    elif current_status == 'FAILED':
                        error = status_result.get('error', 'Unknown error')
                        logger.error(f"❌ Job failed: {error}")
                        
                        return {
                            "success": False,
                            "status": current_status,
                            "error": error,
                            "elapsed_seconds": elapsed,
                            "status_changes": status_changes,
                            "full_response": status_result
                        }
                    
                    elif current_status == 'IN_QUEUE':
                        if elapsed > 30:  # Long queue time indicates worker issues
                            logger.warning(f"⚠️ Job stuck in queue for {elapsed:.1f}s - possible worker health issues")
                    
                    elif current_status == 'IN_PROGRESS':
                        # Job is running - this is good
                        pass
                    
                    else:
                        logger.debug(f"Status: {current_status}")
                    
                    await asyncio.sleep(2)
                        
            except Exception as e:
                logger.warning(f"⚠️ Error checking job status: {e}")
                await asyncio.sleep(5)  # Longer delay on error
                continue
        
        # Timeout analysis
        logger.error(f"⏰ Job timed out after {timeout}s")
        logger.error(f"Final status: {last_status}")
        logger.error(f"Status progression: {[change['status'] for change in status_changes]}")
        
        return {
            "success": False,
            "error": f"Timeout after {timeout}s",
            "status": "TIMEOUT",
            "final_status": last_status,
            "status_changes": status_changes
        }
    
    async def _analyze_diagnostic_results(self, result: Dict[str, Any]) -> None:
        """Analyze diagnostic results with specific focus on RunPod library issues."""
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 ENHANCED DIAGNOSTIC ANALYSIS")
        logger.info("=" * 60)
        
        success = result.get('success', False)
        
        if not success:
            logger.error("❌ DIAGNOSTIC FAILED")
            
            # Analyze failure mode
            status_changes = result.get('status_changes', [])
            
            if len(status_changes) == 0:
                logger.error("🚫 No status changes detected - endpoint may be inactive")
            elif any(change['status'] == 'IN_QUEUE' for change in status_changes):
                if all(change['status'] == 'IN_QUEUE' for change in status_changes):
                    logger.error("⏳ Job stuck in queue - no healthy workers available")
                    logger.error("   This suggests containers are crashing during startup")
                else:
                    logger.info("✅ Queue processing working")
            
            return
        
        # Analyze successful output
        output = result.get('output', {})
        
        logger.info("✅ CONTAINER IS RESPONDING")
        
        # Check specific capabilities
        runpod_available = output.get('runpod_available', False)
        tensorflow_available = output.get('tensorflow_available', False)
        gpu_available = output.get('gpu_available', False)
        
        logger.info(f"📋 CAPABILITY CHECK:")
        logger.info(f"   RunPod Library: {'✅ Available' if runpod_available else '❌ Missing'}")
        logger.info(f"   TensorFlow: {'✅ Available' if tensorflow_available else '❌ Missing'}")
        logger.info(f"   GPU Access: {'✅ Available' if gpu_available else '❌ Missing'}")
        
        # Specific recommendations based on findings
        if not runpod_available:
            logger.error("\n🎯 CRITICAL ISSUE IDENTIFIED:")
            logger.error("   RunPod library is missing from your container!")
            logger.error("   This explains why containers can't process serverless requests.")
            logger.error("\n🔧 IMMEDIATE ACTIONS NEEDED:")
            logger.error("   1. Rebuild container with 'runpod>=1.0.0' in requirements.txt")
            logger.error("   2. Use the enhanced Dockerfile.runpod-fix to update existing container")
            logger.error("   3. Verify requirements.txt installation during Docker build")
        else:
            logger.info("\n✅ RunPod library is properly installed")
            if not gpu_available:
                logger.warning("⚠️ GPU access issue detected")
            else:
                logger.info("🎉 All systems appear functional!")


async def main():
    """Run enhanced diagnostics with focus on RunPod library issues."""
    logger.info("🚀 Starting Enhanced Container Diagnostics")
    logger.info("Focusing on RunPod library installation and serverless capability")
    logger.info("=" * 60)
    
    try:
        diagnostics = EnhancedRunPodDiagnostics()
        
        # Run comprehensive diagnostics
        logger.info("\n🔍 Phase 1: Health Check Diagnostics")
        health_result = await diagnostics.run_enhanced_diagnostics()
        
        # If health check works, test library imports
        if health_result and health_result.get('success'):
            logger.info("\n🧪 Phase 2: Library Installation Test")
            library_result = await diagnostics.test_library_installation()
        else:
            logger.warning("⚠️ Skipping library test due to health check failure")
        
        # Final recommendations
        logger.info("\n" + "=" * 60)
        logger.info("🎯 FINAL RECOMMENDATIONS")
        logger.info("=" * 60)
        
        if health_result and health_result.get('success'):
            output = health_result.get('output', {})
            runpod_available = output.get('runpod_available', False)
            
            if runpod_available:
                logger.info("✅ Container appears to be working correctly")
                logger.info("🔄 Try running your actual optimization workload")
            else:
                logger.error("❌ RunPod library missing - rebuild container needed")
                logger.info("📋 Next step: Use Dockerfile.runpod-fix to update container")
        else:
            logger.error("❌ Container has fundamental issues")
            logger.info("📋 Check RunPod endpoint configuration and container health")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Diagnostics interrupted by user")
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())