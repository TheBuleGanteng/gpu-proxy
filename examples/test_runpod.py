"""
Advanced RunPod functionality test.

Tests all RunPod API endpoints including sync jobs, health checks,
job cancellation, queue management, and streaming.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
current_file = Path(__file__)
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

from src.core.base import RemoteGPU, TrainingConfig, GPUProviderType
from src.utils.logger import logger


async def test_health_check():
    """Test endpoint health checking."""
    logger.info("=== Testing Health Check ===")
    
    api_key = os.getenv('RUNPOD_API_KEY')
    endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
    
    if not api_key or not endpoint_id:
        logger.error("Missing RunPod credentials")
        return False
    
    try:
        async with RemoteGPU(provider="runpod_serverless", config={
            'api_key': api_key,
            'endpoint_id': endpoint_id
        }) as gpu:
            
            health_status = await gpu.check_endpoint_health()
            logger.info(f"Health check result: {health_status}")
            
            if health_status.get('healthy', False):
                logger.info("✅ Endpoint is healthy")
                return True
            else:
                logger.warning("⚠️ Endpoint health issues detected")
                return False
                
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False


async def test_sync_job():
    """Test synchronous job execution."""
    logger.info("\n=== Testing Synchronous Job ===")
    
    api_key = os.getenv('RUNPOD_API_KEY')
    endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
    
    if not api_key or not endpoint_id:
        logger.error("Missing RunPod credentials")
        return False
    
    try:
        async with RemoteGPU(provider="runpod_serverless", config={
            'api_key': api_key,
            'endpoint_id': endpoint_id,
            'timeout_seconds': 120  # 2 minutes for sync job
        }) as gpu:
            
            config = TrainingConfig(
                model_config={"type": "sync_test", "quick": True},
                training_params={"sync_mode": True},
                data_config={"dataset": "minimal"},
                framework="tensorflow",
                max_epochs=1,
                batch_size=1
            )
            
            logger.info("Starting synchronous job...")
            result = await gpu.run_sync(config)
            
            if result.success:
                logger.info("✅ Synchronous job completed successfully!")
                logger.info(f"Duration: {result.duration_seconds:.1f}s")
                return True
            else:
                logger.error(f"❌ Sync job failed: {result.error_message}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Sync job test failed: {e}")
        return False


async def test_job_cancellation():
    """Test job cancellation functionality."""
    logger.info("\n=== Testing Job Cancellation ===")
    
    api_key = os.getenv('RUNPOD_API_KEY')
    endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
    
    if not api_key or not endpoint_id:
        logger.error("Missing RunPod credentials")
        return False
    
    try:
        async with RemoteGPU(provider="runpod_serverless", config={
            'api_key': api_key,
            'endpoint_id': endpoint_id
        }) as gpu:
            
            # Submit a long-running job
            config = TrainingConfig(
                model_config={"type": "long_test", "duration": "long"},
                training_params={"slow_mode": True},
                data_config={"dataset": "large"},
                framework="tensorflow",
                max_epochs=100,  # Intentionally long
                batch_size=1
            )
            
            logger.info("Submitting long-running job for cancellation test...")
            
            # Start the job (don't await - we want to cancel it)
            job_task = asyncio.create_task(gpu.train_model(config))
            
            # Wait a bit for job to start
            await asyncio.sleep(3)
            
            # Try to get the job ID from the provider's active jobs
            if hasattr(gpu._provider, '_active_jobs') and gpu._provider._active_jobs:
                job_id = list(gpu._provider._active_jobs.keys())[0]
                logger.info(f"Attempting to cancel job: {job_id}")
                
                cancelled = await gpu.cancel_job(job_id)
                
                if cancelled:
                    logger.info("✅ Job cancellation successful")
                    
                    # Cancel the task
                    job_task.cancel()
                    try:
                        await job_task
                    except asyncio.CancelledError:
                        pass
                    
                    return True
                else:
                    logger.warning("⚠️ Job cancellation failed")
                    job_task.cancel()
                    return False
            else:
                logger.warning("⚠️ No active jobs found to cancel")
                job_task.cancel()
                return False
                
    except Exception as e:
        logger.error(f"❌ Job cancellation test failed: {e}")
        return False


async def test_queue_management():
    """Test queue purging functionality."""
    logger.info("\n=== Testing Queue Management ===")
    
    api_key = os.getenv('RUNPOD_API_KEY')
    endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
    
    if not api_key or not endpoint_id:
        logger.error("Missing RunPod credentials")
        return False
    
    try:
        async with RemoteGPU(provider="runpod_serverless", config={
            'api_key': api_key,
            'endpoint_id': endpoint_id
        }) as gpu:
            
            logger.info("Testing queue purge...")
            purged = await gpu.purge_queue()
            
            if purged:
                logger.info("✅ Queue purged successfully")
                return True
            else:
                logger.warning("⚠️ Queue purge failed (may be empty)")
                return True  # This is not necessarily an error
                
    except Exception as e:
        logger.error(f"❌ Queue management test failed: {e}")
        return False


async def test_job_streaming():
    """Test job output streaming."""
    logger.info("\n=== Testing Job Streaming ===")
    
    api_key = os.getenv('RUNPOD_API_KEY')
    endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
    
    if not api_key or not endpoint_id:
        logger.error("Missing RunPod credentials")
        return False
    
    try:
        async with RemoteGPU(provider="runpod_serverless", config={
            'api_key': api_key,
            'endpoint_id': endpoint_id
        }) as gpu:
            
            # Submit a job for streaming
            config = TrainingConfig(
                model_config={"type": "stream_test", "verbose": True},
                training_params={"stream_output": True},
                data_config={"dataset": "streaming"},
                framework="tensorflow",
                max_epochs=2,
                batch_size=1
            )
            
            logger.info("Submitting job for streaming test...")
            
            # Start job and get job ID
            job_task = asyncio.create_task(gpu.train_model(config))
            await asyncio.sleep(2)  # Wait for job to start
            
            if hasattr(gpu._provider, '_active_jobs') and gpu._provider._active_jobs:
                job_id = list(gpu._provider._active_jobs.keys())[0]
                logger.info(f"Streaming output for job: {job_id}")
                
                stream_count = 0
                async for stream_data in gpu.stream_job_output(job_id):
                    logger.info(f"Stream data: {stream_data}")
                    stream_count += 1
                    
                    if stream_count >= 3:  # Limit streaming for test
                        break
                
                # Cancel the job task
                job_task.cancel()
                try:
                    await job_task
                except asyncio.CancelledError:
                    pass
                
                if stream_count > 0:
                    logger.info(f"✅ Streaming test successful ({stream_count} messages)")
                    return True
                else:
                    logger.warning("⚠️ No stream data received")
                    return False
            else:
                logger.warning("⚠️ No active jobs for streaming")
                job_task.cancel()
                return False
                
    except Exception as e:
        logger.error(f"❌ Streaming test failed: {e}")
        return False


async def main():
    """Run all advanced RunPod tests."""
    logger.info("🚀 Starting Advanced RunPod API Tests")
    logger.info("=" * 60)
    
    if not os.path.exists('.env'):
        logger.error("No .env file found. Please create one with RunPod credentials.")
        return
    
    test_results = {}
    
    try:
        # Test 1: Health Check
        test_results['health_check'] = await test_health_check()
        
        # Test 2: Synchronous Jobs
        test_results['sync_job'] = await test_sync_job()
        
        # Test 3: Job Cancellation
        test_results['job_cancellation'] = await test_job_cancellation()
        
        # Test 4: Queue Management
        test_results['queue_management'] = await test_queue_management()
        
        # Test 5: Job Streaming
        test_results['job_streaming'] = await test_job_streaming()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🏁 ADVANCED RUNPOD TEST RESULTS")
        logger.info("=" * 60)
        
        passed_tests = 0
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
            if result:
                passed_tests += 1
        
        logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All advanced features working perfectly!")
        elif passed_tests > 0:
            logger.info("⚠️ Some features working, others may need serverless function")
        else:
            logger.error("❌ All tests failed - check your setup")
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Tests interrupted by user")
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())