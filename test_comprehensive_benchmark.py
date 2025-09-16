#!/usr/bin/env python3
"""
Test script for the improved comprehensive Qwen3 vs DiZO benchmark
This script tests the integration and ensures all components work together
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        from comprehensive_qwen3_dizo_benchmark import (
            ComprehensiveQwen3DiZOBenchmark,
            BenchmarkConfig,
            MODEL_CREATION_AVAILABLE,
            GLUE_AVAILABLE,
            MEMORY_AVAILABLE,
            ADVANCED_AVAILABLE,
            COMPREHENSIVE_AVAILABLE,
            DIZO_INTEGRATION_AVAILABLE
        )
        logger.info("✓ Main benchmark imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_config_creation():
    """Test configuration creation"""
    logger.info("Testing configuration creation...")
    
    try:
        from comprehensive_qwen3_dizo_benchmark import BenchmarkConfig
        
        # Test small scale
        config = BenchmarkConfig(scale="small")
        assert config.num_layers == 6
        assert config.hidden_size == 512
        logger.info("✓ Small scale configuration created")
        
        # Test medium scale
        config = BenchmarkConfig(scale="medium")
        assert config.num_layers == 12
        assert config.hidden_size == 768
        logger.info("✓ Medium scale configuration created")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration creation failed: {e}")
        return False

def test_benchmark_initialization():
    """Test benchmark initialization"""
    logger.info("Testing benchmark initialization...")
    
    try:
        from comprehensive_qwen3_dizo_benchmark import ComprehensiveQwen3DiZOBenchmark, BenchmarkConfig
        
        config = BenchmarkConfig(scale="small")
        benchmark = ComprehensiveQwen3DiZOBenchmark(config, device="cpu")  # Use CPU for testing
        
        logger.info("✓ Benchmark initialized successfully")
        logger.info(f"GLUE available: {benchmark.glue_benchmark is not None}")
        logger.info(f"Memory available: {benchmark.memory_benchmark is not None}")
        logger.info(f"Advanced available: {benchmark.advanced_benchmark is not None}")
        logger.info(f"Comprehensive runner available: {benchmark.comprehensive_runner is not None}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Benchmark initialization failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    logger.info("Testing model creation...")
    
    try:
        from comprehensive_qwen3_dizo_benchmark import ComprehensiveQwen3DiZOBenchmark, BenchmarkConfig
        
        # Use CPU for safer testing
        device = "cpu"
        config = BenchmarkConfig(scale="small")
        benchmark = ComprehensiveQwen3DiZOBenchmark(config, device=device)
        
        models = benchmark.trainer.create_models()
        logger.info(f"Created {len(models)} models: {list(models.keys())}")
        
        # Verify all models are on the correct device
        for model_name, model in models.items():
            try:
                model_device = next(model.parameters()).device
                logger.info(f"  {model_name}: {model_device}")
                if str(model_device) != device:
                    logger.warning(f"  Model {model_name} is on {model_device} but expected {device}")
            except Exception as e:
                logger.warning(f"  Could not check device for {model_name}: {e}")
        
        if len(models) > 0:
            logger.info("✓ Model creation successful")
            return True
        else:
            logger.warning("⚠ No models created (may be expected if dependencies missing)")
            return False
            
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        return False

def test_quick_benchmark():
    """Test a quick benchmark run"""
    logger.info("Testing quick benchmark run...")
    
    try:
        from comprehensive_qwen3_dizo_benchmark import ComprehensiveQwen3DiZOBenchmark, BenchmarkConfig
        
        # Use CPU for safer testing and very small config
        device = "cpu"
        config = BenchmarkConfig(
            scale="small",
            num_epochs=1,  # Very short for testing
            eval_size=50,  # Small eval size
            train_size=100,  # Small training size
            run_glue=True,
            run_memory=False,  # Skip memory for quick test
            run_advanced=False  # Skip advanced for quick test
        )
        
        benchmark = ComprehensiveQwen3DiZOBenchmark(config, device=device)
        
        # Run a minimal benchmark
        results = benchmark.run_full_benchmark(
            tasks=['sst2'],  # Single task
            compare_dizo=False
        )
        
        logger.info(f"Benchmark completed with {len(results)} result categories")
        for category, result in results.items():
            logger.info(f"  {category}: {'✓' if not isinstance(result, dict) or 'error' not in result else '✗'}")
        
        logger.info("✓ Quick benchmark test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Quick benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("Starting comprehensive benchmark test suite")
    logger.info("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Config Creation Test", test_config_creation),
        ("Benchmark Initialization Test", test_benchmark_initialization),
        ("Model Creation Test", test_model_creation),
        ("Quick Benchmark Test", test_quick_benchmark),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! The comprehensive benchmark is ready to use.")
    else:
        logger.warning(f"⚠ {len(results) - passed} tests failed. Check the logs above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)