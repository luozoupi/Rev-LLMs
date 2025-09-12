#!/usr/bin/env python3
"""
Quick Integration Test for Enhanced Training Framework
====================================================

This script tests the basic functionality of the enhanced training framework
to ensure all components are properly integrated.
"""

import sys
import os
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from enhanced_train_and_benchmark import EnhancedTrainingAndBenchmarkRunner, BenchmarkDatasetLoader
        print("✅ Enhanced training framework imported successfully")
    except ImportError as e:
        print(f"❌ Enhanced training framework import failed: {e}")
        return False
    
    try:
        from run_advanced_benchmarks import ComprehensiveBenchmarkRunner
        print("✅ Comprehensive benchmark runner imported successfully")
    except ImportError as e:
        print(f"❌ Comprehensive benchmark runner import failed: {e}")
        # Not critical
    
    try:
        from advanced_benchmarks import AdvancedBenchmarkSuite
        print("✅ Advanced benchmarks imported successfully")
    except ImportError as e:
        print(f"❌ Advanced benchmarks import failed: {e}")
        # Not critical
    
    try:
        from test_train_qwen3_rev_v3f import ReversibleQwenPerformanceTester
        print("✅ Base performance tester imported successfully")
    except ImportError as e:
        print(f"❌ Base performance tester import failed: {e}")
        # Not critical
    
    return True

def test_dataset_loader():
    """Test dataset loading functionality"""
    print("\nTesting dataset loader...")
    
    try:
        from enhanced_train_and_benchmark import BenchmarkDatasetLoader, BenchmarkDatasetConfig
        
        loader = BenchmarkDatasetLoader(cache_dir='./test_cache')
        print("✅ Dataset loader created successfully")
        
        # Test simple configuration
        config = BenchmarkDatasetConfig(
            name='test',
            task_type='language_modeling',
            seq_length=128,
            num_train_samples=100,
            num_val_samples=50,
            vocab_size=1000
        )
        print("✅ Dataset configuration created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loader test failed: {e}")
        return False

def test_model_creation():
    """Test model creation functionality"""
    print("\nTesting model creation...")
    
    try:
        from enhanced_train_and_benchmark import EnhancedTrainingAndBenchmarkRunner
        
        runner = EnhancedTrainingAndBenchmarkRunner(device='cpu')  # Use CPU for testing
        print("✅ Training runner created successfully")
        
        # Try to create small models
        models = runner.create_models(
            model_types=['reversible', 'standard'],
            vocab_size=1000,
            hidden_size=64,
            num_layers=2
        )
        
        if models:
            print(f"✅ Created {len(models)} models: {list(models.keys())}")
            return True
        else:
            print("⚠️  No models created (may be due to missing dependencies)")
            return False
            
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def test_training_config():
    """Test training configuration"""
    print("\nTesting training configuration...")
    
    try:
        from enhanced_train_and_benchmark import TrainingConfig, EnhancedTrainingAndBenchmarkRunner
        
        runner = EnhancedTrainingAndBenchmarkRunner()
        
        # Test training configs for different datasets
        datasets = ['wikitext', 'enwik8', 'code', 'math', 'squad']
        model_types = ['reversible', 'standard']
        
        for dataset in datasets:
            for model_type in model_types:
                config = runner.get_training_config(dataset, model_type)
                assert isinstance(config, TrainingConfig)
                assert config.dataset_name == dataset
                
        print("✅ Training configurations work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Training configuration test failed: {e}")
        return False

def test_integration():
    """Test overall integration"""
    print("\nTesting integration...")
    
    try:
        from enhanced_train_and_benchmark import EnhancedTrainingAndBenchmarkRunner
        
        # Create runner
        runner = EnhancedTrainingAndBenchmarkRunner(device='cpu')
        
        # Test that all components are initialized
        assert runner.dataset_loader is not None
        print("✅ Dataset loader initialized")
        
        # Check if optional components are available
        if runner.base_tester:
            print("✅ Base performance tester available")
        else:
            print("⚠️  Base performance tester not available")
            
        if runner.benchmark_runner:
            print("✅ Comprehensive benchmark runner available")
        else:
            print("⚠️  Comprehensive benchmark runner not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("ENHANCED TRAINING FRAMEWORK INTEGRATION TEST")
    print("="*60)
    
    tests = [
        test_imports,
        test_dataset_loader,
        test_model_creation,
        test_training_config,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced framework is ready to use.")
        print("\nTo run a simple training example:")
        print("python enhanced_train_and_benchmark.py --datasets wikitext --models reversible,standard")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("The framework may still work but some features might be unavailable.")
    
    # Clean up test cache
    import shutil
    test_cache = './test_cache'
    if os.path.exists(test_cache):
        try:
            shutil.rmtree(test_cache)
            print(f"\n🧹 Cleaned up test cache: {test_cache}")
        except:
            pass

if __name__ == "__main__":
    main()
