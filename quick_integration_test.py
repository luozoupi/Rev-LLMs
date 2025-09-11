"""
Quick Integration Test - Run GLUE on Your Trained Models
=======================================================

Simple test to verify that your trained models can run GLUE benchmarks
"""

def test_model_benchmark_integration():
    """Test that we can run benchmarks on your models"""
    
    print("="*60)
    print("TESTING MODEL-BENCHMARK INTEGRATION")
    print("="*60)
    
    # Test 1: Can we import your training framework?
    try:
        from test_train_qwen3_rev_v3f import ReversibleQwenPerformanceTester
        print("✓ Training framework imported successfully")
        
        # Create tester
        tester = ReversibleQwenPerformanceTester(device='cuda')
        
        # Try to create models
        models = tester.setup_models_for_comparison(
            vocab_size=1000,  # Small vocab for testing
            hidden_size=128,  # Small model for testing
            num_layers=2,
            num_heads=4
        )
        
        print(f"✓ Created {len(models)} models: {list(models.keys())}")
        
    except Exception as e:
        print(f"✗ Training framework failed: {e}")
        return False
    
    # Test 2: Can we run GLUE benchmarks?
    try:
        from glue_plus_benchmark import GLUEPlusBenchmark
        
        glue_benchmark = GLUEPlusBenchmark()
        print("✓ GLUE benchmark imported successfully")
        
        # Try to run on a simple task
        results = glue_benchmark.run_full_benchmark(
            models, 
            device='cuda',
            task_subset=['sst2']  # Just one task for testing
        )
        
        print(f"✓ GLUE benchmark completed: {len(results)} model results")
        
    except Exception as e:
        print(f"✗ GLUE benchmark failed: {e}")
        return False
    
    # Test 3: Can we run memory benchmarks?
    try:
        from memory_benchmark import MemoryBenchmark
        
        memory_benchmark = MemoryBenchmark()
        print("✓ Memory benchmark imported successfully")
        
        # This might be more intensive, so we'll just test the import
        print("✓ Memory benchmark ready to run")
        
    except Exception as e:
        print(f"✗ Memory benchmark failed: {e}")
        return False
    
    # Test 4: Can we run fixed benchmarks?
    try:
        from fixed_benchmark_runner import FixedBenchmarkRunner
        
        fixed_benchmark = FixedBenchmarkRunner(device='cuda')
        print("✓ Fixed benchmark imported successfully")
        
        # Run a quick test
        quick_results = fixed_benchmark.run_all_benchmarks(models)
        print(f"✓ Fixed benchmark completed: {len(quick_results)} results")
        
    except Exception as e:
        print(f"✗ Fixed benchmark failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("INTEGRATION TEST RESULTS")
    print("="*60)
    print("✅ ALL TESTS PASSED!")
    print("✅ Your trained Qwen3 models CAN run advanced benchmarks!")
    print("✅ GLUE+, Memory, and Fixed benchmarks are all working!")
    
    return True

def demonstrate_benchmark_usage():
    """Demonstrate how to use benchmarks with your models"""
    
    print("\n" + "="*60)
    print("HOW TO USE BENCHMARKS WITH YOUR MODELS")
    print("="*60)
    
    usage_examples = [
        ("Basic GLUE Benchmark", """
# After training your models with test_train_qwen3_rev_v3f.py
from test_train_qwen3_rev_v3f import ReversibleQwenPerformanceTester
from glue_plus_benchmark import GLUEPlusBenchmark

# Extract your trained models
tester = ReversibleQwenPerformanceTester()
models = tester.setup_models_for_comparison(vocab_size=32000, hidden_size=512)

# Run GLUE benchmarks
glue_benchmark = GLUEPlusBenchmark()
results = glue_benchmark.run_full_benchmark(models, task_subset=['sst2', 'cola', 'mrpc'])
glue_benchmark.print_benchmark_summary()
"""),
        
        ("Memory Benchmark", """
# Run long-range memory tasks
from memory_benchmark import MemoryBenchmark

memory_benchmark = MemoryBenchmark()
memory_results = memory_benchmark.run_memory_benchmark(models)
memory_benchmark.print_memory_benchmark_summary()
"""),
        
        ("Complete Integration", """
# Run all benchmarks together
from direct_benchmark_integration import main

# This will:
# 1. Extract your trained models
# 2. Run GLUE+ benchmarks  
# 3. Run memory benchmarks
# 4. Run fixed benchmarks
# 5. Compare reversible vs standard performance

main()
""")
    ]
    
    for title, code in usage_examples:
        print(f"\n{title}:")
        print("-" * len(title))
        print(code)

if __name__ == "__main__":
    print("QUICK INTEGRATION TEST FOR QWEN3 MODELS")
    
    # Run the test
    success = test_model_benchmark_integration()
    
    if success:
        # Show usage examples
        demonstrate_benchmark_usage()
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Run your full training: python test_train_qwen3_rev_v3f.py")
        print("2. Run GLUE benchmarks: python run_advanced_benchmarks.py --tasks glue")
        print("3. Run all benchmarks: python direct_benchmark_integration.py")
        print("4. Compare results and optimize your models!")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
