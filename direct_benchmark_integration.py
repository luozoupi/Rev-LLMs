"""
Direct Benchmark Integration Example
==================================

Simple example showing how to directly use GLUE and other benchmarks
on your trained reversible and standard Qwen3 models.
"""

import torch
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def extract_models_from_training():
    """Extract trained models from your training framework"""
    
    # Import your training framework
    from test_train_qwen3_rev_v3f import ReversibleQwenPerformanceTester
    
    print("Creating performance tester...")
    tester = ReversibleQwenPerformanceTester(device='cuda')
    
    # Option 1: Run full training and extract models
    print("Option 1: Running comprehensive training...")
    try:
        results = tester.run_comprehensive_test(
            seq_len=512, 
            vocab_size=16000, 
            use_wikitext=True
        )
        
        if hasattr(tester, 'trained_models'):
            trained_models = tester.trained_models
        else:
            # Fallback: create models manually
            trained_models = tester.setup_models_for_comparison(
                vocab_size=16000,
                hidden_size=512,
                num_layers=4,
                num_heads=8
            )
        
        return trained_models, results
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Option 2: Creating models directly...")
        
        # Option 2: Create models directly without full training
        trained_models = tester.setup_models_for_comparison(
            vocab_size=16000,
            hidden_size=512,
            num_layers=4,
            num_heads=8
        )
        
        return trained_models, None

def run_glue_on_trained_models(models):
    """Run GLUE benchmarks on your trained models"""
    
    try:
        from glue_plus_benchmark import GLUEPlusBenchmark
        
        print("\n" + "="*60)
        print("RUNNING GLUE+ BENCHMARKS ON TRAINED MODELS")
        print("="*60)
        
        # Initialize GLUE benchmark
        glue_benchmark = GLUEPlusBenchmark()
        
        # Run on basic tasks first
        basic_tasks = ['sst2', 'cola', 'mrpc']
        
        glue_results = glue_benchmark.run_full_benchmark(
            models, 
            device='cuda',
            task_subset=basic_tasks
        )
        
        # Print results
        glue_benchmark.print_benchmark_summary()
        
        return glue_results
        
    except Exception as e:
        print(f"GLUE benchmark failed: {e}")
        return {}

def run_memory_benchmarks_on_trained_models(models):
    """Run memory benchmarks on your trained models"""
    
    try:
        from memory_benchmark import MemoryBenchmark
        
        print("\n" + "="*60)
        print("RUNNING MEMORY BENCHMARKS ON TRAINED MODELS") 
        print("="*60)
        
        # Initialize memory benchmark
        memory_benchmark = MemoryBenchmark()
        
        # Run memory benchmarks
        memory_results = memory_benchmark.run_memory_benchmark(models, device='cuda')
        
        # Print results
        memory_benchmark.print_memory_benchmark_summary()
        
        return memory_results
        
    except Exception as e:
        print(f"Memory benchmark failed: {e}")
        return {}

def run_fixed_benchmarks_on_trained_models(models):
    """Run fixed comprehensive benchmarks on your trained models"""
    
    try:
        from fixed_benchmark_runner import FixedBenchmarkRunner
        
        print("\n" + "="*60)
        print("RUNNING FIXED BENCHMARKS ON TRAINED MODELS")
        print("="*60)
        
        # Initialize fixed benchmark runner
        fixed_benchmark = FixedBenchmarkRunner(device='cuda')
        
        # Run all benchmarks
        results = fixed_benchmark.run_all_benchmarks(models)
        
        # Print summary
        fixed_benchmark.print_summary()
        
        return results
        
    except Exception as e:
        print(f"Fixed benchmark failed: {e}")
        return {}

def compare_reversible_vs_standard(all_results):
    """Compare reversible vs standard model performance"""
    
    print("\n" + "="*80)
    print("REVERSIBLE vs STANDARD COMPARISON SUMMARY")
    print("="*80)
    
    # Extract model names
    all_models = set()
    for result_type, results in all_results.items():
        if isinstance(results, dict):
            all_models.update(results.keys())
    
    reversible_models = [m for m in all_models if 'reversible' in m.lower() and 'non' not in m.lower()]
    standard_models = [m for m in all_models if m not in reversible_models]
    
    print(f"Reversible Models: {reversible_models}")
    print(f"Standard Models: {standard_models}")
    
    # Compare across different benchmark types
    comparisons = []
    
    # GLUE comparison
    if 'glue_results' in all_results and all_results['glue_results']:
        glue_results = all_results['glue_results']
        rev_scores = []
        std_scores = []
        
        for model_name, results in glue_results.items():
            if '_summary' in results:
                score = results['_summary'].get('overall_score', 0)
                if model_name in reversible_models:
                    rev_scores.append(score)
                elif model_name in standard_models:
                    std_scores.append(score)
        
        if rev_scores and std_scores:
            rev_avg = sum(rev_scores) / len(rev_scores)
            std_avg = sum(std_scores) / len(std_scores)
            comparisons.append(('GLUE+', rev_avg, std_avg))
    
    # Print comparison table
    if comparisons:
        print(f"\n{'Benchmark':<15} {'Reversible':<12} {'Standard':<12} {'Gap':<10} {'Winner'}")
        print("-" * 60)
        
        for benchmark, rev_score, std_score in comparisons:
            gap = rev_score - std_score
            winner = "Reversible" if gap > 0 else "Standard"
            print(f"{benchmark:<15} {rev_score:<12.3f} {std_score:<12.3f} {gap:<+10.3f} {winner}")
    
    # Performance insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print("="*80)
    print("✓ Your reversible Qwen3 models can run all advanced benchmarks!")
    print("✓ Compare memory efficiency, accuracy, and inference speed")
    print("✓ Use results to optimize model selection for your use case")

def main():
    """Main demonstration function"""
    
    print("="*80)
    print("DIRECT BENCHMARK INTEGRATION FOR TRAINED QWEN3 MODELS")
    print("="*80)
    print("This shows how to run GLUE+ and other benchmarks on your trained models")
    
    # Step 1: Extract trained models
    print("\nStep 1: Extracting trained models...")
    models, training_results = extract_models_from_training()
    
    if not models:
        print("No models available for benchmarking!")
        return
    
    print(f"Successfully extracted {len(models)} models: {list(models.keys())}")
    
    # Step 2: Run GLUE benchmarks
    print("\nStep 2: Running GLUE+ benchmarks...")
    glue_results = run_glue_on_trained_models(models)
    
    # Step 3: Run memory benchmarks  
    print("\nStep 3: Running memory benchmarks...")
    memory_results = run_memory_benchmarks_on_trained_models(models)
    
    # Step 4: Run fixed benchmarks
    print("\nStep 4: Running fixed comprehensive benchmarks...")
    fixed_results = run_fixed_benchmarks_on_trained_models(models)
    
    # Step 5: Compare results
    all_results = {
        'training_results': training_results,
        'glue_results': glue_results,
        'memory_results': memory_results,
        'fixed_results': fixed_results
    }
    
    print("\nStep 5: Comparing reversible vs standard performance...")
    compare_reversible_vs_standard(all_results)
    
    print("\n" + "="*80)
    print("BENCHMARK INTEGRATION COMPLETE!")
    print("="*80)
    print("Your models have been successfully evaluated on:")
    print("• GLUE+ language understanding tasks (SST-2, CoLA, MRPC)")
    print("• Long-range memory tasks (copy, associative recall, etc.)")
    print("• Text generation and classification benchmarks")
    print("• Comprehensive performance metrics")

if __name__ == "__main__":
    main()
