#!/usr/bin/env python3
"""
Simple runner script for the comprehensive Qwen3 vs DiZO benchmark

Usage examples:
    python run_benchmark.py --quick                    # Quick test with basic settings
    python run_benchmark.py --scale medium            # Medium scale benchmark
    python run_benchmark.py --full --compare-dizo     # Full benchmark with DiZO comparison
    python run_benchmark.py --test                    # Just run tests to verify setup
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test_suite():
    """Run the test suite"""
    logger.info("Running test suite...")
    from test_comprehensive_benchmark import run_all_tests
    return run_all_tests()

def run_quick_benchmark():
    """Run a quick benchmark for testing"""
    logger.info("Running quick benchmark...")
    
    from comprehensive_qwen3_dizo_benchmark import ComprehensiveQwen3DiZOBenchmark, BenchmarkConfig
    
    config = BenchmarkConfig(
        scale="small",
        num_epochs=1,
        eval_size=100,
        run_glue=True,
        run_memory=True,
        run_advanced=False
    )
    
    benchmark = ComprehensiveQwen3DiZOBenchmark(config, device="cpu")
    
    results = benchmark.run_full_benchmark(
        tasks=['sst2', 'cola'],
        compare_dizo=False
    )
    
    benchmark.print_comprehensive_summary()
    return results

def run_full_benchmark(scale="medium", compare_dizo=False, device="cuda"):
    """Run full benchmark"""
    logger.info(f"Running full benchmark with scale={scale}, compare_dizo={compare_dizo}")
    
    from comprehensive_qwen3_dizo_benchmark import ComprehensiveQwen3DiZOBenchmark, BenchmarkConfig
    
    config = BenchmarkConfig(
        scale=scale,
        run_glue=True,
        run_memory=True,
        run_advanced=True,
        run_generation=True
    )
    
    benchmark = ComprehensiveQwen3DiZOBenchmark(config, device=device)
    
    # Select tasks based on scale
    if scale == "small":
        tasks = ['sst2', 'cola', 'mrpc']
    elif scale == "medium":
        tasks = ['sst2', 'cola', 'mrpc', 'sts-b', 'qqp']
    else:
        tasks = None  # Use all available tasks
    
    results = benchmark.run_full_benchmark(
        tasks=tasks,
        compare_dizo=compare_dizo
    )
    
    benchmark.print_comprehensive_summary()
    
    # Save results
    output_file = f"benchmark_results_{scale}.json"
    benchmark.save_results(output_file)
    logger.info(f"Results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive Qwen3 vs DiZO benchmark")
    
    parser.add_argument("--test", action="store_true", help="Run test suite only")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark test")
    parser.add_argument("--full", action="store_true", help="Run full benchmark")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium",
                        help="Benchmark scale")
    parser.add_argument("--compare-dizo", action="store_true", help="Include DiZO comparison")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    try:
        if args.test:
            success = run_test_suite()
            if not success:
                logger.error("Tests failed!")
                sys.exit(1)
            logger.info("All tests passed!")
            
        elif args.quick:
            results = run_quick_benchmark()
            logger.info("Quick benchmark completed successfully!")
            
        elif args.full:
            results = run_full_benchmark(
                scale=args.scale,
                compare_dizo=args.compare_dizo,
                device=args.device
            )
            logger.info("Full benchmark completed successfully!")
            
        else:
            # Default: run quick test
            logger.info("No specific option selected, running quick benchmark...")
            results = run_quick_benchmark()
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()