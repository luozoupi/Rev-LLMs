"""
Integration Script for Qwen3 Evaluation Suite
============================================

This script integrates the Qwen3 Evaluation Suite with the existing training frameworks
and provides easy access to run comprehensive evaluations.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.extend([str(current_dir), str(parent_dir)])

def run_enhanced_training_and_evaluation():
    """Run the enhanced training script followed by evaluation"""
    
    logger.info("Running enhanced training and benchmarking...")
    
    # Change to parent directory
    os.chdir(parent_dir)
    
    # Run enhanced training
    cmd = "python enhanced_train_and_benchmark.py --datasets wikitext --models reversible,standard"
    logger.info(f"Executing: {cmd}")
    
    result = os.system(cmd)
    if result != 0:
        logger.error("Enhanced training failed")
        return False
    
    # Change back to evaluation suite directory
    os.chdir(current_dir)
    
    # Run evaluation suite
    cmd = "python run_evaluation.py --config configs/quick_eval.yaml"
    logger.info(f"Executing: {cmd}")
    
    result = os.system(cmd)
    if result != 0:
        logger.error("Evaluation suite failed")
        return False
    
    logger.info("✓ Enhanced training and evaluation completed successfully")
    return True

def run_comprehensive_comparison():
    """Run comprehensive comparison between reversible and standard models"""
    
    logger.info("Running comprehensive model comparison...")
    
    # Run comprehensive evaluation
    cmd = "python run_evaluation.py --config configs/comprehensive_eval.yaml"
    logger.info(f"Executing: {cmd}")
    
    result = os.system(cmd)
    if result != 0:
        logger.error("Comprehensive evaluation failed")
        return False
    
    # Generate analysis report
    logger.info("Generating comprehensive analysis report...")
    
    try:
        from analysis_and_comparison import ComprehensiveReportGenerator
        import json
        
        # Load results
        results_path = Path("comprehensive_evaluation_results/evaluation_results.json")
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Generate report
            report_generator = ComprehensiveReportGenerator()
            report_path = report_generator.generate_full_report(results)
            
            logger.info(f"✓ Comprehensive analysis report generated: {report_path}")
            return True
        else:
            logger.error("No evaluation results found")
            return False
            
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return False

def run_zero_order_comparison():
    """Run zero-order optimization comparison"""
    
    logger.info("Running zero-order optimization comparison...")
    
    try:
        from zero_order_optimization import create_zo_trainer_for_qwen3, ZeroOrderConfig
        from run_evaluation import ModelFactory, DatasetLoader
        from transformers import AutoTokenizer
        import torch
        from torch.utils.data import DataLoader
        
        # Create models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        reversible_model = ModelFactory.create_model("reversible_qwen3", "small", device)
        standard_model = ModelFactory.create_model("standard_qwen3", "small", device)
        
        # Create tokenizer and dataset
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dataset_loader = DatasetLoader(tokenizer)
        memory_dataset = dataset_loader.create_memory_stress_dataset([512], num_samples=50)
        
        # Create dataloader
        dataloader = DataLoader(memory_dataset, batch_size=4, shuffle=True)
        
        # Train both models with ZO
        results = {}
        
        for model_name, model in [("reversible", reversible_model), ("standard", standard_model)]:
            logger.info(f"Training {model_name} model with zero-order optimization...")
            
            trainer = create_zo_trainer_for_qwen3(
                model=model,
                train_dataloader=dataloader,
                enhanced_zo=True,
                learning_rate=1e-3,
                num_epochs=2
            )
            
            training_results = trainer.train()
            results[model_name] = training_results
            
            logger.info(f"✓ {model_name} ZO training completed")
        
        # Compare results
        logger.info("Zero-Order Optimization Comparison Results:")
        logger.info("=" * 50)
        
        for model_name, result in results.items():
            optimizer_stats = result.get('optimizer_stats', {})
            final_eval = result.get('final_eval', {})
            
            logger.info(f"{model_name.upper()} Model:")
            logger.info(f"  Final Loss: {optimizer_stats.get('final_loss', 0):.4f}")
            logger.info(f"  Convergence Rate: {optimizer_stats.get('convergence_rate', 0):.4f}")
            logger.info(f"  Training Time: {result.get('total_training_time', 0):.2f}s")
            if final_eval:
                logger.info(f"  Final Accuracy: {final_eval.get('eval_accuracy', 0):.4f}")
        
        logger.info("✓ Zero-order optimization comparison completed")
        return True
        
    except Exception as e:
        logger.error(f"Zero-order optimization comparison failed: {e}")
        return False

def run_memory_scaling_analysis():
    """Run detailed memory scaling analysis"""
    
    logger.info("Running memory scaling analysis...")
    
    try:
        from run_evaluation import ModelFactory, DatasetLoader, MemoryProfiler
        from transformers import AutoTokenizer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create models
        models = {
            "reversible_small": ModelFactory.create_model("reversible_qwen3", "small", device),
            "standard_small": ModelFactory.create_model("standard_qwen3", "small", device),
        }
        
        # Test sequence lengths
        test_lengths = [256, 512, 1024, 2048]
        if device == "cuda":
            test_lengths.extend([4096, 8192])
        
        # Memory profiler
        memory_profiler = MemoryProfiler(device)
        
        # Results storage
        memory_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Testing memory scaling for {model_name}...")
            
            model_memory_results = {}
            
            for length in test_lengths:
                try:
                    # Create input batch
                    batch = {
                        'input_ids': torch.randint(0, 1000, (1, length)).to(device),
                        'attention_mask': torch.ones((1, length)).to(device),
                    }
                    
                    # Measure memory
                    with memory_profiler.measure_memory(f"length_{length}"):
                        with torch.no_grad():
                            outputs = model(**batch)
                    
                    memory_info = memory_profiler.measurements[f"length_{length}"]
                    model_memory_results[f"length_{length}"] = memory_info
                    
                    logger.info(f"  Length {length}: {memory_info['memory_used_mb']:.1f} MB")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"  Length {length}: OOM")
                        model_memory_results[f"length_{length}"] = {"error": "OOM"}
                    else:
                        raise e
            
            memory_results[model_name] = {"memory_profiles": model_memory_results}
        
        # Analyze results
        from analysis_and_comparison import MemoryAnalyzer
        
        memory_analyzer = MemoryAnalyzer()
        scaling_analysis = memory_analyzer.analyze_memory_scaling(memory_results)
        
        logger.info("Memory Scaling Analysis Results:")
        logger.info("=" * 40)
        
        for model_name, scaling_info in scaling_analysis.items():
            if 'error' not in scaling_info:
                exponent = scaling_info['scaling_exponent']
                complexity = scaling_info['memory_complexity']
                r_squared = scaling_info['r_squared']
                
                logger.info(f"{model_name}:")
                logger.info(f"  Scaling Exponent: {exponent:.2f}")
                logger.info(f"  Complexity: {complexity}")
                logger.info(f"  R-squared: {r_squared:.3f}")
        
        logger.info("✓ Memory scaling analysis completed")
        return True
        
    except Exception as e:
        logger.error(f"Memory scaling analysis failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Qwen3 Evaluation Suite Integration')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'comprehensive', 'training', 'zero_order', 'memory'],
                        help='Evaluation mode to run')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Running Qwen3 Evaluation Suite Integration")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)
    
    success = False
    
    if args.mode == 'quick':
        # Quick test
        logger.info("Running quick test and evaluation...")
        os.system("python quick_test.py")
        os.system("python run_evaluation.py --config configs/quick_eval.yaml --device " + device)
        success = True
        
    elif args.mode == 'comprehensive':
        success = run_comprehensive_comparison()
        
    elif args.mode == 'training':
        success = run_enhanced_training_and_evaluation()
        
    elif args.mode == 'zero_order':
        success = run_zero_order_comparison()
        
    elif args.mode == 'memory':
        success = run_memory_scaling_analysis()
    
    if success:
        logger.info("🎉 Integration completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Check the generated reports in the output directories")
        logger.info("2. Review the visualizations in the figures/ directory")
        logger.info("3. Compare results with DiZO benchmarks")
    else:
        logger.error("❌ Integration failed. Check the logs above for details.")
    
    return success

if __name__ == "__main__":
    import torch
    success = main()
    sys.exit(0 if success else 1)
