"""
DiZO-Scale Training and Benchmarking
===================================

This script properly trains your reversible and standard Qwen3 models on the same
datasets and scale as DiZO, then runs comprehensive benchmarks for comparison.

Usage:
python train_and_benchmark_dizo_scale.py --scale medium --tasks sst2,cola,mrpc
python train_and_benchmark_dizo_scale.py --scale large --tasks all --full_training
python train_and_benchmark_dizo_scale.py --replicate_dizo --tasks sst2  # Match DiZO's exact setup
"""

import torch
import sys
import os
import json
import time
import argparse
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the fixed comprehensive benchmark
try:
    from comprehensive_qwen3_dizo_benchmark_2 import (
        ComprehensiveQwen3DiZOBenchmark, BenchmarkConfig,
        DiZODatasetCompatibilityLayer, ModelTrainer
    )
    BENCHMARK_AVAILABLE = True
    print("✓ Fixed comprehensive benchmark available")
except ImportError as e:
    print(f"✗ Comprehensive benchmark not available: {e}")
    BENCHMARK_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DiZOScaleTrainer:
    """Trainer that replicates DiZO's exact training setup and scale"""
    
    def __init__(self, scale='medium', replicate_dizo=False):
        self.scale = scale
        self.replicate_dizo = replicate_dizo
        
        # Ensure device is consistently handled as torch.device
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
        
        logger.info(f"Using device: {self.device}")
        
        # DiZO-specific configurations
        self.dizo_configs = {
            'sst2': {
                'epochs': 4,
                'batch_size': 32,
                'learning_rate': 2e-5,
                'warmup_ratio': 0.1,
                'max_seq_length': 128,
                'expected_accuracy': 0.936,  # From DiZO logs
            },
            'cola': {
                'epochs': 5,
                'batch_size': 16,
                'learning_rate': 1e-5,
                'warmup_ratio': 0.1,
                'max_seq_length': 128,
                'expected_accuracy': 0.847,
            },
            'mrpc': {
                'epochs': 3,
                'batch_size': 16,
                'learning_rate': 2e-5,
                'warmup_ratio': 0.1,
                'max_seq_length': 256,
                'expected_accuracy': 0.85,
            }
        }
        
        # Create benchmark config
        if replicate_dizo:
            # Match DiZO's exact model configuration
            self.config = BenchmarkConfig(
                scale='medium',
                vocab_size=32000,
                hidden_size=768,
                num_layers=12,
                num_attention_heads=12,
                max_seq_length=512,
                batch_size=32,
                learning_rate=2e-5,
                num_epochs=4,
                warmup_ratio=0.1,
                weight_decay=0.01,
                gradient_clip_norm=1.0
            )
        else:
            self.config = BenchmarkConfig(scale=scale)
            
        logger.info(f"Initialized DiZO-scale trainer: {scale} scale, replicate_dizo={replicate_dizo}")
    
    def train_and_benchmark(self, tasks):
        """Main training and benchmarking pipeline"""
        logger.info(f"Starting DiZO-scale training on tasks: {tasks}")
        
        if not BENCHMARK_AVAILABLE:
            logger.error("Benchmark framework not available!")
            return None
        
        # Initialize benchmark
        benchmark = ComprehensiveQwen3DiZOBenchmark(self.config, device=self.device)
        
        # Phase 1: Create and verify models
        logger.info("Phase 1: Creating models...")
        models = benchmark.trainer.create_models()
        
        if not models:
            logger.error("Failed to create models!")
            return None
        
        # Test models with sample input
        self._test_models(models)
        
        # Phase 2: Fine-tune on each task
        logger.info("Phase 2: Fine-tuning on tasks...")
        training_results = {}
        
        for task in tasks:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING ON TASK: {task.upper()}")
            logger.info(f"{'='*60}")
            
            # Update config for task-specific training
            if self.replicate_dizo and task in self.dizo_configs:
                task_config = self.dizo_configs[task]
                logger.info(f"Using DiZO configuration for {task}: {task_config}")
                
                # Temporarily update benchmark config
                original_epochs = benchmark.config.num_epochs
                original_batch = benchmark.config.batch_size
                original_lr = benchmark.config.learning_rate
                
                benchmark.config.num_epochs = task_config['epochs']
                benchmark.config.batch_size = task_config['batch_size']
                benchmark.config.learning_rate = task_config['learning_rate']
            
            task_results = {}
            
            # Train each model on the task
            for model_name, model in models.items():
                logger.info(f"\nTraining {model_name} on {task}...")
                
                try:
                    start_time = time.time()
                    result = benchmark.trainer.train_model_on_task(model, task, model_name)
                    training_time = time.time() - start_time
                    
                    result['training_time'] = training_time
                    task_results[model_name] = result
                    
                    logger.info(f"Completed {model_name} on {task}:")
                    logger.info(f"  Best metric: {result.get('best_metric', 0):.4f}")
                    logger.info(f"  Epochs trained: {result.get('epochs_trained', 0)}")
                    logger.info(f"  Training time: {training_time:.1f}s")
                    
                    # Compare with DiZO if available
                    if self.replicate_dizo and task in self.dizo_configs:
                        expected = self.dizo_configs[task]['expected_accuracy']
                        actual = result.get('best_metric', 0)
                        diff = actual - expected
                        status = "✓" if diff > -0.02 else "⚠" if diff > -0.05 else "✗"
                        logger.info(f"  vs DiZO: {actual:.3f} vs {expected:.3f} ({diff:+.3f}) {status}")
                    
                except Exception as e:
                    logger.error(f"Training failed for {model_name} on {task}: {e}")
                    import traceback
                    traceback.print_exc()
            
            training_results[task] = task_results
            
            # Restore original config
            if self.replicate_dizo and task in self.dizo_configs:
                benchmark.config.num_epochs = original_epochs
                benchmark.config.batch_size = original_batch
                benchmark.config.learning_rate = original_lr
        
        # Phase 3: Comprehensive evaluation
        logger.info("\nPhase 3: Comprehensive evaluation...")
        benchmark.results['training'] = training_results
        
        # Run full benchmark suite
        try:
            full_results = benchmark.run_full_benchmark(tasks=tasks, compare_dizo=self.replicate_dizo)
            benchmark.results.update(full_results)
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
        
        # Phase 4: Results analysis
        self._analyze_results(benchmark.results, tasks)
        
        # Save results
        self._save_results(benchmark.results)
        
        return benchmark.results
    
    def _test_models(self, models):
        """Test that models work correctly with sample inputs"""
        logger.info("Testing models with sample inputs...")
        
        for model_name, model in models.items():
            try:
                # Test forward pass
                model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                test_input = torch.randint(1, 1000, (2, 64)).to(model_device)
                
                with torch.no_grad():
                    output = model(test_input)
                
                # Handle dictionary output
                if isinstance(output, dict):
                    if 'logits' in output:
                        shape_info = f"logits: {output['logits'].shape}"
                    else:
                        first_key = next(iter(output.keys()))
                        shape_info = f"{first_key}: {output[first_key].shape}"
                else:
                    shape_info = f"tensor: {output.shape}"
                
                logger.info(f"  ✓ {model_name}: {shape_info}")
                
                # Test backward pass
                if isinstance(output, dict):
                    logits = output.get('logits', output[next(iter(output.keys()))])
                else:
                    logits = output
                    
                loss = logits.sum()
                loss.backward()
                logger.info(f"  ✓ {model_name}: backward pass successful")
                
            except Exception as e:
                logger.error(f"  ✗ {model_name}: test failed - {e}")
    
    def _analyze_results(self, results, tasks):
        """Analyze and print comprehensive results"""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE RESULTS ANALYSIS")
        logger.info("="*80)
        
        # Training results analysis
        if 'training' in results:
            logger.info("\n📊 TRAINING RESULTS:")
            
            for task, task_results in results['training'].items():
                logger.info(f"\n{task.upper()}:")
                
                reversible_result = task_results.get('reversible_qwen3', {})
                standard_result = task_results.get('standard_qwen3', {})
                
                if reversible_result and standard_result:
                    rev_metric = reversible_result.get('best_metric', 0)
                    std_metric = standard_result.get('best_metric', 0)
                    rev_time = reversible_result.get('training_time', 0)
                    std_time = standard_result.get('training_time', 0)
                    
                    logger.info(f"  Reversible: {rev_metric:.4f} ({rev_time:.1f}s)")
                    logger.info(f"  Standard:   {std_metric:.4f} ({std_time:.1f}s)")
                    logger.info(f"  Difference: {rev_metric - std_metric:+.4f}")
                    
                    # DiZO comparison
                    if self.replicate_dizo and task in self.dizo_configs:
                        expected = self.dizo_configs[task]['expected_accuracy']
                        logger.info(f"  DiZO target: {expected:.4f}")
                        logger.info(f"  Rev vs DiZO: {rev_metric - expected:+.4f}")
                        logger.info(f"  Std vs DiZO: {std_metric - expected:+.4f}")
        
        # Overall comparison
        if 'training' in results:
            all_rev_scores = []
            all_std_scores = []
            
            for task_results in results['training'].values():
                if 'reversible_qwen3' in task_results and 'standard_qwen3' in task_results:
                    all_rev_scores.append(task_results['reversible_qwen3'].get('best_metric', 0))
                    all_std_scores.append(task_results['standard_qwen3'].get('best_metric', 0))
            
            if all_rev_scores and all_std_scores:
                import numpy as np
                rev_mean = np.mean(all_rev_scores)
                std_mean = np.mean(all_std_scores)
                
                logger.info(f"\n📈 OVERALL PERFORMANCE:")
                logger.info(f"  Reversible average: {rev_mean:.4f}")
                logger.info(f"  Standard average:   {std_mean:.4f}")
                logger.info(f"  Performance gap:    {rev_mean - std_mean:+.4f}")
                
                if rev_mean > std_mean:
                    logger.info("  🏆 Reversible models perform better on average")
                elif std_mean > rev_mean:
                    logger.info("  🏆 Standard models perform better on average")
                else:
                    logger.info("  🤝 Models perform similarly on average")
    
    def _save_results(self, results):
        """Save detailed results to files"""
        timestamp = int(time.time())
        results_dir = Path(f'dizo_scale_results_{timestamp}')
        results_dir.mkdir(exist_ok=True)
        
        # Save full results
        results_file = results_dir / 'full_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = results_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("DiZO-Scale Training and Benchmarking Results\n")
            f.write("=" * 50 + "\n\n")
            
            if 'training' in results:
                f.write("Training Results:\n")
                for task, task_results in results['training'].items():
                    f.write(f"\n{task.upper()}:\n")
                    for model_name, result in task_results.items():
                        best_metric = result.get('best_metric', 0)
                        epochs = result.get('epochs_trained', 0)
                        f.write(f"  {model_name}: {best_metric:.4f} ({epochs} epochs)\n")
        
        logger.info(f"Results saved to {results_dir}")
        return results_dir

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DiZO-Scale Training and Benchmarking")
    
    parser.add_argument('--scale', choices=['small', 'medium', 'large'], default='medium',
                        help='Training scale')
    parser.add_argument('--tasks', type=str, default='sst2,cola,mrpc',
                        help='Tasks to train on (comma-separated)')
    parser.add_argument('--replicate_dizo', action='store_true',
                        help='Use exact DiZO configurations')
    parser.add_argument('--full_training', action='store_true',
                        help='Train with full epochs (no early stopping)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()
    
    # Parse tasks
    if args.tasks == 'all':
        tasks = ['sst2', 'cola', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte']
    elif args.tasks == 'basic':
        tasks = ['sst2', 'cola', 'mrpc']
    else:
        tasks = args.tasks.split(',')
    
    print("DiZO-Scale Training and Benchmarking")
    print("=" * 50)
    print(f"Scale: {args.scale}")
    print(f"Tasks: {tasks}")
    print(f"Replicate DiZO: {args.replicate_dizo}")
    print(f"Device: {args.device}")
    print()
    
    # Initialize trainer
    trainer = DiZOScaleTrainer(
        scale=args.scale,
        replicate_dizo=args.replicate_dizo
    )
    
    # Run training and benchmarking
    try:
        results = trainer.train_and_benchmark(tasks)
        
        if results:
            print("\n✅ Training and benchmarking completed successfully!")
            print("Check the generated results directory for detailed outputs.")
        else:
            print("\n❌ Training and benchmarking failed!")
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()