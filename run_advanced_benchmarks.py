"""
Comprehensive Benchmark Runner for Reversible Qwen3 Models
=========================================================

This script runs your trained models on various advanced benchmarks including:
- BLEU/ROUGE/METEOR for text generation
- GLUE+ tasks for language understanding  
- Memory benchmarks for long-range tasks
- Advanced domain-specific benchmarks

Usage:
python run_advanced_benchmarks.py --models reversible_qwen3,standard_qwen3 --tasks all
"""

import torch
import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import time

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing benchmark frameworks
try:
    from glue_plus_benchmark import GLUEPlusBenchmark, get_recommended_task_progression
    GLUE_AVAILABLE = True
except ImportError as e:
    print(f"GLUE+ benchmark not available: {e}")
    GLUE_AVAILABLE = False

try:
    from memory_benchmark import MemoryBenchmark, get_recommended_memory_task_sequence
    MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"Memory benchmark not available: {e}")
    MEMORY_AVAILABLE = False

try:
    from advanced_benchmarks import AdvancedBenchmarkSuite
    ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"Advanced benchmark not available: {e}")
    ADVANCED_AVAILABLE = False

# Import your model creation functions
try:
    from qwen3_reversible_02_3 import create_reversible_qwen3_model
    MODEL_CREATION_AVAILABLE = True
except ImportError as e:
    print(f"Model creation not available: {e}")
    MODEL_CREATION_AVAILABLE = False

# BLEU/ROUGE metrics
try:
    # Use evaluate instead of deprecated load_metric
    import evaluate
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Evaluation metrics not available: {e}")
    print("Install with: pip install evaluate datasets rouge-score nltk")
    METRICS_AVAILABLE = False

class ComprehensiveBenchmarkRunner:
    """Run all available benchmarks on your models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
        # Initialize benchmark suites
        if GLUE_AVAILABLE:
            self.glue_benchmark = GLUEPlusBenchmark()
        if MEMORY_AVAILABLE:
            self.memory_benchmark = MemoryBenchmark()
        if ADVANCED_AVAILABLE:
            self.advanced_benchmark = AdvancedBenchmarkSuite(device=device)
            
        # Initialize BLEU/ROUGE scorers
        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.smoothing = SmoothingFunction().method1
    
    def load_models_from_checkpoint(self, model_configs: Dict[str, str]) -> Dict[str, torch.nn.Module]:
        """Load models from checkpoint paths or create new ones"""
        models = {}
        
        for model_name, config_or_path in model_configs.items():
            print(f"Loading {model_name}...")
            
            try:
                if os.path.exists(config_or_path):
                    # Load from checkpoint
                    model = torch.load(config_or_path, map_location=self.device)
                else:
                    # Create new model with given config
                    if not MODEL_CREATION_AVAILABLE:
                        print(f"Cannot create model {model_name} - model creation not available")
                        continue
                    
                    # Default model configuration
                    model = create_reversible_qwen3_model(
                        vocab_size=32000,
                        hidden_size=512,
                        num_hidden_layers=4,
                        num_attention_heads=8,
                        num_key_value_heads=4,
                        attention_type="standard",
                        use_reversible="reversible" in model_name.lower(),
                        reverse_thres=256 if "reversible" in model_name.lower() else 999999,
                        intermediate_size=2048,
                        max_position_embeddings=2048,
                        rms_norm_eps=1e-6
                    )
                
                model = model.to(self.device)
                model.eval()
                models[model_name] = model
                print(f"Successfully loaded {model_name}")
                
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue
        
        return models
    
    def run_text_generation_benchmark(self, models: Dict[str, torch.nn.Module]) -> Dict[str, Dict]:
        """Run BLEU/ROUGE text generation benchmarks"""
        if not METRICS_AVAILABLE:
            print("Metrics not available for text generation benchmark")
            return {}
        
        print("\n" + "="*60)
        print("TEXT GENERATION BENCHMARK (BLEU/ROUGE)")
        print("="*60)
        
        # Simple text generation task: complete sentences
        test_prompts = [
            "The weather today is",
            "Artificial intelligence will",
            "In the future, we will",
            "The most important thing in life is",
            "Technology has changed the way we",
            "Education is important because",
            "Climate change affects",
            "The internet has revolutionized",
            "Space exploration helps us",
            "Medical research has led to"
        ]
        
        # Reference completions (simplified)
        references = [
            "sunny and warm with clear skies",
            "transform how we work and live",
            "have better technology and medicine",
            "happiness and good relationships",
            "communicate and share information",
            "it develops critical thinking skills",
            "global weather patterns significantly",
            "communication and information access",
            "understand the universe better",
            "breakthrough treatments for diseases"
        ]
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name} on text generation...")
            
            predictions = []
            generation_times = []
            
            for prompt in test_prompts:
                start_time = time.time()
                
                try:
                    # Simple generation (you may need to adjust this based on your model's interface)
                    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(self.device)  # Dummy tokens
                    
                    with torch.no_grad():
                        # This is a placeholder - replace with your actual generation logic
                        output = model(input_ids)
                        # For now, generate dummy text
                        generated_text = f"generated response for '{prompt}'"
                        
                    predictions.append(generated_text)
                    generation_times.append((time.time() - start_time) * 1000)
                    
                except Exception as e:
                    print(f"Generation failed for {model_name}: {e}")
                    predictions.append("")
                    generation_times.append(0)
            
            # Calculate BLEU scores
            bleu_scores = []
            for pred, ref in zip(predictions, references):
                if pred.strip():
                    try:
                        score = sentence_bleu([ref.split()], pred.split(), 
                                            weights=(0.25, 0.25, 0.25, 0.25), 
                                            smoothing_function=self.smoothing)
                        bleu_scores.append(score)
                    except:
                        bleu_scores.append(0.0)
            
            # Calculate ROUGE scores
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            for pred, ref in zip(predictions, references):
                if pred.strip():
                    try:
                        scores = self.rouge_scorer.score(ref, pred)
                        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
                    except:
                        rouge_scores['rouge1'].append(0.0)
                        rouge_scores['rouge2'].append(0.0)
                        rouge_scores['rougeL'].append(0.0)
            
            results[model_name] = {
                'bleu_4': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
                'rouge_1': sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0,
                'rouge_2': sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0,
                'rouge_l': sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0,
                'avg_generation_time': sum(generation_times) / len(generation_times) if generation_times else 0.0,
                'num_successful_generations': len([p for p in predictions if p.strip()])
            }
            
            print(f"  BLEU-4: {results[model_name]['bleu_4']:.3f}")
            print(f"  ROUGE-1: {results[model_name]['rouge_1']:.3f}")
            print(f"  ROUGE-L: {results[model_name]['rouge_l']:.3f}")
            print(f"  Avg Generation Time: {results[model_name]['avg_generation_time']:.1f}ms")
        
        return results
    
    def run_all_benchmarks(self, models: Dict[str, torch.nn.Module], 
                          benchmark_types: List[str] = None) -> Dict[str, Dict]:
        """Run all available benchmarks"""
        
        if benchmark_types is None:
            benchmark_types = ['text_generation', 'glue', 'memory', 'advanced']
        
        all_results = {}
        
        # 1. Text Generation (BLEU/ROUGE)
        if 'text_generation' in benchmark_types:
            try:
                text_gen_results = self.run_text_generation_benchmark(models)
                all_results['text_generation'] = text_gen_results
            except Exception as e:
                print(f"Text generation benchmark failed: {e}")
        
        # 2. GLUE+ Language Understanding
        if 'glue' in benchmark_types and GLUE_AVAILABLE:
            try:
                print("\n" + "="*60)
                print("GLUE+ LANGUAGE UNDERSTANDING BENCHMARK")
                print("="*60)
                
                # Run basic GLUE tasks
                basic_tasks = ['sst2', 'cola', 'mrpc']
                glue_results = self.glue_benchmark.run_full_benchmark(
                    models, task_subset=basic_tasks
                )
                all_results['glue_plus'] = glue_results
                
                print("\nGLUE+ Results Summary:")
                self.glue_benchmark.print_benchmark_summary()
                
            except Exception as e:
                print(f"GLUE+ benchmark failed: {e}")
        
        # 3. Long-Range Memory Tasks
        if 'memory' in benchmark_types and MEMORY_AVAILABLE:
            try:
                print("\n" + "="*60)
                print("LONG-RANGE MEMORY BENCHMARK")
                print("="*60)
                
                memory_results = self.memory_benchmark.run_memory_benchmark(models)
                all_results['memory'] = memory_results
                
                print("\nMemory Benchmark Results:")
                self.memory_benchmark.print_memory_benchmark_summary()
                
            except Exception as e:
                print(f"Memory benchmark failed: {e}")
        
        # 4. Advanced Domain-Specific Tasks  
        if 'advanced' in benchmark_types and ADVANCED_AVAILABLE:
            try:
                print("\n" + "="*60)
                print("ADVANCED DOMAIN-SPECIFIC BENCHMARK")
                print("="*60)
                
                # Run comprehensive advanced benchmarks
                advanced_results = self.advanced_benchmark.run_comprehensive_benchmark(
                    models, quick_test=True
                )
                all_results['advanced_domain'] = advanced_results
                
                print("\nAdvanced Benchmark Results:")
                for model_name, results in advanced_results.items():
                    print(f"  {model_name}:")
                    if 'long_range' in results:
                        print(f"    Long-range accuracy: {results['long_range'].get('accuracy', 0):.3f}")
                    if 'memory_stress' in results:
                        print(f"    Memory stress passed: {len(results['memory_stress'])}")
                
            except Exception as e:
                print(f"Advanced benchmark failed: {e}")
        
        self.results = all_results
        return all_results
    
    def print_comprehensive_summary(self):
        """Print comprehensive summary of all benchmark results"""
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print("="*80)
        
        # Text Generation Summary
        if 'text_generation' in self.results:
            print("\nText Generation (BLEU/ROUGE) Results:")
            print("-" * 50)
            print(f"{'Model':<25} {'BLEU-4':<8} {'ROUGE-1':<9} {'ROUGE-L':<9} {'Time(ms)':<10}")
            print("-" * 50)
            
            for model_name, results in self.results['text_generation'].items():
                print(f"{model_name:<25} {results['bleu_4']:.3f}    {results['rouge_1']:.3f}     "
                      f"{results['rouge_l']:.3f}     {results['avg_generation_time']:.1f}")
        
        # Overall Performance Comparison
        print("\n" + "="*80)
        print("REVERSIBLE vs STANDARD COMPARISON")
        print("="*80)
        
        reversible_models = []
        standard_models = []
        
        for benchmark_type, benchmark_results in self.results.items():
            for model_name in benchmark_results.keys():
                if 'reversible' in model_name.lower():
                    reversible_models.append(model_name)
                else:
                    standard_models.append(model_name)
        
        reversible_models = list(set(reversible_models))
        standard_models = list(set(standard_models))
        
        print(f"Reversible Models: {reversible_models}")
        print(f"Standard Models: {standard_models}")
        
        if reversible_models and standard_models:
            print("\nPerformance gaps (Reversible - Standard):")
            
            if 'text_generation' in self.results:
                rev_bleu = [self.results['text_generation'][m]['bleu_4'] for m in reversible_models 
                           if m in self.results['text_generation']]
                std_bleu = [self.results['text_generation'][m]['bleu_4'] for m in standard_models 
                           if m in self.results['text_generation']]
                
                if rev_bleu and std_bleu:
                    bleu_gap = sum(rev_bleu)/len(rev_bleu) - sum(std_bleu)/len(std_bleu)
                    print(f"  BLEU-4 Gap: {bleu_gap:+.3f}")
    
    def save_results(self, filename: str = 'comprehensive_benchmark_results.json'):
        """Save all benchmark results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive benchmarks on Qwen3 models')
    parser.add_argument('--models', type=str, default='reversible_qwen3,standard_qwen3',
                        help='Comma-separated list of model names or checkpoint paths')
    parser.add_argument('--tasks', type=str, default='text_generation',
                        help='Comma-separated list of benchmark types: text_generation,glue,memory,advanced')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Parse model configurations
    model_names = [name.strip() for name in args.models.split(',')]
    model_configs = {name: name for name in model_names}  # Use name as config for now
    
    # Parse benchmark types
    benchmark_types = [t.strip() for t in args.tasks.split(',')]
    
    # Initialize benchmark runner
    runner = ComprehensiveBenchmarkRunner(device=args.device)
    
    print("="*80)
    print("COMPREHENSIVE QWEN3 BENCHMARK SUITE")
    print("="*80)
    print(f"Models: {model_names}")
    print(f"Benchmarks: {benchmark_types}")
    print(f"Device: {args.device}")
    
    # Load models
    print("\nLoading models...")
    models = runner.load_models_from_checkpoint(model_configs)
    
    if not models:
        print("No models loaded successfully!")
        return
    
    print(f"Successfully loaded {len(models)} models: {list(models.keys())}")
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    results = runner.run_all_benchmarks(models, benchmark_types)
    
    # Print summary
    runner.print_comprehensive_summary()
    
    # Save results
    runner.save_results(args.output)
    
    print(f"\nBenchmarking complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()
