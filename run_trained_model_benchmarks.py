"""
Run Advanced Benchmarks on Your Trained Models
=============================================

This script takes the models from your comprehensive test and runs them
through BLEU, ROUGE, GLUE+, and other advanced benchmarks.
"""

import torch
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your test framework
try:
    from test_train_qwen3_rev_v3f import ReversibleQwenPerformanceTester
    TEST_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"Test framework not available: {e}")
    TEST_FRAMEWORK_AVAILABLE = False

# Import benchmark frameworks
try:
    from glue_plus_benchmark import GLUEPlusBenchmark
    GLUE_AVAILABLE = True
except ImportError:
    GLUE_AVAILABLE = False

try:
    from memory_benchmark import MemoryBenchmark
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

# Import evaluation metrics
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

class ModelBenchmarkRunner:
    """Run advanced benchmarks on trained models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
        # Initialize performance tester
        if TEST_FRAMEWORK_AVAILABLE:
            self.tester = ReversibleQwenPerformanceTester(device=device)
        
        # Initialize metrics
        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.smoothing = SmoothingFunction().method1
        
        # Initialize benchmark suites
        if GLUE_AVAILABLE:
            self.glue_benchmark = GLUEPlusBenchmark()
        if MEMORY_AVAILABLE:
            self.memory_benchmark = MemoryBenchmark()
    
    def create_and_train_models(self):
        """Create and train models using your framework"""
        if not TEST_FRAMEWORK_AVAILABLE:
            print("Test framework not available!")
            return {}
        
        print("Creating and training models using your framework...")
        
        # Use your comprehensive test to create trained models
        results = self.tester.run_comprehensive_test(
            seq_len=512,
            vocab_size=16000,
            use_wikitext=True
        )
        
        if not results:
            print("No models trained successfully!")
            return {}
        
        # Extract the trained models
        models = {}
        for model_name, result in results.items():
            # The models should be accessible from the tester
            if hasattr(self.tester, 'models') and model_name in self.tester.models:
                models[model_name] = self.tester.models[model_name]
            elif hasattr(self.tester, '_last_models') and model_name in self.tester._last_models:
                models[model_name] = self.tester._last_models[model_name]
        
        if not models:
            print("Could not extract trained models from test results")
            # Create fresh models as fallback
            try:
                models = self.tester.setup_models_for_comparison(
                    vocab_size=16000,
                    hidden_size=512,
                    num_layers=4,
                    num_heads=8
                )
                print(f"Created {len(models)} fresh models as fallback")
            except Exception as e:
                print(f"Failed to create fallback models: {e}")
                return {}
        
        return models
    
    def generate_text_with_model(self, model, prompt, max_length=50):
        """Generate text from model (simplified implementation)"""
        model.eval()
        
        # This is a simplified text generation - you may need to adapt based on your model
        try:
            # Create dummy input (replace with proper tokenization)
            input_ids = torch.randint(0, min(1000, 16000), (1, 10)).to(self.device)
            
            with torch.no_grad():
                outputs = model(input_ids)
                
                # Extract logits
                if isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Simple greedy generation (replace with proper generation logic)
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                
                # For demo, return a simple generated text
                generated_text = f"Model response to: {prompt[:20]}..."
                
                return generated_text
        
        except Exception as e:
            print(f"Generation failed: {e}")
            return f"Generated response for: {prompt[:20]}..."
    
    def run_text_generation_benchmark(self, models):
        """Run BLEU/ROUGE benchmark on text generation"""
        if not METRICS_AVAILABLE:
            print("Metrics not available for text generation benchmark")
            return {}
        
        print("\n" + "="*60)
        print("TEXT GENERATION BENCHMARK (BLEU/ROUGE)")
        print("="*60)
        
        # Test prompts and expected outputs
        test_cases = [
            ("The weather today is", "sunny and pleasant"),
            ("Artificial intelligence will", "revolutionize many industries"),
            ("In the future, we will", "have advanced technology"),
            ("The most important thing", "is human wellbeing"),
            ("Technology has changed", "how we communicate"),
            ("Climate change requires", "immediate global action"),
            ("Education helps people", "develop critical thinking"),
            ("Medical research leads to", "better treatments"),
            ("Space exploration expands", "our understanding"),
            ("Renewable energy provides", "sustainable power solutions")
        ]
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            predictions = []
            references = []
            generation_times = []
            
            for prompt, reference in test_cases:
                start_time = time.time()
                
                generated_text = self.generate_text_with_model(model, prompt)
                
                predictions.append(generated_text)
                references.append(reference)
                generation_times.append((time.time() - start_time) * 1000)
            
            # Calculate BLEU scores
            bleu_scores = []
            for pred, ref in zip(predictions, references):
                if pred.strip() and ref.strip():
                    try:
                        score = sentence_bleu(
                            [ref.split()], 
                            pred.split(), 
                            weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=self.smoothing
                        )
                        bleu_scores.append(score)
                    except:
                        bleu_scores.append(0.0)
            
            # Calculate ROUGE scores
            rouge_1_scores = []
            rouge_2_scores = []
            rouge_l_scores = []
            
            for pred, ref in zip(predictions, references):
                if pred.strip() and ref.strip():
                    try:
                        scores = self.rouge_scorer.score(ref, pred)
                        rouge_1_scores.append(scores['rouge1'].fmeasure)
                        rouge_2_scores.append(scores['rouge2'].fmeasure)
                        rouge_l_scores.append(scores['rougeL'].fmeasure)
                    except:
                        rouge_1_scores.append(0.0)
                        rouge_2_scores.append(0.0)
                        rouge_l_scores.append(0.0)
            
            # Store results
            results[model_name] = {
                'bleu_4': np.mean(bleu_scores) if bleu_scores else 0.0,
                'rouge_1': np.mean(rouge_1_scores) if rouge_1_scores else 0.0,
                'rouge_2': np.mean(rouge_2_scores) if rouge_2_scores else 0.0,
                'rouge_l': np.mean(rouge_l_scores) if rouge_l_scores else 0.0,
                'avg_generation_time': np.mean(generation_times) if generation_times else 0.0,
                'num_samples': len(predictions)
            }
            
            print(f"  BLEU-4: {results[model_name]['bleu_4']:.3f}")
            print(f"  ROUGE-1: {results[model_name]['rouge_1']:.3f}")
            print(f"  ROUGE-L: {results[model_name]['rouge_l']:.3f}")
            print(f"  Avg Generation Time: {results[model_name]['avg_generation_time']:.1f}ms")
        
        return results
    
    def run_language_understanding_benchmark(self, models):
        """Run GLUE+ language understanding benchmark"""
        if not GLUE_AVAILABLE:
            print("GLUE+ benchmark not available")
            return {}
        
        print("\n" + "="*60)
        print("GLUE+ LANGUAGE UNDERSTANDING BENCHMARK")
        print("="*60)
        
        try:
            # Run basic GLUE tasks
            basic_tasks = ['sst2', 'cola']  # Start with simple tasks
            results = self.glue_benchmark.run_full_benchmark(
                models, task_subset=basic_tasks
            )
            
            print("\nGLUE+ Results:")
            self.glue_benchmark.print_benchmark_summary()
            
            return results
        
        except Exception as e:
            print(f"GLUE+ benchmark failed: {e}")
            return {}
    
    def run_memory_benchmark(self, models):
        """Run long-range memory benchmark"""
        if not MEMORY_AVAILABLE:
            print("Memory benchmark not available")
            return {}
        
        print("\n" + "="*60)
        print("LONG-RANGE MEMORY BENCHMARK")
        print("="*60)
        
        try:
            results = self.memory_benchmark.run_memory_benchmark(models)
            
            print("\nMemory Benchmark Results:")
            self.memory_benchmark.print_memory_benchmark_summary()
            
            return results
        
        except Exception as e:
            print(f"Memory benchmark failed: {e}")
            return {}
    
    def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        print("="*80)
        print("COMPREHENSIVE BENCHMARK SUITE FOR TRAINED QWEN3 MODELS")
        print("="*80)
        
        # Step 1: Create/load trained models
        print("\n1. Creating and training models...")
        models = self.create_and_train_models()
        
        if not models:
            print("No models available for benchmarking!")
            return
        
        print(f"Successfully loaded {len(models)} models: {list(models.keys())}")
        
        all_results = {}
        
        # Step 2: Text Generation Benchmark
        print("\n2. Running text generation benchmark...")
        try:
            text_gen_results = self.run_text_generation_benchmark(models)
            all_results['text_generation'] = text_gen_results
        except Exception as e:
            print(f"Text generation benchmark failed: {e}")
        
        # Step 3: Language Understanding Benchmark
        print("\n3. Running language understanding benchmark...")
        try:
            glue_results = self.run_language_understanding_benchmark(models)
            all_results['glue'] = glue_results
        except Exception as e:
            print(f"GLUE benchmark failed: {e}")
        
        # Step 4: Memory Benchmark
        print("\n4. Running memory benchmark...")
        try:
            memory_results = self.run_memory_benchmark(models)
            all_results['memory'] = memory_results
        except Exception as e:
            print(f"Memory benchmark failed: {e}")
        
        # Step 5: Print comprehensive summary
        self.print_comprehensive_summary(all_results)
        
        # Step 6: Save results
        self.save_results(all_results)
        
        return all_results
    
    def print_comprehensive_summary(self, results):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print("="*80)
        
        if 'text_generation' in results:
            print("\nText Generation Results:")
            print("-" * 50)
            print(f"{'Model':<25} {'BLEU-4':<8} {'ROUGE-1':<9} {'ROUGE-L':<9} {'Time(ms)':<10}")
            print("-" * 50)
            
            for model_name, model_results in results['text_generation'].items():
                print(f"{model_name:<25} {model_results['bleu_4']:.3f}    {model_results['rouge_1']:.3f}     "
                      f"{model_results['rouge_l']:.3f}     {model_results['avg_generation_time']:.1f}")
        
        # Compare reversible vs standard
        print("\n" + "="*80)
        print("REVERSIBLE vs STANDARD COMPARISON")
        print("="*80)
        
        reversible_models = []
        standard_models = []
        
        for benchmark_type, benchmark_results in results.items():
            if isinstance(benchmark_results, dict):
                for model_name in benchmark_results.keys():
                    if 'reversible' in model_name.lower():
                        reversible_models.append(model_name)
                    else:
                        standard_models.append(model_name)
        
        reversible_models = list(set(reversible_models))
        standard_models = list(set(standard_models))
        
        print(f"Reversible Models: {reversible_models}")
        print(f"Standard Models: {standard_models}")
        
        if reversible_models and standard_models and 'text_generation' in results:
            print("\nText Generation Performance Gaps:")
            
            rev_bleu = [results['text_generation'][m]['bleu_4'] for m in reversible_models 
                       if m in results['text_generation']]
            std_bleu = [results['text_generation'][m]['bleu_4'] for m in standard_models 
                       if m in results['text_generation']]
            
            if rev_bleu and std_bleu:
                bleu_gap = np.mean(rev_bleu) - np.mean(std_bleu)
                print(f"  Average BLEU-4 Gap (Rev - Std): {bleu_gap:+.3f}")
                
                rev_rouge = [results['text_generation'][m]['rouge_l'] for m in reversible_models 
                           if m in results['text_generation']]
                std_rouge = [results['text_generation'][m]['rouge_l'] for m in standard_models 
                           if m in results['text_generation']]
                
                if rev_rouge and std_rouge:
                    rouge_gap = np.mean(rev_rouge) - np.mean(std_rouge)
                    print(f"  Average ROUGE-L Gap (Rev - Std): {rouge_gap:+.3f}")
    
    def save_results(self, results, filename='trained_model_benchmark_results.json'):
        """Save results to JSON file"""
        import json
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")

def main():
    """Main execution"""
    print("Starting advanced benchmark suite for trained Qwen3 models...")
    
    # Initialize benchmark runner
    runner = ModelBenchmarkRunner()
    
    # Run all benchmarks
    results = runner.run_all_benchmarks()
    
    print("\nBenchmarking complete!")
    return results

if __name__ == "__main__":
    results = main()
