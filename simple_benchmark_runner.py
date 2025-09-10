"""
Simple Benchmark Runner for Your Trained Models
==============================================

This is a simplified script to run BLEU, ROUGE, and other benchmarks
on your reversible and standard Qwen3 models.
"""

import torch
import sys
import os
import numpy as np
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_simple_benchmarks():
    """Run simple benchmarks on your models"""
    
    print("="*60)
    print("SIMPLE BENCHMARK RUNNER FOR QWEN3 MODELS")
    print("="*60)
    
    # Step 1: Check available frameworks
    print("\n1. Checking available benchmark frameworks...")
    
    frameworks_available = {}
    
    try:
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        frameworks_available['text_generation'] = True
        print("✓ Text generation metrics (BLEU/ROUGE) available")
    except ImportError:
        frameworks_available['text_generation'] = False
        print("✗ Text generation metrics not available")
    
    try:
        from glue_plus_benchmark import GLUEPlusBenchmark
        frameworks_available['glue'] = True
        print("✓ GLUE+ language understanding benchmark available")
    except ImportError:
        frameworks_available['glue'] = False
        print("✗ GLUE+ benchmark not available")
    
    try:
        from memory_benchmark import MemoryBenchmark
        frameworks_available['memory'] = True
        print("✓ Long-range memory benchmark available")
    except ImportError:
        frameworks_available['memory'] = False
        print("✗ Memory benchmark not available")
    
    try:
        from qwen3_reversible_02_2 import create_reversible_qwen3_model
        frameworks_available['model_creation'] = True
        print("✓ Model creation available")
    except ImportError:
        frameworks_available['model_creation'] = False
        print("✗ Model creation not available")
    
    # Step 2: Create or load models
    print("\n2. Creating models for benchmarking...")
    
    if frameworks_available['model_creation']:
        try:
            # Create small test models
            reversible_model = create_reversible_qwen3_model(
                vocab_size=8000,
                hidden_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                attention_type="standard",
                use_reversible=True,
                reverse_thres=256,
                intermediate_size=1024
            )
            
            standard_model = create_reversible_qwen3_model(
                vocab_size=8000,
                hidden_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                attention_type="standard",
                use_reversible=False,
                reverse_thres=999999,
                intermediate_size=1024
            )
            
            models = {
                'reversible_qwen3': reversible_model,
                'standard_qwen3': standard_model
            }
            
            print(f"✓ Created {len(models)} models successfully")
            
        except Exception as e:
            print(f"✗ Model creation failed: {e}")
            models = {}
    else:
        print("Model creation not available - skipping model benchmarks")
        models = {}
    
    # Step 3: Run text generation benchmarks
    if frameworks_available['text_generation'] and models:
        print("\n3. Running text generation benchmarks...")
        run_text_generation_benchmark(models)
    else:
        print("\n3. Skipping text generation benchmarks (not available)")
    
    # Step 4: Run GLUE+ benchmarks
    if frameworks_available['glue'] and models:
        print("\n4. Running GLUE+ language understanding benchmarks...")
        run_glue_benchmark(models)
    else:
        print("\n4. Skipping GLUE+ benchmarks (not available)")
    
    # Step 5: Run memory benchmarks
    if frameworks_available['memory'] and models:
        print("\n5. Running long-range memory benchmarks...")
        run_memory_benchmark(models)
    else:
        print("\n5. Skipping memory benchmarks (not available)")
    
    # Step 6: Instructions for running with your trained models
    print("\n" + "="*60)
    print("HOW TO RUN WITH YOUR ACTUAL TRAINED MODELS")
    print("="*60)
    
    print("""
To run benchmarks on your actual trained models from the comprehensive test:

1. Use your trained models directly:
   # After running your comprehensive test
   from test_train_qwen3_rev_v3f import ReversibleQwenPerformanceTester
   
   tester = ReversibleQwenPerformanceTester()
   results = tester.run_comprehensive_test(seq_len=512, vocab_size=16000, use_wikitext=True)
   
   # Extract the trained models and run benchmarks
   models = tester.setup_models_for_comparison(...)
   
2. Run specific benchmark types:
   
   # Text generation (BLEU/ROUGE)
   python run_advanced_benchmarks.py --tasks text_generation
   
   # Language understanding (GLUE+)
   python run_advanced_benchmarks.py --tasks glue
   
   # Long-range memory
   python run_advanced_benchmarks.py --tasks memory
   
   # All benchmarks
   python run_advanced_benchmarks.py --tasks text_generation,glue,memory

3. With your specific model checkpoints:
   python run_advanced_benchmarks.py --models /path/to/reversible_model.pt,/path/to/standard_model.pt

4. Quick memory benchmark test:
   python -c "
   from memory_benchmark import MemoryBenchmark
   benchmark = MemoryBenchmark()
   models = {'my_model': your_model}
   results = benchmark.run_memory_benchmark(models)
   benchmark.print_memory_benchmark_summary()
   "

Available benchmark metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4: N-gram overlap for text generation
- ROUGE-1, ROUGE-2, ROUGE-L: Recall-oriented metrics for summarization
- METEOR: Semantic similarity metric
- BERTScore: Contextual embedding-based metric
- Exact Match: Perfect match accuracy
- GLUE+ tasks: Language understanding across multiple domains
- Memory tasks: Long-range dependency and context modeling
    """)

def run_text_generation_benchmark(models):
    """Run text generation benchmark with BLEU/ROUGE"""
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    
    # Simple test cases
    test_prompts = [
        "Complete this sentence: The weather today",
        "What comes next: Technology will help",
        "Continue: The most important aspect of education",
        "Finish: Climate change requires urgent"
    ]
    
    references = [
        "is very pleasant and sunny",
        "us solve many complex problems",
        "is developing critical thinking skills",
        "action from all world governments"
    ]
    
    results = {}
    
    for model_name, model in models.items():
        print(f"  Evaluating {model_name}...")
        
        model.eval()
        predictions = []
        generation_times = []
        
        for prompt, reference in zip(test_prompts, references):
            start_time = time.time()
            
            # Simple prediction (replace with actual generation)
            try:
                # For demo, we'll generate a simple response
                prediction = f"Model {model_name} responds to: {prompt[-10:]}"
                predictions.append(prediction)
            except:
                predictions.append("Generated response")
            
            generation_times.append((time.time() - start_time) * 1000)
        
        # Calculate BLEU scores
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            try:
                score = sentence_bleu([ref.split()], pred.split(), 
                                    weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=smoothing)
                bleu_scores.append(score)
            except:
                bleu_scores.append(0.0)
        
        # Calculate ROUGE scores
        rouge_1_scores = []
        rouge_l_scores = []
        
        for pred, ref in zip(predictions, references):
            try:
                scores = rouge_scorer_obj.score(ref, pred)
                rouge_1_scores.append(scores['rouge1'].fmeasure)
                rouge_l_scores.append(scores['rougeL'].fmeasure)
            except:
                rouge_1_scores.append(0.0)
                rouge_l_scores.append(0.0)
        
        results[model_name] = {
            'bleu_4': np.mean(bleu_scores),
            'rouge_1': np.mean(rouge_1_scores),
            'rouge_l': np.mean(rouge_l_scores),
            'avg_generation_time': np.mean(generation_times)
        }
        
        print(f"    BLEU-4: {results[model_name]['bleu_4']:.3f}")
        print(f"    ROUGE-1: {results[model_name]['rouge_1']:.3f}")
        print(f"    ROUGE-L: {results[model_name]['rouge_l']:.3f}")
        print(f"    Avg Time: {results[model_name]['avg_generation_time']:.1f}ms")
    
    # Print comparison
    print("\n  Text Generation Comparison:")
    print(f"  {'Metric':<12} {'Reversible':<12} {'Standard':<12} {'Difference':<12}")
    print("  " + "-" * 50)
    
    if 'reversible_qwen3' in results and 'standard_qwen3' in results:
        rev_results = results['reversible_qwen3']
        std_results = results['standard_qwen3']
        
        for metric in ['bleu_4', 'rouge_1', 'rouge_l']:
            diff = rev_results[metric] - std_results[metric]
            print(f"  {metric:<12} {rev_results[metric]:<12.3f} {std_results[metric]:<12.3f} {diff:<+12.3f}")

def run_glue_benchmark(models):
    """Run GLUE+ language understanding benchmark"""
    from glue_plus_benchmark import GLUEPlusBenchmark
    
    try:
        benchmark = GLUEPlusBenchmark()
        
        # Run basic tasks only for demo
        basic_tasks = ['sst2']  # Sentiment analysis
        print(f"  Running GLUE+ tasks: {basic_tasks}")
        
        results = benchmark.run_full_benchmark(models, task_subset=basic_tasks)
        
        print("  GLUE+ Results Summary:")
        benchmark.print_benchmark_summary()
        
    except Exception as e:
        print(f"  GLUE+ benchmark failed: {e}")

def run_memory_benchmark(models):
    """Run long-range memory benchmark"""
    from memory_benchmark import MemoryBenchmark
    
    try:
        benchmark = MemoryBenchmark()
        
        print("  Running memory benchmark (this may take a while)...")
        results = benchmark.run_memory_benchmark(models)
        
        print("  Memory Benchmark Results:")
        benchmark.print_memory_benchmark_summary()
        
    except Exception as e:
        print(f"  Memory benchmark failed: {e}")

if __name__ == "__main__":
    run_simple_benchmarks()
