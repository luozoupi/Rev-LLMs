"""
Quick Benchmark Demo for Your Trained Models
===========================================

This script shows how to run BLEU, ROUGE, and other advanced benchmarks
on your trained reversible and standard Qwen3 models.
"""

import torch
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def install_required_packages():
    """Install required packages for advanced benchmarks"""
    import subprocess
    
    packages = [
        'evaluate',
        'datasets', 
        'rouge-score',
        'nltk',
        'bert-score',
        'sacrebleu'  # Alternative BLEU implementation
    ]
    
    print("Installing required packages for advanced benchmarks...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def quick_benchmark_demo():
    """Quick demonstration of advanced benchmarks"""
    
    print("="*60)
    print("QUICK BENCHMARK DEMO FOR QWEN3 MODELS")
    print("="*60)
    
    # First, let's install required packages
    try:
        import evaluate
        import nltk
        from rouge_score import rouge_scorer
        print("✓ All required packages available")
    except ImportError:
        print("Installing required packages...")
        install_required_packages()
        print("Please restart the script after installation.")
        return
    
    # Try to load your models
    print("\nLoading models...")
    try:
        from qwen3_reversible_02_2 import create_reversible_qwen3_model
        
        # Create small test models (adjust parameters as needed)
        print("Creating reversible model...")
        reversible_model = create_reversible_qwen3_model(
            vocab_size=10000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_type="standard",
            use_reversible=True,
            reverse_thres=256,
            intermediate_size=1024,
            max_position_embeddings=1024
        )
        
        print("Creating standard model...")
        standard_model = create_reversible_qwen3_model(
            vocab_size=10000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_type="standard",
            use_reversible=False,
            reverse_thres=999999,
            intermediate_size=1024,
            max_position_embeddings=1024
        )
        
        models = {
            'reversible_qwen3': reversible_model,
            'standard_qwen3': standard_model
        }
        
        print(f"✓ Successfully created {len(models)} models")
        
    except Exception as e:
        print(f"✗ Failed to create models: {e}")
        print("Using dummy models for demonstration...")
        
        # Create dummy models for demonstration
        class DummyModel(torch.nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name
                self.linear = torch.nn.Linear(10, 1000)
            
            def forward(self, x):
                return {'logits': self.linear(x.float())}
        
        models = {
            'reversible_qwen3': DummyModel('reversible'),
            'standard_qwen3': DummyModel('standard')
        }
    
    # Initialize benchmark metrics
    print("\nInitializing benchmark metrics...")
    try:
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        smoothing = SmoothingFunction().method1
        
        print("✓ ROUGE and BLEU metrics initialized")
    except Exception as e:
        print(f"✗ Failed to initialize metrics: {e}")
        return
    
    # Prepare test data
    print("\nPreparing test data...")
    test_texts = [
        ("The weather today is sunny", "It is a bright and sunny day"),
        ("AI will change the world", "Artificial intelligence will transform society"),
        ("Learning is important", "Education plays a crucial role in development"),
        ("Technology advances rapidly", "Tech innovation progresses at high speed"),
        ("Climate change affects everyone", "Global warming impacts all populations")
    ]
    
    # Run benchmark on each model
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Benchmarking {model_name}")
        print(f"{'='*40}")
        
        model.eval()
        
        # Simulate text generation (replace with actual generation logic)
        predictions = []
        references = []
        generation_times = []
        
        for prompt, reference in test_texts:
            start_time = time.time()
            
            # For demonstration, we'll use the reference as prediction with slight modification
            # In practice, you would use your model's actual generation capability
            try:
                # Dummy generation - replace with actual model generation
                prediction = f"Generated: {reference.lower()}"
                predictions.append(prediction)
                references.append(reference)
                
                generation_time = (time.time() - start_time) * 1000  # ms
                generation_times.append(generation_time)
                
            except Exception as e:
                print(f"Generation failed: {e}")
                predictions.append("")
                references.append(reference)
                generation_times.append(0)
        
        # Calculate BLEU scores
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            if pred.strip() and ref.strip():
                try:
                    score = sentence_bleu(
                        [ref.split()], 
                        pred.split(), 
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothing
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
                    scores = rouge_scorer_obj.score(ref, pred)
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
        
        # Print individual model results
        print(f"BLEU-4: {results[model_name]['bleu_4']:.3f}")
        print(f"ROUGE-1: {results[model_name]['rouge_1']:.3f}")
        print(f"ROUGE-2: {results[model_name]['rouge_2']:.3f}")
        print(f"ROUGE-L: {results[model_name]['rouge_l']:.3f}")
        print(f"Avg Generation Time: {results[model_name]['avg_generation_time']:.1f}ms")
        print(f"Samples: {results[model_name]['num_samples']}")
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("BENCHMARK COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Metric':<15} {'Reversible':<12} {'Standard':<12} {'Difference':<12}")
    print("-" * 60)
    
    rev_results = results.get('reversible_qwen3', {})
    std_results = results.get('standard_qwen3', {})
    
    metrics = ['bleu_4', 'rouge_1', 'rouge_2', 'rouge_l', 'avg_generation_time']
    for metric in metrics:
        rev_val = rev_results.get(metric, 0.0)
        std_val = std_results.get(metric, 0.0)
        diff = rev_val - std_val
        
        print(f"{metric:<15} {rev_val:<12.3f} {std_val:<12.3f} {diff:<+12.3f}")
    
    print(f"\n{'='*60}")
    print("HOW TO RUN FULL BENCHMARKS")
    print(f"{'='*60}")
    
    print("""
To run comprehensive benchmarks on your trained models:

1. Basic BLEU/ROUGE benchmark:
   python run_advanced_benchmarks.py --models reversible_qwen3,standard_qwen3 --tasks text_generation

2. GLUE+ language understanding:
   python run_advanced_benchmarks.py --tasks glue

3. Long-range memory tasks:
   python run_advanced_benchmarks.py --tasks memory

4. All benchmarks:
   python run_advanced_benchmarks.py --tasks text_generation,glue,memory

5. With custom model checkpoints:
   python run_advanced_benchmarks.py --models /path/to/model1.pt,/path/to/model2.pt --tasks all

Available benchmark types:
- text_generation: BLEU, ROUGE, METEOR scores
- glue: GLUE+ language understanding tasks  
- memory: Long-range dependency and memory tasks
- advanced: Domain-specific benchmarks

For more sophisticated evaluation:
- Install additional packages: pip install evaluate datasets rouge-score nltk bert-score
- Use the GLUE+ framework for comprehensive language understanding
- Use the Memory benchmark for long-context evaluation
    """)
    
    return results

if __name__ == "__main__":
    results = quick_benchmark_demo()
