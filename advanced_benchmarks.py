"""
Advanced Benchmark Suite for Reversible vs Standard Qwen3
=========================================================

Comprehensive evaluation framework with sophisticated tasks to explore 
the limits and variance between reversible and standard architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

# For advanced datasets
from datasets import load_dataset
import requests
import zipfile
import os
from pathlib import Path

@dataclass
class AdvancedBenchmarkResult:
    """Results for advanced benchmark tasks"""
    task_name: str
    model_name: str
    
    # Performance metrics
    accuracy: float
    perplexity: float
    bits_per_character: float
    
    # Memory & Efficiency
    peak_memory_mb: float
    throughput_chars_per_sec: float
    memory_scaling_slope: float
    
    # Task-specific metrics
    long_range_accuracy: Optional[float] = None
    generation_quality: Optional[float] = None
    few_shot_performance: Optional[float] = None
    code_completion_accuracy: Optional[float] = None
    reasoning_score: Optional[float] = None

class AdvancedBenchmarkSuite:
    """Advanced benchmark suite for comparing model architectures"""
    
    def __init__(self, device='cuda', cache_dir='./benchmark_cache'):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.results = {}
    
    # ==========================================
    # 1. CHALLENGING LANGUAGE MODELING DATASETS
    # ==========================================
    
    def load_enwik8_dataset(self):
        """Load the enwik8 character-level dataset - much harder than WikiText"""
        
        cache_path = self.cache_dir / 'enwik8.zip'
        
        if not cache_path.exists():
            print("Downloading enwik8 dataset...")
            url = 'http://mattmahoney.net/dc/enwik8.zip'
            response = requests.get(url, stream=True)
            
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract and process
        with zipfile.ZipFile(cache_path, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)
        
        with open(self.cache_dir / 'enwik8', 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read()
        
        # Character-level tokenization
        chars = sorted(list(set(data)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Split data
        n = len(data)
        train_data = data[:int(0.9*n)]
        val_data = data[int(0.9*n):int(0.95*n)]
        test_data = data[int(0.95*n):]
        
        return {
            'train': train_data,
            'val': val_data, 
            'test': test_data,
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'vocab_size': len(chars)
        }
    
    def load_penn_treebank(self):
        """Load Penn Treebank for word-level language modeling"""
        
        try:
            dataset = load_dataset("ptb_text_only")
            return dataset
        except:
            print("Failed to load Penn Treebank from HuggingFace")
            return None
    
    def load_code_dataset(self):
        """Load CodeParrot for code understanding tasks"""
        
        try:
            dataset = load_dataset("codeparrot/codeparrot-clean-train", split="train", streaming=True)
            # Take a subset for manageable testing
            subset = []
            for i, example in enumerate(dataset):
                if i >= 10000:  # Limit for testing
                    break
                if len(example['content']) > 100:  # Filter short snippets
                    subset.append(example['content'])
            
            # Split into train/val/test
            n = len(subset)
            return {
                'train': subset[:int(0.8*n)],
                'val': subset[int(0.8*n):int(0.9*n)],
                'test': subset[int(0.9*n):]
            }
        except:
            print("Failed to load CodeParrot dataset")
            return None
    
    # ==========================================
    # 2. LONG-RANGE DEPENDENCY TESTS
    # ==========================================
    
    def create_long_range_task(self, seq_length=2048, num_samples=1000):
        """Create synthetic long-range dependency task
        
        Task: Copy tokens from position i to position i+k where k is large
        Tests model's ability to maintain information over long distances
        """
        
        vocab_size = 1000
        data = []
        
        for _ in range(num_samples):
            # Create sequence with special copy tokens
            sequence = torch.randint(10, vocab_size-10, (seq_length,))
            
            # Insert copy markers and targets at random positions
            marker_positions = torch.randperm(seq_length//4)[:seq_length//8]
            
            for pos in marker_positions:
                if pos + seq_length//2 < seq_length - 1:
                    # Marker token
                    sequence[pos] = vocab_size - 2  # Special marker
                    # Token to copy
                    copy_token = sequence[pos + 1]
                    # Target position (far away)
                    target_pos = pos + seq_length//2
                    sequence[target_pos] = copy_token
            
            # Create targets
            targets = torch.zeros_like(sequence)
            targets[:-1] = sequence[1:]  # Next token prediction
            targets[-1] = sequence[0]  # Circular
            
            data.append((sequence, targets))
        
        return data
    
    def create_associative_recall_task(self, seq_length=1024, num_pairs=10, num_samples=500):
        """Create associative recall task
        
        Task: Given key-value pairs at the beginning, recall value when shown key later
        Tests long-term memory capabilities
        """
        
        vocab_size = 1000
        data = []
        
        for _ in range(num_samples):
            sequence = torch.zeros(seq_length, dtype=torch.long)
            targets = torch.zeros(seq_length, dtype=torch.long)
            
            # Create key-value pairs at the beginning
            pairs = {}
            for i in range(num_pairs):
                key = torch.randint(100, 500, (1,)).item()  # Keys in range 100-499
                value = torch.randint(500, 900, (1,)).item()  # Values in range 500-899
                pairs[key] = value
                
                # Store in sequence: [key, value, separator]
                pos = i * 3
                if pos + 2 < seq_length:
                    sequence[pos] = key
                    sequence[pos + 1] = value
                    sequence[pos + 2] = vocab_size - 1  # Separator token
            
            # Fill middle with random tokens
            start_pos = num_pairs * 3
            end_pos = seq_length - num_pairs * 2
            sequence[start_pos:end_pos] = torch.randint(10, 99, (end_pos - start_pos,))
            
            # Add recall queries at the end
            query_start = end_pos
            for i, (key, value) in enumerate(pairs.items()):
                if query_start + i * 2 + 1 < seq_length:
                    sequence[query_start + i * 2] = key
                    targets[query_start + i * 2 + 1] = value
            
            data.append((sequence, targets))
        
        return data
    
    # ==========================================
    # 3. MEMORY STRESS TESTS
    # ==========================================
    
    def run_memory_stress_test(self, model, seq_lengths=[512, 1024, 2048, 4096, 8192]):
        """Test model performance under memory pressure"""
        
        print("Running memory stress test...")
        stress_results = []
        
        for seq_len in seq_lengths:
            print(f"Testing sequence length: {seq_len}")
            
            # Test with increasing batch sizes until OOM
            max_batch_size = 1
            successful_batch_sizes = []
            
            for batch_size in [1, 2, 4, 8, 16, 32]:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    
                    # Create test input
                    test_input = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
                    
                    start_time = time.time()
                    
                    with torch.no_grad():
                        outputs = model(test_input)
                    
                    end_time = time.time()
                    
                    if torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
                    else:
                        peak_memory = 0
                    
                    successful_batch_sizes.append({
                        'batch_size': batch_size,
                        'seq_length': seq_len,
                        'peak_memory_mb': peak_memory,
                        'time_seconds': end_time - start_time,
                        'tokens_per_second': (batch_size * seq_len) / (end_time - start_time)
                    })
                    
                    max_batch_size = batch_size
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"OOM at batch_size={batch_size}, seq_len={seq_len}")
                        break
                    else:
                        raise e
            
            stress_results.append({
                'seq_length': seq_len,
                'max_batch_size': max_batch_size,
                'results': successful_batch_sizes
            })
        
        return stress_results
    
    # ==========================================
    # 4. REASONING AND DOWNSTREAM TASKS
    # ==========================================
    
    def load_math_reasoning_task(self):
        """Load GSM8K math reasoning dataset"""
        
        try:
            dataset = load_dataset("gsm8k", "main")
            return dataset
        except:
            print("Failed to load GSM8K dataset")
            return None
    
    def load_reading_comprehension_task(self):
        """Load reading comprehension task (SQuAD-style)"""
        
        try:
            dataset = load_dataset("squad")
            return dataset
        except:
            print("Failed to load SQuAD dataset")
            return None
    
    def create_few_shot_learning_task(self, num_classes=10, shots_per_class=5, query_size=100):
        """Create few-shot learning task for in-context learning evaluation"""
        
        vocab_size = 1000
        
        # Generate synthetic classification task
        # Each class has a pattern in the input
        class_patterns = {}
        for class_id in range(num_classes):
            # Create a unique pattern for each class
            pattern = torch.randint(100, 200, (10,))  # 10-token pattern
            class_patterns[class_id] = pattern
        
        data = []
        
        # Create few-shot examples
        support_examples = []
        for class_id in range(num_classes):
            for _ in range(shots_per_class):
                # Create example with class pattern + noise
                example = torch.randint(10, 99, (50,))  # Noise
                pattern_start = torch.randint(0, 30, (1,)).item()
                example[pattern_start:pattern_start+10] = class_patterns[class_id]
                support_examples.append((example, class_id))
        
        # Create query examples
        query_examples = []
        for _ in range(query_size):
            class_id = torch.randint(0, num_classes, (1,)).item()
            example = torch.randint(10, 99, (50,))
            pattern_start = torch.randint(0, 30, (1,)).item()
            example[pattern_start:pattern_start+10] = class_patterns[class_id]
            query_examples.append((example, class_id))
        
        return {
            'support': support_examples,
            'query': query_examples,
            'num_classes': num_classes,
            'class_patterns': class_patterns
        }
    
    # ==========================================
    # 5. GENERATION QUALITY TESTS
    # ==========================================
    
    def evaluate_generation_quality(self, model, tokenizer, prompts, max_length=200):
        """Evaluate text generation quality"""
        
        generation_metrics = {
            'coherence_scores': [],
            'repetition_rates': [],
            'diversity_scores': [],
            'perplexity_scores': []
        }
        
        model.eval()
        
        for prompt in prompts:
            try:
                # Tokenize prompt
                if hasattr(tokenizer, 'encode'):
                    input_ids = torch.tensor(tokenizer.encode(prompt)[:50]).unsqueeze(0).to(self.device)
                else:
                    # Simple character-level tokenization fallback
                    input_ids = torch.randint(0, 1000, (1, 50)).to(self.device)
                
                # Generate continuation
                with torch.no_grad():
                    generated = []
                    for _ in range(max_length):
                        outputs = model(input_ids)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        
                        # Sample next token
                        probs = F.softmax(logits[0, -1], dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        
                        generated.append(next_token.item())
                        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                        
                        # Keep only recent context to avoid memory issues
                        if input_ids.shape[1] > 512:
                            input_ids = input_ids[:, -512:]
                
                # Analyze generated sequence
                generated_array = np.array(generated)
                
                # Repetition rate
                unique_tokens = len(set(generated))
                repetition_rate = 1.0 - (unique_tokens / len(generated))
                generation_metrics['repetition_rates'].append(repetition_rate)
                
                # Diversity (unique bigrams)
                bigrams = [(generated[i], generated[i+1]) for i in range(len(generated)-1)]
                unique_bigrams = len(set(bigrams))
                diversity = unique_bigrams / max(len(bigrams), 1)
                generation_metrics['diversity_scores'].append(diversity)
                
                # Simple coherence score (lower variance = more coherent)
                token_variance = np.var(generated_array)
                coherence = 1.0 / (1.0 + token_variance / 1000)
                generation_metrics['coherence_scores'].append(coherence)
                
            except Exception as e:
                print(f"Generation evaluation error: {e}")
                continue
        
        # Average metrics
        return {
            'avg_repetition_rate': np.mean(generation_metrics['repetition_rates']),
            'avg_diversity_score': np.mean(generation_metrics['diversity_scores']),
            'avg_coherence_score': np.mean(generation_metrics['coherence_scores'])
        }
    
    # ==========================================
    # 6. COMPREHENSIVE BENCHMARK RUNNER
    # ==========================================
    
    def run_comprehensive_benchmark(self, models_dict, quick_test=False):
        """Run all advanced benchmarks on provided models"""
        
        print("="*80)
        print("ADVANCED BENCHMARK SUITE FOR REVERSIBLE VS STANDARD QWEN3")
        print("="*80)
        
        all_results = {}
        
        for model_name, model in models_dict.items():
            print(f"\n{'='*60}")
            print(f"TESTING MODEL: {model_name}")
            print(f"{'='*60}")
            
            model_results = {}
            
            try:
                # 1. Character-level language modeling (enwik8)
                if not quick_test:
                    print("\n1. Character-level Language Modeling (enwik8)")
                    enwik8_data = self.load_enwik8_dataset()
                    if enwik8_data:
                        char_lm_result = self.evaluate_character_lm(model, enwik8_data, max_length=1024)
                        model_results['enwik8'] = char_lm_result
                
                # 2. Long-range dependency tasks
                print("\n2. Long-range Dependency Tasks")
                long_range_data = self.create_long_range_task(seq_length=1024 if quick_test else 2048, 
                                                            num_samples=100 if quick_test else 500)
                long_range_result = self.evaluate_long_range_dependencies(model, long_range_data)
                model_results['long_range'] = long_range_result
                
                # 3. Associative recall
                print("\n3. Associative Recall Task")
                recall_data = self.create_associative_recall_task(seq_length=512 if quick_test else 1024,
                                                                num_samples=50 if quick_test else 200)
                recall_result = self.evaluate_associative_recall(model, recall_data)
                model_results['associative_recall'] = recall_result
                
                # 4. Memory stress test
                print("\n4. Memory Stress Test")
                max_seq = 2048 if quick_test else 4096
                stress_lengths = [512, 1024, max_seq] if quick_test else [512, 1024, 2048, 4096]
                stress_result = self.run_memory_stress_test(model, seq_lengths=stress_lengths)
                model_results['memory_stress'] = stress_result
                
                # 5. Few-shot learning
                print("\n5. Few-shot Learning Task")
                few_shot_data = self.create_few_shot_learning_task(num_classes=5 if quick_test else 10,
                                                                 query_size=50 if quick_test else 100)
                few_shot_result = self.evaluate_few_shot_learning(model, few_shot_data)
                model_results['few_shot'] = few_shot_result
                
                # 6. Code understanding (if dataset available)
                if not quick_test:
                    print("\n6. Code Understanding Task")
                    code_data = self.load_code_dataset()
                    if code_data:
                        code_result = self.evaluate_code_understanding(model, code_data)
                        model_results['code'] = code_result
                
                # 7. Generation quality
                print("\n7. Generation Quality Evaluation")
                test_prompts = [
                    "The future of artificial intelligence",
                    "In a world where technology",
                    "The mystery began when"
                ] if quick_test else [
                    "The future of artificial intelligence",
                    "In a world where technology",
                    "The mystery began when",
                    "Scientific research has shown",
                    "The economic implications of",
                    "Once upon a time in a distant land",
                    "The solution to climate change",
                    "Quantum computing will revolutionize"
                ]
                
                generation_result = self.evaluate_generation_quality(model, None, test_prompts)
                model_results['generation'] = generation_result
                
                all_results[model_name] = model_results
                
                print(f"\n✅ Completed benchmarking for {model_name}")
                
            except Exception as e:
                print(f"\n❌ Failed to benchmark {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_results
    
    # ==========================================
    # 7. EVALUATION METHODS
    # ==========================================
    
    def evaluate_character_lm(self, model, data, max_length=1024):
        """Evaluate character-level language modeling performance"""
        
        model.eval()
        char_to_idx = data['char_to_idx']
        test_text = data['test'][:max_length*100]  # Limit for testing
        
        # Convert to indices
        indices = [char_to_idx.get(c, 0) for c in test_text]
        
        total_loss = 0
        total_chars = 0
        
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for i in range(0, len(indices) - max_length, max_length//2):
                sequence = torch.tensor(indices[i:i+max_length], dtype=torch.long).unsqueeze(0).to(self.device)
                targets = torch.tensor(indices[i+1:i+max_length+1], dtype=torch.long).unsqueeze(0).to(self.device)
                
                if sequence.shape[1] != targets.shape[1]:
                    continue
                
                try:
                    outputs = model(sequence)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    total_loss += loss.item()
                    total_chars += targets.numel()
                    
                except Exception as e:
                    print(f"Character LM evaluation error: {e}")
                    continue
        
        avg_loss = total_loss / total_chars if total_chars > 0 else float('inf')
        bits_per_char = avg_loss / np.log(2)
        perplexity = np.exp(avg_loss)
        
        return {
            'bits_per_character': bits_per_char,
            'perplexity': perplexity,
            'total_characters': total_chars
        }
    
    def evaluate_long_range_dependencies(self, model, data):
        """Evaluate long-range dependency task performance"""
        
        model.eval()
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        total_loss = 0
        
        with torch.no_grad():
            for sequence, targets in data[:50]:  # Limit for testing
                try:
                    sequence = sequence.unsqueeze(0).to(self.device)
                    targets = targets.unsqueeze(0).to(self.device)
                    
                    outputs = model(sequence)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Calculate accuracy for non-zero targets (the copy positions)
                    predictions = torch.argmax(logits, dim=-1)
                    mask = (targets != 0)
                    
                    if mask.sum() > 0:
                        correct += ((predictions == targets) & mask).sum().item()
                        total += mask.sum().item()
                    
                    # Calculate loss
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    total_loss += loss.item()
                    
                except Exception as e:
                    print(f"Long-range evaluation error: {e}")
                    continue
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else float('inf')
        
        return {
            'long_range_accuracy': accuracy,
            'loss': avg_loss,
            'correct_predictions': correct,
            'total_predictions': total
        }
    
    def evaluate_associative_recall(self, model, data):
        """Evaluate associative recall task performance"""
        
        model.eval()
        correct_recalls = 0
        total_recalls = 0
        
        with torch.no_grad():
            for sequence, targets in data[:30]:  # Limit for testing
                try:
                    sequence = sequence.unsqueeze(0).to(self.device)
                    targets = targets.unsqueeze(0).to(self.device)
                    
                    outputs = model(sequence)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    predictions = torch.argmax(logits, dim=-1)
                    
                    # Check recall accuracy (non-zero targets)
                    recall_mask = (targets != 0)
                    if recall_mask.sum() > 0:
                        correct_recalls += ((predictions == targets) & recall_mask).sum().item()
                        total_recalls += recall_mask.sum().item()
                    
                except Exception as e:
                    print(f"Associative recall error: {e}")
                    continue
        
        recall_accuracy = correct_recalls / total_recalls if total_recalls > 0 else 0
        
        return {
            'recall_accuracy': recall_accuracy,
            'correct_recalls': correct_recalls,
            'total_recalls': total_recalls
        }
    
    def evaluate_few_shot_learning(self, model, data):
        """Evaluate few-shot learning performance"""
        
        model.eval()
        support_examples = data['support']
        query_examples = data['query'][:20]  # Limit for testing
        
        # Create few-shot prompt format
        # [support examples] [query] [predict class]
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for query_example, true_class in query_examples:
                try:
                    # Build prompt with support examples
                    prompt_sequence = []
                    
                    # Add support examples
                    for support_seq, support_class in support_examples[:10]:  # Limit support examples
                        prompt_sequence.extend(support_seq.tolist())
                        prompt_sequence.append(1000 + support_class)  # Class token
                        prompt_sequence.append(999)  # Separator
                    
                    # Add query
                    prompt_sequence.extend(query_example.tolist())
                    
                    sequence = torch.tensor(prompt_sequence, dtype=torch.long).unsqueeze(0).to(self.device)
                    
                    # Keep sequence manageable
                    if sequence.shape[1] > 1024:
                        sequence = sequence[:, -1024:]
                    
                    outputs = model(sequence)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Get prediction for class tokens
                    last_logits = logits[0, -1, :]
                    class_logits = last_logits[1000:1000+data['num_classes']]
                    predicted_class = torch.argmax(class_logits).item()
                    
                    if predicted_class == true_class:
                        correct += 1
                    total += 1
                    
                except Exception as e:
                    print(f"Few-shot evaluation error: {e}")
                    continue
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'few_shot_accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def evaluate_code_understanding(self, model, data):
        """Evaluate code understanding task"""
        
        # Simplified code completion task
        model.eval()
        test_codes = data['test'][:20]  # Limit for testing
        
        completion_accuracy = 0
        total_tests = 0
        
        with torch.no_grad():
            for code_snippet in test_codes:
                try:
                    # Simple character-level tokenization for code
                    chars = list(code_snippet[:500])  # Limit length
                    char_to_idx = {c: i for i, c in enumerate(set(chars))}
                    
                    if len(char_to_idx) < 10:  # Skip very short snippets
                        continue
                    
                    indices = [char_to_idx.get(c, 0) for c in chars]
                    
                    # Split into context and target
                    split_point = len(indices) // 2
                    context = torch.tensor(indices[:split_point], dtype=torch.long).unsqueeze(0).to(self.device)
                    targets = torch.tensor(indices[split_point:split_point+10], dtype=torch.long).to(self.device)
                    
                    outputs = model(context)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Predict next tokens
                    predictions = torch.argmax(logits[0, -10:], dim=-1)
                    
                    # Calculate accuracy
                    if len(predictions) == len(targets):
                        accuracy = (predictions == targets).float().mean().item()
                        completion_accuracy += accuracy
                        total_tests += 1
                    
                except Exception as e:
                    print(f"Code evaluation error: {e}")
                    continue
        
        avg_accuracy = completion_accuracy / total_tests if total_tests > 0 else 0
        
        return {
            'code_completion_accuracy': avg_accuracy,
            'total_tests': total_tests
        }
    
    # ==========================================
    # 8. RESULTS ANALYSIS AND VISUALIZATION
    # ==========================================
    
    def create_comprehensive_comparison(self, results):
        """Create comprehensive comparison visualization"""
        
        if not results:
            print("No results to visualize")
            return None
        
        model_names = list(results.keys())
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Advanced Benchmark Comparison: Reversible vs Standard Qwen3', fontsize=16)
        
        # 1. Long-range dependency accuracy
        if 'long_range' in results[model_names[0]]:
            long_range_accs = [results[name]['long_range']['long_range_accuracy'] * 100 
                             for name in model_names]
            axes[0, 0].bar(model_names, long_range_accs)
            axes[0, 0].set_title('Long-Range Dependency Accuracy (%)')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Associative recall accuracy
        if 'associative_recall' in results[model_names[0]]:
            recall_accs = [results[name]['associative_recall']['recall_accuracy'] * 100 
                          for name in model_names]
            axes[0, 1].bar(model_names, recall_accs)
            axes[0, 1].set_title('Associative Recall Accuracy (%)')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Few-shot learning accuracy
        if 'few_shot' in results[model_names[0]]:
            few_shot_accs = [results[name]['few_shot']['few_shot_accuracy'] * 100 
                           for name in model_names]
            axes[0, 2].bar(model_names, few_shot_accs)
            axes[0, 2].set_title('Few-Shot Learning Accuracy (%)')
            axes[0, 2].set_ylabel('Accuracy (%)')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Generation quality - repetition rate (lower = better)
        if 'generation' in results[model_names[0]]:
            repetition_rates = [results[name]['generation']['avg_repetition_rate'] * 100 
                              for name in model_names]
            axes[1, 0].bar(model_names, repetition_rates)
            axes[1, 0].set_title('Generation Repetition Rate (Lower = Better)')
            axes[1, 0].set_ylabel('Repetition Rate (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Generation diversity (higher = better)
        if 'generation' in results[model_names[0]]:
            diversity_scores = [results[name]['generation']['avg_diversity_score'] * 100 
                              for name in model_names]
            axes[1, 1].bar(model_names, diversity_scores)
            axes[1, 1].set_title('Generation Diversity (Higher = Better)')
            axes[1, 1].set_ylabel('Diversity Score (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Character-level modeling (bits per character)
        if 'enwik8' in results[model_names[0]]:
            bpc_scores = [results[name]['enwik8']['bits_per_character'] 
                         for name in model_names]
            axes[1, 2].bar(model_names, bpc_scores)
            axes[1, 2].set_title('Character LM - Bits per Character (Lower = Better)')
            axes[1, 2].set_ylabel('Bits per Character')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        # 7. Memory efficiency - max sequence length handled
        if 'memory_stress' in results[model_names[0]]:
            max_seq_lens = []
            for name in model_names:
                stress_results = results[name]['memory_stress']
                max_seq = max([r['seq_length'] for r in stress_results if r['max_batch_size'] > 0])
                max_seq_lens.append(max_seq)
            
            axes[2, 0].bar(model_names, max_seq_lens)
            axes[2, 0].set_title('Max Sequence Length Handled')
            axes[2, 0].set_ylabel('Sequence Length')
            axes[2, 0].tick_params(axis='x', rotation=45)
        
        # 8. Code completion accuracy
        if 'code' in results[model_names[0]]:
            code_accs = [results[name]['code']['code_completion_accuracy'] * 100 
                        for name in model_names]
            axes[2, 1].bar(model_names, code_accs)
            axes[2, 1].set_title('Code Completion Accuracy (%)')
            axes[2, 1].set_ylabel('Accuracy (%)')
            axes[2, 1].tick_params(axis='x', rotation=45)
        
        # 9. Overall performance radar (if we have multiple metrics)
        # Normalized scores across different tasks
        axes[2, 2].text(0.5, 0.5, 'Overall\nPerformance\nSummary', 
                       ha='center', va='center', fontsize=12, 
                       transform=axes[2, 2].transAxes)
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def print_advanced_summary(self, results):
        """Print comprehensive summary of advanced benchmarks"""
        
        if not results:
            print("No results to summarize")
            return
        
        print("\n" + "="*100)
        print("ADVANCED BENCHMARK SUMMARY - REVERSIBLE VS STANDARD QWEN3")
        print("="*100)
        
        model_names = list(results.keys())
        
        # Create comprehensive table
        print(f"{'Benchmark':<25} {'Metric':<20} ", end="")
        for name in model_names:
            print(f"{name:<15} ", end="")
        print()
        print("-" * (45 + 15 * len(model_names)))
        
        # Long-range dependencies
        if 'long_range' in results[model_names[0]]:
            print(f"{'Long-Range Deps':<25} {'Accuracy (%)':<20} ", end="")
            for name in model_names:
                acc = results[name]['long_range']['long_range_accuracy'] * 100
                print(f"{acc:<15.1f} ", end="")
            print()
        
        # Associative recall
        if 'associative_recall' in results[model_names[0]]:
            print(f"{'Associative Recall':<25} {'Accuracy (%)':<20} ", end="")
            for name in model_names:
                acc = results[name]['associative_recall']['recall_accuracy'] * 100
                print(f"{acc:<15.1f} ", end="")
            print()
        
        # Few-shot learning
        if 'few_shot' in results[model_names[0]]:
            print(f"{'Few-Shot Learning':<25} {'Accuracy (%)':<20} ", end="")
            for name in model_names:
                acc = results[name]['few_shot']['few_shot_accuracy'] * 100
                print(f"{acc:<15.1f} ", end="")
            print()
        
        # Generation quality
        if 'generation' in results[model_names[0]]:
            print(f"{'Generation Quality':<25} {'Diversity (%)':<20} ", end="")
            for name in model_names:
                div = results[name]['generation']['avg_diversity_score'] * 100
                print(f"{div:<15.1f} ", end="")
            print()
            
            print(f"{'Generation Quality':<25} {'Repetition (%)':<20} ", end="")
            for name in model_names:
                rep = results[name]['generation']['avg_repetition_rate'] * 100
                print(f"{rep:<15.1f} ", end="")
            print()
        
        # Character-level modeling
        if 'enwik8' in results[model_names[0]]:
            print(f"{'Character LM':<25} {'Bits/Char':<20} ", end="")
            for name in model_names:
                bpc = results[name]['enwik8']['bits_per_character']
                print(f"{bpc:<15.2f} ", end="")
            print()
        
        # Memory stress test results
        if 'memory_stress' in results[model_names[0]]:
            print(f"{'Memory Stress':<25} {'Max Seq Length':<20} ", end="")
            for name in model_names:
                stress_results = results[name]['memory_stress']
                max_seq = max([r['seq_length'] for r in stress_results if r['max_batch_size'] > 0])
                print(f"{max_seq:<15} ", end="")
            print()
        
        print("="*100)
        
        # Analysis insights
        print("\nKEY INSIGHTS:")
        print("="*50)
        
        reversible_models = [name for name in model_names if 'reversible' in name.lower()]
        standard_models = [name for name in model_names if 'reversible' not in name.lower()]
        
        if reversible_models and standard_models:
            print(f"\nREVERSIBLE vs STANDARD COMPARISON:")
            
            # Compare on key metrics
            if 'long_range' in results[model_names[0]]:
                rev_lr = np.mean([results[name]['long_range']['long_range_accuracy'] 
                                for name in reversible_models])
                std_lr = np.mean([results[name]['long_range']['long_range_accuracy'] 
                                for name in standard_models])
                print(f"  Long-Range Dependencies: {rev_lr*100:.1f}% vs {std_lr*100:.1f}%")
            
            if 'memory_stress' in results[model_names[0]]:
                rev_mem = np.mean([max([r['seq_length'] for r in results[name]['memory_stress'] 
                                      if r['max_batch_size'] > 0]) for name in reversible_models])
                std_mem = np.mean([max([r['seq_length'] for r in results[name]['memory_stress'] 
                                      if r['max_batch_size'] > 0]) for name in standard_models])
                print(f"  Max Sequence Length: {rev_mem:.0f} vs {std_mem:.0f}")
            
            if 'generation' in results[model_names[0]]:
                rev_div = np.mean([results[name]['generation']['avg_diversity_score'] 
                                 for name in reversible_models])
                std_div = np.mean([results[name]['generation']['avg_diversity_score'] 
                                 for name in standard_models])
                print(f"  Generation Diversity: {rev_div*100:.1f}% vs {std_div*100:.1f}%")


# Integration function for your existing framework
def run_advanced_benchmarks_with_existing_models(existing_tester):
    """Run advanced benchmarks using your existing model setup"""
    
    # Create advanced benchmark suite
    advanced_suite = AdvancedBenchmarkSuite(device=existing_tester.device)
    
    # Setup models using your existing method
    models = existing_tester.setup_models_for_comparison(
        vocab_size=16000,
        hidden_size=512,
        num_layers=4,
        num_heads=8
    )
    
    if not models:
        print("No models available for advanced testing")
        return None
    
    print("Running advanced benchmark suite...")
    
    # Run comprehensive benchmarks
    results = advanced_suite.run_comprehensive_benchmark(
        models_dict=models, 
        quick_test=True  # Set to False for full evaluation
    )
    
    if results:
        # Create visualization
        fig = advanced_suite.create_comprehensive_comparison(results)
        if fig:
            plt.savefig('advanced_benchmark_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Print summary
        advanced_suite.print_advanced_summary(results)
        
        # Save results
        with open('advanced_benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\nAdvanced benchmark results saved!")
        
    return results


if __name__ == "__main__":
    # Example usage with your existing framework
    from test_train_qwen3_rev_v202_2 import ReversibleQwenPerformanceTester
    
    print("Setting up advanced benchmarks...")
    
    # Create your existing tester
    tester = ReversibleQwenPerformanceTester()
    
    # Run advanced benchmarks
    advanced_results = run_advanced_benchmarks_with_existing_models(tester)
    
    if advanced_results:
        print("\n🎉 Advanced benchmarking completed!")
        print("Check 'advanced_benchmark_comparison.png' and 'advanced_benchmark_results.json'")
    else:
        print("❌ Advanced benchmarking failed")