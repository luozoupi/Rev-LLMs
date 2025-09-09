"""
Long-Range Memory Benchmark for Reversible Qwen3
================================================

Specialized tasks to evaluate memory efficiency and long-context capabilities
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MemoryTask:
    name: str
    sequence_length: int
    memory_span: int
    difficulty: str
    description: str

MEMORY_TASKS = {
    'copy_task': MemoryTask('Copy Task', 1024, 512, 'medium', 'Copy sequence after delay'),
    'associative_recall': MemoryTask('Associative Recall', 2048, 1000, 'hard', 'Recall associated pairs'),
    'needle_in_haystack': MemoryTask('Needle in Haystack', 4096, 2000, 'extreme', 'Find specific info in long text'),
    'long_arithmetic': MemoryTask('Long Arithmetic', 1024, 800, 'hard', 'Multi-step arithmetic chains'),
    'narrative_coherence': MemoryTask('Narrative Coherence', 3072, 1500, 'extreme', 'Track story elements'),
    'variable_binding': MemoryTask('Variable Binding', 1536, 1000, 'hard', 'Track variable assignments'),
    'hierarchical_structure': MemoryTask('Hierarchical Structure', 2048, 1200, 'extreme', 'Parse nested structures'),
}

class LongRangeMemoryDataset(Dataset):
    """Dataset for long-range memory tasks"""
    
    def __init__(self, task_name: str, num_samples: int = 1000, vocab_size: int = 1000):
        self.task_name = task_name
        self.task_config = MEMORY_TASKS[task_name]
        self.vocab_size = vocab_size
        self.examples = []
        
        # Generate task-specific examples
        for _ in range(num_samples):
            if task_name == 'copy_task':
                example = self._generate_copy_task()
            elif task_name == 'associative_recall':
                example = self._generate_associative_recall()
            elif task_name == 'needle_in_haystack':
                example = self._generate_needle_haystack()
            elif task_name == 'long_arithmetic':
                example = self._generate_arithmetic_chain()
            elif task_name == 'narrative_coherence':
                example = self._generate_narrative_task()
            elif task_name == 'variable_binding':
                example = self._generate_variable_binding()
            elif task_name == 'hierarchical_structure':
                example = self._generate_hierarchical_task()
            
            if example:
                self.examples.append(example)
    
    def _generate_copy_task(self):
        """Generate copy task: [sequence] [delay] [copy] [sequence] [end]"""
        seq_len = self.task_config.sequence_length
        memory_span = self.task_config.memory_span
        
        # Generate random sequence to memorize
        sequence_length = np.random.randint(20, 50)
        sequence = np.random.randint(10, self.vocab_size // 2, sequence_length)
        
        # Create delay tokens
        delay_length = memory_span - sequence_length - 10
        delay_tokens = np.random.randint(self.vocab_size // 2, self.vocab_size - 10, delay_length)
        
        # Special tokens
        copy_token = 1
        end_token = 2
        
        # Construct full sequence
        full_sequence = np.concatenate([
            sequence,
            delay_tokens,
            [copy_token],
            sequence,  # Target: model should copy the original sequence
            [end_token]
        ])
        
        # Pad or truncate
        if len(full_sequence) > seq_len:
            full_sequence = full_sequence[:seq_len]
        else:
            padding = np.zeros(seq_len - len(full_sequence))
            full_sequence = np.concatenate([full_sequence, padding])
        
        # Create input/target
        input_seq = full_sequence[:-1]
        target_seq = full_sequence[1:]
        
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'labels': torch.tensor(target_seq, dtype=torch.long)
        }
    
    def _generate_associative_recall(self):
        """Generate associative recall: learn A->B pairs, then test recall"""
        seq_len = self.task_config.sequence_length
        
        # Generate key-value pairs
        num_pairs = 10
        keys = np.random.randint(10, 100, num_pairs)
        values = np.random.randint(100, 200, num_pairs)
        
        # Special tokens
        pair_sep = 3
        query_token = 4
        answer_token = 5
        
        # Teaching phase: present all pairs
        teaching_sequence = []
        for k, v in zip(keys, values):
            teaching_sequence.extend([k, pair_sep, v, pair_sep])
        
        # Testing phase: query random pairs
        test_queries = []
        for _ in range(5):
            idx = np.random.randint(num_pairs)
            test_queries.extend([query_token, keys[idx], answer_token, values[idx]])
        
        # Combine phases
        full_sequence = teaching_sequence + test_queries
        
        # Pad or truncate
        if len(full_sequence) > seq_len:
            full_sequence = full_sequence[:seq_len]
        else:
            padding = np.zeros(seq_len - len(full_sequence))
            full_sequence = np.concatenate([full_sequence, padding])
        
        input_seq = full_sequence[:-1]
        target_seq = full_sequence[1:]
        
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'labels': torch.tensor(target_seq, dtype=torch.long)
        }
    
    def _generate_needle_haystack(self):
        """Generate needle in haystack: find specific info in long text"""
        seq_len = self.task_config.sequence_length
        
        # Generate "haystack" - random tokens
        haystack_length = seq_len - 100
        haystack = np.random.randint(10, self.vocab_size - 10, haystack_length)
        
        # Insert "needle" - specific pattern at random position
        needle = [500, 501, 502, 503]  # Special pattern
        needle_pos = np.random.randint(100, haystack_length - len(needle) - 100)
        
        # Insert needle into haystack
        full_text = np.concatenate([
            haystack[:needle_pos],
            needle,
            haystack[needle_pos:]
        ])[:haystack_length]
        
        # Add query at the end asking for needle
        query_token = 6
        answer_tokens = needle
        
        full_sequence = np.concatenate([
            full_text,
            [query_token],
            answer_tokens
        ])
        
        # Pad to sequence length
        if len(full_sequence) < seq_len:
            padding = np.zeros(seq_len - len(full_sequence))
            full_sequence = np.concatenate([full_sequence, padding])
        
        input_seq = full_sequence[:-1]
        target_seq = full_sequence[1:]
        
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'labels': torch.tensor(target_seq, dtype=torch.long)
        }
    
    def _generate_arithmetic_chain(self):
        """Generate long arithmetic computation chains"""
        seq_len = self.task_config.sequence_length
        
        # Start with a number
        current_value = np.random.randint(1, 20)
        sequence = [current_value]
        
        # Add operation symbols
        add_token = 700
        sub_token = 701
        mul_token = 702
        equals_token = 703
        
        # Generate chain of operations
        for _ in range(20):
            operation = np.random.choice([add_token, sub_token, mul_token])
            operand = np.random.randint(1, 10)
            
            sequence.extend([operation, operand])
            
            # Track actual computation
            if operation == add_token:
                current_value += operand
            elif operation == sub_token:
                current_value -= operand
            else:  # multiply
                current_value *= operand
        
        # Add equals and result
        sequence.extend([equals_token, current_value])
        
        # Pad with random tokens to full length
        remaining_length = seq_len - len(sequence)
        if remaining_length > 0:
            filler = np.random.randint(10, 100, remaining_length)
            sequence = np.concatenate([filler, sequence])
        
        # Truncate if too long
        sequence = sequence[-seq_len:]
        
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'labels': torch.tensor(target_seq, dtype=torch.long)
        }
    
    def _generate_narrative_task(self):
        """Generate narrative coherence task"""
        seq_len = self.task_config.sequence_length
        
        # Character tokens
        characters = list(range(800, 810))  # 10 characters
        
        # Action tokens
        actions = list(range(810, 830))  # 20 actions
        
        # Location tokens
        locations = list(range(830, 840))  # 10 locations
        
        # Generate story events
        story = []
        character_locations = {}  # Track where each character is
        
        for _ in range(50):
            character = np.random.choice(characters)
            action = np.random.choice(actions)
            location = np.random.choice(locations)
            
            # Create event: [character] [action] [location]
            event = [character, action, location]
            story.extend(event)
            
            # Track character location
            character_locations[character] = location
        
        # Add query: where is character X?
        query_char = np.random.choice(list(character_locations.keys()))
        where_token = 840
        answer_location = character_locations[query_char]
        
        story.extend([where_token, query_char, answer_location])
        
        # Pad to sequence length
        if len(story) < seq_len:
            padding = np.zeros(seq_len - len(story))
            story = np.concatenate([padding, story])
        else:
            story = story[-seq_len:]
        
        input_seq = story[:-1]
        target_seq = story[1:]
        
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'labels': torch.tensor(target_seq, dtype=torch.long)
        }
    
    def _generate_variable_binding(self):
        """Generate variable binding task"""
        seq_len = self.task_config.sequence_length
        
        # Variable names (tokens 900-910)
        variables = list(range(900, 910))
        
        # Assignment operator
        assign_token = 950
        
        # Query operator
        query_token = 951
        
        sequence = []
        variable_values = {}
        
        # Generate assignments
        for _ in range(15):
            var = np.random.choice(variables)
            value = np.random.randint(100, 500)
            
            # var = value
            sequence.extend([var, assign_token, value])
            variable_values[var] = value
        
        # Add some filler operations
        filler_length = seq_len // 2
        filler = np.random.randint(10, 99, filler_length)
        sequence.extend(filler)
        
        # Query a variable
        query_var = np.random.choice(list(variable_values.keys()))
        expected_value = variable_values[query_var]
        
        sequence.extend([query_token, query_var, expected_value])
        
        # Adjust length
        if len(sequence) > seq_len:
            sequence = sequence[-seq_len:]
        else:
            padding = np.zeros(seq_len - len(sequence))
            sequence = np.concatenate([padding, sequence])
        
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'labels': torch.tensor(target_seq, dtype=torch.long)
        }
    
    def _generate_hierarchical_task(self):
        """Generate hierarchical structure parsing task"""
        seq_len = self.task_config.sequence_length
        
        # Bracket tokens
        open_bracket = 990
        close_bracket = 991
        
        # Generate nested structure
        def generate_nested_structure(depth=0, max_depth=5):
            if depth >= max_depth:
                return [np.random.randint(100, 200)]
            
            structure = [open_bracket]
            
            # Add some content
            num_elements = np.random.randint(2, 5)
            for _ in range(num_elements):
                if np.random.random() < 0.7 and depth < max_depth - 1:
                    # Add nested structure
                    structure.extend(generate_nested_structure(depth + 1, max_depth))
                else:
                    # Add simple element
                    structure.append(np.random.randint(100, 200))
            
            structure.append(close_bracket)
            return structure
        
        # Generate the main structure
        structure = generate_nested_structure()
        
        # Calculate nesting depth at each position
        depth_sequence = []
        current_depth = 0
        
        for token in structure:
            if token == open_bracket:
                current_depth += 1
                depth_sequence.append(current_depth)
            elif token == close_bracket:
                depth_sequence.append(current_depth)
                current_depth -= 1
            else:
                depth_sequence.append(current_depth)
        
        # Create task: given structure, predict depth at each position
        # We'll interleave structure tokens with depth predictions
        task_sequence = []
        for token, depth in zip(structure, depth_sequence):
            task_sequence.extend([token, depth + 200])  # Offset depth tokens
        
        # Pad to sequence length
        if len(task_sequence) < seq_len:
            padding = np.zeros(seq_len - len(task_sequence))
            task_sequence = np.concatenate([padding, task_sequence])
        else:
            task_sequence = task_sequence[:seq_len]
        
        input_seq = task_sequence[:-1]
        target_seq = task_sequence[1:]
        
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'labels': torch.tensor(target_seq, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return example['input_ids'], example['labels']

class MemoryBenchmark:
    """Benchmark runner for memory-intensive tasks"""
    
    def __init__(self):
        self.results = {}
    
    def create_memory_datasets(self, task_names: List[str], num_samples: int = 500):
        """Create datasets for memory tasks"""
        datasets = {}
        
        for task_name in task_names:
            if task_name not in MEMORY_TASKS:
                print(f"Unknown memory task: {task_name}")
                continue
            
            print(f"Generating {task_name} dataset...")
            
            try:
                train_dataset = LongRangeMemoryDataset(task_name, num_samples)
                val_dataset = LongRangeMemoryDataset(task_name, num_samples // 4)
                
                datasets[task_name] = {
                    'train': train_dataset,
                    'val': val_dataset,
                    'config': MEMORY_TASKS[task_name]
                }
                
                print(f"  Created {len(train_dataset)} train, {len(val_dataset)} val examples")
                
            except Exception as e:
                print(f"Failed to create {task_name}: {e}")
                continue
        
        return datasets
    
    def evaluate_memory_task(self, model, task_name: str, dataset, device='cuda'):
        """Evaluate model on memory task with detailed metrics"""
        model.eval()
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Small batch for memory tasks
        
        total_loss = 0
        total_tokens = 0
        correct_predictions = 0
        sequence_accuracies = []
        memory_usage = []
        
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= 100:  # Limit for memory efficiency
                    break
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Measure memory before forward pass
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                try:
                    outputs = model(inputs)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Calculate loss and accuracy
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    
                    loss = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # Accuracy calculation
                    predictions = torch.argmax(shift_logits, dim=-1)
                    mask = (shift_labels != 0)  # Non-padding tokens
                    correct = (predictions == shift_labels) & mask
                    
                    batch_tokens = mask.sum().item()
                    batch_correct = correct.sum().item()
                    
                    if batch_tokens > 0:
                        total_loss += loss.item()
                        total_tokens += batch_tokens
                        correct_predictions += batch_correct
                        
                        # Sequence-level accuracy
                        seq_accuracy = batch_correct / batch_tokens
                        sequence_accuracies.append(seq_accuracy)
                    
                    # Measure peak memory
                    if torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                        memory_usage.append(peak_memory)
                    
                except Exception as e:
                    print(f"Error evaluating batch {batch_idx}: {e}")
                    continue
        
        # Calculate final metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        token_accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
        sequence_accuracy = np.mean(sequence_accuracies) if sequence_accuracies else 0.0
        avg_memory = np.mean(memory_usage) if memory_usage else 0.0
        perplexity = np.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        return {
            'perplexity': perplexity,
            'token_accuracy': token_accuracy,
            'sequence_accuracy': sequence_accuracy,
            'avg_memory_mb': avg_memory,
            'sequence_length': MEMORY_TASKS[task_name].sequence_length,
            'memory_span': MEMORY_TASKS[task_name].memory_span,
            'difficulty': MEMORY_TASKS[task_name].difficulty,
            'samples_evaluated': len(sequence_accuracies)
        }
    
    def run_memory_benchmark(self, models: Dict[str, nn.Module], device='cuda'):
        """Run comprehensive memory benchmark"""
        
        # Start with basic tasks, progress to more challenging
        task_progression = [
            ['copy_task', 'associative_recall'],  # Basic memory
            ['needle_in_haystack', 'long_arithmetic'],  # Medium complexity
            ['variable_binding', 'narrative_coherence'],  # High complexity
            ['hierarchical_structure']  # Extreme complexity
        ]
        
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name} on memory tasks...")
            model_results = {}
            
            for phase_idx, task_phase in enumerate(task_progression):
                print(f"  Phase {phase_idx + 1}: {task_phase}")
                
                # Create datasets for this phase
                try:
                    datasets = self.create_memory_datasets(task_phase, num_samples=200)
                except Exception as e:
                    print(f"    Failed to create datasets: {e}")
                    continue
                
                phase_results = {}
                
                for task_name in task_phase:
                    if task_name not in datasets:
                        continue
                    
                    print(f"    Evaluating {task_name}...")
                    
                    try:
                        result = self.evaluate_memory_task(
                            model, task_name, datasets[task_name]['val'], device
                        )
                        
                        phase_results[task_name] = result
                        
                        print(f"      Perplexity: {result['perplexity']:.2f}")
                        print(f"      Token Accuracy: {result['token_accuracy']*100:.1f}%")
                        print(f"      Sequence Accuracy: {result['sequence_accuracy']*100:.1f}%")
                        print(f"      Memory Usage: {result['avg_memory_mb']:.1f} MB")
                        
                    except Exception as e:
                        print(f"      Failed: {e}")
                        phase_results[task_name] = {'error': str(e)}
                
                model_results[f'phase_{phase_idx + 1}'] = phase_results
                
                # Calculate phase summary
                valid_results = [r for r in phase_results.values() if 'perplexity' in r]
                if valid_results:
                    phase_summary = {
                        'avg_perplexity': np.mean([r['perplexity'] for r in valid_results if r['perplexity'] != float('inf')]),
                        'avg_token_accuracy': np.mean([r['token_accuracy'] for r in valid_results]),
                        'avg_sequence_accuracy': np.mean([r['sequence_accuracy'] for r in valid_results]),
                        'avg_memory_mb': np.mean([r['avg_memory_mb'] for r in valid_results]),
                        'tasks_completed': len(valid_results)
                    }
                    
                    model_results[f'phase_{phase_idx + 1}_summary'] = phase_summary
                    
                    print(f"    Phase Summary: {phase_summary['avg_perplexity']:.2f} PPL, "
                          f"{phase_summary['avg_token_accuracy']*100:.1f}% accuracy, "
                          f"{phase_summary['avg_memory_mb']:.1f} MB")
            
            all_results[model_name] = model_results
        
        self.results = all_results
        return all_results
    
    def print_memory_benchmark_summary(self):
        """Print comprehensive memory benchmark summary"""
        if not self.results:
            print("No memory benchmark results available")
            return
        
        print("\n" + "="*100)
        print("LONG-RANGE MEMORY BENCHMARK SUMMARY")
        print("="*100)
        
        # Phase-wise comparison
        for phase_idx in range(1, 5):  # 4 phases
            phase_key = f'phase_{phase_idx}'
            summary_key = f'phase_{phase_idx}_summary'
            
            print(f"\nPHASE {phase_idx} RESULTS:")
            print(f"{'Model':<20} {'PPL':<8} {'Token%':<8} {'Seq%':<8} {'Memory':<10} {'Tasks':<6}")
            print("-" * 70)
            
            for model_name, results in self.results.items():
                if summary_key in results:
                    summary = results[summary_key]
                    ppl = summary['avg_perplexity']
                    token_acc = summary['avg_token_accuracy'] * 100
                    seq_acc = summary['avg_sequence_accuracy'] * 100
                    memory = summary['avg_memory_mb']
                    tasks = summary['tasks_completed']
                    
                    ppl_str = f"{ppl:.1f}" if ppl != float('inf') else "inf"
                    print(f"{model_name:<20} {ppl_str:<8} {token_acc:<7.1f} {seq_acc:<7.1f} "
                          f"{memory:<9.1f} {tasks:<6}")
        
        print("\n" + "="*100)
        print("TASK-SPECIFIC ANALYSIS")
        print("="*100)
        
        # Collect all tasks
        all_tasks = set()
        for results in self.results.values():
            for phase_key, phase_results in results.items():
                if phase_key.startswith('phase_') and not phase_key.endswith('_summary'):
                    all_tasks.update(phase_results.keys())
        
        for task in sorted(all_tasks):
            if task in MEMORY_TASKS:
                task_config = MEMORY_TASKS[task]
                print(f"\n{task.upper()} (Length: {task_config.sequence_length}, "
                      f"Span: {task_config.memory_span}, Difficulty: {task_config.difficulty}):")
                print(f"  {'Model':<20} {'PPL':<8} {'Token%':<8} {'Seq%':<8} {'Memory':<10}")
                print("  " + "-" * 60)
                
                for model_name, results in self.results.items():
                    # Find this task in any phase
                    task_result = None
                    for phase_key, phase_results in results.items():
                        if phase_key.startswith('phase_') and not phase_key.endswith('_summary'):
                            if task in phase_results:
                                task_result = phase_results[task]
                                break
                    
                    if task_result and 'perplexity' in task_result:
                        ppl = task_result['perplexity']
                        token_acc = task_result['token_accuracy'] * 100
                        seq_acc = task_result['sequence_accuracy'] * 100
                        memory = task_result['avg_memory_mb']
                        
                        ppl_str = f"{ppl:.1f}" if ppl != float('inf') else "inf"
                        print(f"  {model_name:<20} {ppl_str:<8} {token_acc:<7.1f} {seq_acc:<7.1f} {memory:<9.1f}")
                    else:
                        print(f"  {model_name:<20} {'ERROR':<8} {'-':<7} {'-':<7} {'-':<9}")
        
        print("\n" + "="*100)
        print("MEMORY EFFICIENCY INSIGHTS")
        print("="*100)
        
        # Memory efficiency analysis
        reversible_models = [name for name in self.results.keys() if 'reversible' in name.lower()]
        standard_models = [name for name in self.results.keys() if 'reversible' not in name.lower()]
        
        if reversible_models and standard_models:
            print("Memory Usage Comparison (MB):")
            print(f"{'Task':<20} {'Reversible Avg':<15} {'Standard Avg':<15} {'Efficiency':<12}")
            print("-" * 70)
            
            for task in sorted(all_tasks):
                if task in MEMORY_TASKS:
                    rev_memories = []
                    std_memories = []
                    
                    for model_name in reversible_models:
                        if model_name in self.results:
                            for phase_results in self.results[model_name].values():
                                if isinstance(phase_results, dict) and task in phase_results:
                                    if 'avg_memory_mb' in phase_results[task]:
                                        rev_memories.append(phase_results[task]['avg_memory_mb'])
                    
                    for model_name in standard_models:
                        if model_name in self.results:
                            for phase_results in self.results[model_name].values():
                                if isinstance(phase_results, dict) and task in phase_results:
                                    if 'avg_memory_mb' in phase_results[task]:
                                        std_memories.append(phase_results[task]['avg_memory_mb'])
                    
                    if rev_memories and std_memories:
                        rev_avg = np.mean(rev_memories)
                        std_avg = np.mean(std_memories)
                        efficiency = std_avg / rev_avg if rev_avg > 0 else 1.0
                        
                        print(f"{task:<20} {rev_avg:<14.1f} {std_avg:<14.1f} {efficiency:<11.2f}x")

def get_recommended_memory_task_sequence():
    """Get recommended sequence for memory task evaluation"""
    return {
        'quick_test': ['copy_task'],
        'basic_memory': ['copy_task', 'associative_recall'],
        'comprehensive': ['copy_task', 'associative_recall', 'needle_in_haystack', 'long_arithmetic'],
        'extreme_test': ['copy_task', 'needle_in_haystack', 'variable_binding', 'narrative_coherence'],
        'full_suite': list(MEMORY_TASKS.keys())
    }

if __name__ == "__main__":
    print("Long-Range Memory Benchmark")
    print("Usage example:")
    print("""
    from memory_benchmark import MemoryBenchmark
    
    benchmark = MemoryBenchmark()
    models = {'reversible': reversible_model, 'standard': standard_model}
    results = benchmark.run_memory_benchmark(models)
    benchmark.print_memory_benchmark_summary()
    """)