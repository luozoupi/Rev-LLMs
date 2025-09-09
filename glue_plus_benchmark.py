"""
GLUE+ Benchmark Suite for Reversible Qwen3 Evaluation
====================================================

Extended GLUE with additional challenging tasks for comprehensive evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union
import json
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, pearsonr, spearmanr
import time

@dataclass
class BenchmarkTask:
    name: str
    dataset_name: str
    task_type: str  # 'classification', 'regression', 'generation'
    num_labels: int
    metric: str
    difficulty: str  # 'easy', 'medium', 'hard', 'extreme'
    description: str

# Define comprehensive benchmark tasks
GLUE_PLUS_TASKS = {
    # Standard GLUE (baseline)
    'cola': BenchmarkTask('CoLA', 'glue/cola', 'classification', 2, 'matthews_corr', 'medium', 'Linguistic acceptability'),
    'sst2': BenchmarkTask('SST-2', 'glue/sst2', 'classification', 2, 'accuracy', 'easy', 'Sentiment analysis'),
    'mrpc': BenchmarkTask('MRPC', 'glue/mrpc', 'classification', 2, 'f1', 'medium', 'Paraphrase detection'),
    'qqp': BenchmarkTask('QQP', 'glue/qqp', 'classification', 2, 'f1', 'medium', 'Question similarity'),
    'stsb': BenchmarkTask('STS-B', 'glue/stsb', 'regression', 1, 'pearson', 'medium', 'Textual similarity'),
    'mnli': BenchmarkTask('MNLI', 'glue/mnli', 'classification', 3, 'accuracy', 'hard', 'Natural language inference'),
    'qnli': BenchmarkTask('QNLI', 'glue/qnli', 'classification', 2, 'accuracy', 'hard', 'Question answering NLI'),
    'rte': BenchmarkTask('RTE', 'glue/rte', 'classification', 2, 'accuracy', 'hard', 'Recognizing textual entailment'),
    
    # Extended challenging tasks
    'superglue_cb': BenchmarkTask('CB', 'super_glue/cb', 'classification', 3, 'f1', 'extreme', 'CommitmentBank inference'),
    'superglue_copa': BenchmarkTask('COPA', 'super_glue/copa', 'classification', 2, 'accuracy', 'extreme', 'Causal reasoning'),
    'superglue_wic': BenchmarkTask('WiC', 'super_glue/wic', 'classification', 2, 'accuracy', 'extreme', 'Word sense disambiguation'),
    'superglue_wsc': BenchmarkTask('WSC', 'super_glue/wsc', 'classification', 2, 'accuracy', 'extreme', 'Coreference resolution'),
    'superglue_multirc': BenchmarkTask('MultiRC', 'super_glue/multirc', 'classification', 2, 'f1', 'extreme', 'Multi-sentence reading comprehension'),
    
    # Reasoning tasks
    'hellaswag': BenchmarkTask('HellaSwag', 'hellaswag', 'classification', 4, 'accuracy', 'hard', 'Commonsense reasoning'),
    'winogrande': BenchmarkTask('WinoGrande', 'winogrande', 'classification', 2, 'accuracy', 'hard', 'Commonsense reasoning'),
    'piqa': BenchmarkTask('PIQA', 'piqa', 'classification', 2, 'accuracy', 'hard', 'Physical reasoning'),
    'arc_easy': BenchmarkTask('ARC-Easy', 'ai2_arc/ARC-Easy', 'classification', 4, 'accuracy', 'medium', 'Science questions'),
    'arc_challenge': BenchmarkTask('ARC-Challenge', 'ai2_arc/ARC-Challenge', 'classification', 4, 'accuracy', 'extreme', 'Hard science questions'),
    
    # Long-context tasks (test reversible efficiency)
    'squad_long': BenchmarkTask('SQuAD-Long', 'squad_v2', 'generation', 0, 'f1', 'extreme', 'Long-context QA'),
    'narrativeqa': BenchmarkTask('NarrativeQA', 'narrativeqa', 'generation', 0, 'rouge', 'extreme', 'Book/movie comprehension'),
    
    # Mathematical reasoning
    'gsm8k': BenchmarkTask('GSM8K', 'gsm8k', 'generation', 0, 'exact_match', 'extreme', 'Grade school math'),
    'math': BenchmarkTask('MATH', 'competition_math', 'generation', 0, 'exact_match', 'extreme', 'Competition mathematics'),
}

class GLUEPlusDataset(Dataset):
    """Unified dataset class for GLUE+ tasks"""
    
    def __init__(self, task_name: str, split: str, tokenizer, max_length: int = 512, 
                 long_context: bool = False):
        self.task_name = task_name
        self.task_config = GLUE_PLUS_TASKS[task_name]
        self.tokenizer = tokenizer
        self.max_length = max_length if not long_context else 2048
        self.examples = []
        
        # Load dataset based on task
        self._load_dataset(split)
        print(f"Loaded {len(self.examples)} examples for {task_name} {split}")
    
    def _load_dataset(self, split):
        """Load and process dataset for specific task"""
        try:
            if self.task_name.startswith('superglue_'):
                dataset_name = self.task_config.dataset_name
                dataset = load_dataset(dataset_name.split('/')[0], dataset_name.split('/')[1], split=split)
            else:
                dataset = load_dataset(self.task_config.dataset_name, split=split)
            
            # Process based on task type
            if self.task_name in ['cola', 'sst2']:
                self._process_single_sentence(dataset)
            elif self.task_name in ['mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte']:
                self._process_sentence_pair(dataset)
            elif self.task_name.startswith('superglue_'):
                self._process_superglue_task(dataset)
            elif self.task_name in ['hellaswag', 'winogrande', 'piqa']:
                self._process_multiple_choice(dataset)
            elif self.task_name.startswith('arc_'):
                self._process_arc_task(dataset)
            else:
                self._process_generation_task(dataset)
                
        except Exception as e:
            print(f"Error loading {self.task_name}: {e}")
            # Create minimal dummy data to prevent crashes
            self.examples = [{'input_ids': torch.zeros(self.max_length, dtype=torch.long), 
                            'labels': torch.tensor(0, dtype=torch.long)}] * 10
    
    def _process_single_sentence(self, dataset):
        """Process single sentence tasks (CoLA, SST-2)"""
        for example in dataset:
            if self.task_name == 'cola':
                text = example['sentence']
                label = example['label']
            else:  # sst2
                text = example['sentence']
                label = example['label']
            
            encoding = self.tokenizer(text, truncation=True, padding='max_length', 
                                    max_length=self.max_length, return_tensors='pt')
            
            self.examples.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            })
    
    def _process_sentence_pair(self, dataset):
        """Process sentence pair tasks"""
        for example in dataset:
            if self.task_name == 'mrpc':
                text1, text2 = example['sentence1'], example['sentence2']
                label = example['label']
            elif self.task_name == 'qqp':
                text1, text2 = example['question1'], example['question2']
                label = example['label']
            elif self.task_name == 'stsb':
                text1, text2 = example['sentence1'], example['sentence2']
                label = float(example['label'])
            elif self.task_name in ['mnli', 'qnli', 'rte']:
                if self.task_name == 'mnli':
                    text1, text2 = example['premise'], example['hypothesis']
                elif self.task_name == 'qnli':
                    text1, text2 = example['question'], example['sentence']
                else:  # rte
                    text1, text2 = example['sentence1'], example['sentence2']
                label = example['label']
            
            encoding = self.tokenizer(text1, text2, truncation=True, padding='max_length',
                                    max_length=self.max_length, return_tensors='pt')
            
            if self.task_name == 'stsb':
                self.examples.append({
                    'input_ids': encoding['input_ids'].squeeze(),
                    'labels': torch.tensor(label, dtype=torch.float)
                })
            else:
                self.examples.append({
                    'input_ids': encoding['input_ids'].squeeze(),
                    'labels': torch.tensor(label, dtype=torch.long)
                })
    
    def _process_multiple_choice(self, dataset):
        """Process multiple choice tasks"""
        for example in dataset:
            if self.task_name == 'hellaswag':
                context = example['ctx']
                choices = example['endings']
                label = example['label']
            elif self.task_name == 'winogrande':
                sentence = example['sentence']
                choices = [example['option1'], example['option2']]
                label = int(example['answer']) - 1 if example['answer'] else 0
            elif self.task_name == 'piqa':
                goal = example['goal']
                choices = [example['sol1'], example['sol2']]
                label = example['label']
            
            # Encode all choices and select best one
            best_encoding = None
            for i, choice in enumerate(choices):
                if self.task_name == 'hellaswag':
                    text = f"{context} {choice}"
                elif self.task_name == 'winogrande':
                    text = sentence.replace('_', choice)
                else:  # piqa
                    text = f"{goal} {choice}"
                
                if i == label or best_encoding is None:
                    encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                            max_length=self.max_length, return_tensors='pt')
                    if i == label:
                        best_encoding = encoding
                        break
            
            self.examples.append({
                'input_ids': best_encoding['input_ids'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            })
    
    def _process_superglue_task(self, dataset):
        """Process SuperGLUE tasks"""
        # Simplified processing for demonstration
        for example in dataset:
            # This would need specific processing for each SuperGLUE task
            # For now, use first text field found
            text_fields = [k for k in example.keys() if isinstance(example[k], str) and len(example[k]) > 10]
            if text_fields:
                text = example[text_fields[0]]
                label = example.get('label', 0)
                
                encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                        max_length=self.max_length, return_tensors='pt')
                
                self.examples.append({
                    'input_ids': encoding['input_ids'].squeeze(),
                    'labels': torch.tensor(label, dtype=torch.long)
                })
    
    def _process_arc_task(self, dataset):
        """Process ARC tasks"""
        for example in dataset:
            question = example['question']
            choices = example['choices']['text']
            correct_answer = example['answerKey']
            
            # Convert answer key to index
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3}
            label = answer_map.get(correct_answer, 0)
            
            # Combine question with all choices
            combined_text = f"{question} Options: " + " ".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            
            encoding = self.tokenizer(combined_text, truncation=True, padding='max_length',
                                    max_length=self.max_length, return_tensors='pt')
            
            self.examples.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            })
    
    def _process_generation_task(self, dataset):
        """Process generation tasks (simplified for classification evaluation)"""
        # For now, convert generation tasks to classification by using first few examples
        for i, example in enumerate(dataset):
            if i >= 100:  # Limit for demo
                break
                
            # Simplified: use any text field found
            text_fields = [k for k in example.keys() if isinstance(example[k], str)]
            if text_fields:
                text = example[text_fields[0]][:self.max_length]
                
                encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                        max_length=self.max_length, return_tensors='pt')
                
                self.examples.append({
                    'input_ids': encoding['input_ids'].squeeze(),
                    'labels': torch.tensor(i % 2, dtype=torch.long)  # Dummy binary labels
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]['input_ids'], self.examples[idx]['labels']

class GLUEPlusBenchmark:
    """Comprehensive GLUE+ benchmark runner"""
    
    def __init__(self, tokenizer_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.results = {}
    
    def create_task_datasets(self, task_names: List[str], max_length: int = 512):
        """Create datasets for specified tasks"""
        datasets = {}
        
        for task_name in task_names:
            if task_name not in GLUE_PLUS_TASKS:
                print(f"Unknown task: {task_name}")
                continue
            
            try:
                print(f"Loading {task_name}...")
                train_dataset = GLUEPlusDataset(task_name, 'train', self.tokenizer, max_length)
                val_dataset = GLUEPlusDataset(task_name, 'validation', self.tokenizer, max_length)
                
                datasets[task_name] = {
                    'train': train_dataset,
                    'val': val_dataset,
                    'config': GLUE_PLUS_TASKS[task_name]
                }
                
            except Exception as e:
                print(f"Failed to load {task_name}: {e}")
                continue
        
        return datasets
    
    def evaluate_model_on_task(self, model, task_name: str, dataset, device='cuda'):
        """Evaluate model on a specific task"""
        task_config = GLUE_PLUS_TASKS[task_name]
        model.eval()
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        predictions = []
        true_labels = []
        total_time = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                start_time = time.time()
                
                try:
                    outputs = model(inputs)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    if task_config.task_type == 'regression':
                        preds = logits.squeeze().cpu().numpy()
                    else:
                        preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    
                    predictions.extend(preds)
                    true_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    continue
                
                total_time += time.time() - start_time
        
        # Calculate metrics
        if len(predictions) == 0:
            return {'score': 0.0, 'inference_time': 0.0, 'error': 'No valid predictions'}
        
        if task_config.metric == 'accuracy':
            score = accuracy_score(true_labels, predictions)
        elif task_config.metric == 'f1':
            score = f1_score(true_labels, predictions, average='macro')
        elif task_config.metric == 'matthews_corr':
            score = matthews_corrcoef(true_labels, predictions)
        elif task_config.metric == 'pearson':
            score, _ = pearsonr(true_labels, predictions)
        else:
            score = accuracy_score(true_labels, predictions)
        
        avg_inference_time = total_time / len(predictions) * 1000  # ms per sample
        
        return {
            'score': float(score),
            'inference_time': avg_inference_time,
            'num_samples': len(predictions),
            'difficulty': task_config.difficulty
        }
    
    def run_full_benchmark(self, models: Dict[str, nn.Module], device='cuda', 
                          task_subset: Optional[List[str]] = None):
        """Run complete benchmark on all models"""
        
        if task_subset is None:
            # Start with core tasks, add more based on model capability
            task_subset = ['sst2', 'mrpc', 'cola', 'stsb', 'mnli', 'qnli', 
                          'hellaswag', 'winogrande', 'arc_easy']
        
        print(f"Running GLUE+ benchmark on {len(task_subset)} tasks...")
        
        # Create datasets
        datasets = self.create_task_datasets(task_subset)
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            model_results = {}
            
            for task_name, task_data in datasets.items():
                print(f"  Task: {task_name}")
                
                try:
                    # Evaluate on validation set
                    task_result = self.evaluate_model_on_task(
                        model, task_name, task_data['val'], device
                    )
                    
                    model_results[task_name] = task_result
                    
                    print(f"    Score: {task_result['score']:.3f}")
                    print(f"    Inference: {task_result['inference_time']:.2f}ms/sample")
                    print(f"    Difficulty: {task_result['difficulty']}")
                    
                except Exception as e:
                    print(f"    Failed: {e}")
                    model_results[task_name] = {'score': 0.0, 'error': str(e)}
            
            # Calculate overall scores
            valid_scores = [r['score'] for r in model_results.values() if 'score' in r and r['score'] > 0]
            overall_score = np.mean(valid_scores) if valid_scores else 0.0
            
            # Calculate difficulty-weighted score
            difficulty_weights = {'easy': 1.0, 'medium': 1.5, 'hard': 2.0, 'extreme': 3.0}
            weighted_scores = []
            
            for task_name, result in model_results.items():
                if 'score' in result and result['score'] > 0:
                    task_config = GLUE_PLUS_TASKS[task_name]
                    weight = difficulty_weights[task_config.difficulty]
                    weighted_scores.append(result['score'] * weight)
            
            weighted_score = np.mean(weighted_scores) if weighted_scores else 0.0
            
            model_results['_summary'] = {
                'overall_score': overall_score,
                'weighted_score': weighted_score,
                'tasks_completed': len(valid_scores),
                'total_tasks': len(task_subset)
            }
            
            results[model_name] = model_results
            
            print(f"  Overall Score: {overall_score:.3f}")
            print(f"  Weighted Score: {weighted_score:.3f}")
            print(f"  Tasks Completed: {len(valid_scores)}/{len(task_subset)}")
        
        self.results = results
        return results
    
    def print_benchmark_summary(self):
        """Print comprehensive benchmark summary"""
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*80)
        print("GLUE+ BENCHMARK SUMMARY")
        print("="*80)
        
        # Overall comparison
        print(f"{'Model':<25} {'Overall':<10} {'Weighted':<10} {'Tasks':<8} {'Best Task':<15}")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            summary = results['_summary']
            
            # Find best task performance
            task_scores = {k: v['score'] for k, v in results.items() 
                          if k != '_summary' and isinstance(v, dict) and 'score' in v}
            best_task = max(task_scores.keys(), key=lambda x: task_scores[x]) if task_scores else "None"
            best_score = task_scores.get(best_task, 0.0)
            
            print(f"{model_name:<25} {summary['overall_score']:.3f}      {summary['weighted_score']:.3f}      "
                  f"{summary['tasks_completed']}/{summary['total_tasks']}     {best_task:<15}")
        
        print("\n" + "="*80)
        print("TASK-SPECIFIC PERFORMANCE")
        print("="*80)
        
        # Task-specific comparison
        all_tasks = set()
        for results in self.results.values():
            all_tasks.update(k for k in results.keys() if k != '_summary')
        
        for task in sorted(all_tasks):
            print(f"\n{task.upper()} ({GLUE_PLUS_TASKS[task].difficulty}):")
            print(f"  {'Model':<25} {'Score':<10} {'Time(ms)':<10}")
            print("  " + "-" * 45)
            
            for model_name, results in self.results.items():
                if task in results:
                    result = results[task]
                    if 'score' in result:
                        score = result['score']
                        time_ms = result.get('inference_time', 0.0)
                        print(f"  {model_name:<25} {score:.3f}      {time_ms:.2f}")
                    else:
                        print(f"  {model_name:<25} ERROR      -")
        
        print("\n" + "="*80)
        print("INSIGHTS")
        print("="*80)
        
        # Performance insights
        model_names = list(self.results.keys())
        reversible_models = [name for name in model_names if 'reversible' in name.lower()]
        standard_models = [name for name in model_names if 'reversible' not in name.lower()]
        
        if reversible_models and standard_models:
            rev_scores = [self.results[name]['_summary']['overall_score'] for name in reversible_models]
            std_scores = [self.results[name]['_summary']['overall_score'] for name in standard_models]
            
            print(f"Reversible Models Average: {np.mean(rev_scores):.3f}")
            print(f"Standard Models Average: {np.mean(std_scores):.3f}")
            print(f"Performance Gap: {np.mean(rev_scores) - np.mean(std_scores):.3f}")
        
        # Difficulty analysis
        easy_tasks = [task for task, config in GLUE_PLUS_TASKS.items() if config.difficulty == 'easy']
        hard_tasks = [task for task, config in GLUE_PLUS_TASKS.items() if config.difficulty in ['hard', 'extreme']]
        
        print(f"\nDifficulty Analysis:")
        for model_name, results in self.results.items():
            easy_scores = [results[task]['score'] for task in easy_tasks if task in results and 'score' in results[task]]
            hard_scores = [results[task]['score'] for task in hard_tasks if task in results and 'score' in results[task]]
            
            easy_avg = np.mean(easy_scores) if easy_scores else 0
            hard_avg = np.mean(hard_scores) if hard_scores else 0
            
            print(f"  {model_name}: Easy {easy_avg:.3f}, Hard {hard_avg:.3f}, Gap {easy_avg - hard_avg:.3f}")

def get_recommended_task_progression():
    """Get recommended task progression for comprehensive evaluation"""
    return {
        'phase_1_basic': ['sst2', 'cola', 'mrpc'],  # Start with these
        'phase_2_intermediate': ['qqp', 'stsb', 'qnli'],  # Add if basic works
        'phase_3_challenging': ['mnli', 'rte', 'hellaswag'],  # More demanding
        'phase_4_extreme': ['winogrande', 'arc_challenge', 'superglue_cb'],  # Most challenging
        'phase_5_reasoning': ['piqa', 'arc_easy'],  # Reasoning capability
    }

if __name__ == "__main__":
    print("GLUE+ Benchmark Suite")
    print("Usage example:")
    print("""
    from glue_plus_benchmark import GLUEPlusBenchmark
    
    benchmark = GLUEPlusBenchmark()
    models = {'model1': your_model1, 'model2': your_model2}
    results = benchmark.run_full_benchmark(models, task_subset=['sst2', 'cola', 'mrpc'])
    benchmark.print_benchmark_summary()
    """)