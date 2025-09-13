"""
Qwen3 Evaluation Suite - Comprehensive Benchmark Framework
=========================================================

This framework provides comprehensive evaluation for Reversible vs Standard Qwen3 models,
inspired by DiZO's evaluation methodology but adapted for our specific use case.

Features:
- Multiple benchmark datasets (GLUE, SuperGLUE, custom tasks)
- Memory efficiency evaluation
- Training efficiency comparison
- Zero-order optimization compatibility
- Detailed performance analysis
- Statistical significance testing

Usage:
    python run_evaluation.py --config configs/standard_eval.yaml
    python run_evaluation.py --models reversible,standard --tasks glue,memory
"""

import logging
import os
import sys
import argparse
import time
import yaml
import json
import numpy as np
import torch
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our models
try:
    from qwen3_reversible_02_3 import create_modern_reversible_qwen3_model, modern_train_with_mixed_precision
    from test_train_qwen3_rev_v202_21 import create_standard_qwen3_model
    MODEL_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Model imports not available: {e}")
    MODEL_IMPORTS_AVAILABLE = False

# Import evaluation frameworks
try:
    from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
    from transformers.trainer_utils import set_seed
    from datasets import load_dataset, Dataset
    import evaluate
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs"""
    
    # Model configuration
    model_names: List[str] = field(default_factory=lambda: ["reversible_qwen3", "standard_qwen3"])
    model_sizes: List[str] = field(default_factory=lambda: ["small", "medium"])  # small: 512d, medium: 1024d
    
    # Task configuration
    tasks: List[str] = field(default_factory=lambda: ["sst2", "cola", "mrpc", "rte"])
    max_samples_per_task: int = 1000
    few_shot_k: int = 4
    
    # Training configuration
    num_epochs: int = 3
    learning_rate: float = 3e-4
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    
    # Evaluation configuration
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 500
    
    # Memory and efficiency
    memory_test_lengths: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    efficiency_metrics: List[str] = field(default_factory=lambda: ["training_time", "memory_usage", "inference_speed"])
    
    # Output configuration
    output_dir: str = "evaluation_results"
    save_models: bool = False
    detailed_analysis: bool = True
    
    # Hardware configuration
    device: str = "cuda"
    mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    
    # Random seeds for reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

class TaskRegistry:
    """Registry for evaluation tasks"""
    
    GLUE_TASKS = {
        'sst2': {'num_labels': 2, 'metric': 'accuracy'},
        'cola': {'num_labels': 2, 'metric': 'matthews_correlation'},
        'mrpc': {'num_labels': 2, 'metric': 'f1'},
        'rte': {'num_labels': 2, 'metric': 'accuracy'},
        'qnli': {'num_labels': 2, 'metric': 'accuracy'},
        'qqp': {'num_labels': 2, 'metric': 'f1'},
        'mnli': {'num_labels': 3, 'metric': 'accuracy'},
        'wnli': {'num_labels': 2, 'metric': 'accuracy'},
    }
    
    CUSTOM_TASKS = {
        'memory_stress': {'description': 'Long sequence memory test'},
        'arithmetic': {'description': 'Basic arithmetic reasoning'},
        'qa_short': {'description': 'Short-form question answering'},
        'generation': {'description': 'Text generation quality'},
    }
    
    @classmethod
    def get_task_info(cls, task_name: str) -> Dict:
        """Get task information"""
        if task_name in cls.GLUE_TASKS:
            return {**cls.GLUE_TASKS[task_name], 'type': 'glue', 'name': task_name}
        elif task_name in cls.CUSTOM_TASKS:
            return {**cls.CUSTOM_TASKS[task_name], 'type': 'custom', 'name': task_name}
        else:
            raise ValueError(f"Unknown task: {task_name}")

class ModelFactory:
    """Factory for creating different model configurations"""
    
    MODEL_CONFIGS = {
        'small': {
            'hidden_size': 512,
            'num_hidden_layers': 8,
            'num_attention_heads': 8,
            'num_key_value_heads': 4,
            'intermediate_size': 2048,
            'max_position_embeddings': 2048,
        },
        'medium': {
            'hidden_size': 1024,
            'num_hidden_layers': 16,
            'num_attention_heads': 16,
            'num_key_value_heads': 8,
            'intermediate_size': 4096,
            'max_position_embeddings': 4096,
        },
        'large': {
            'hidden_size': 2048,
            'num_hidden_layers': 24,
            'num_attention_heads': 32,
            'num_key_value_heads': 16,
            'intermediate_size': 8192,
            'max_position_embeddings': 8192,
        }
    }
    
    @classmethod
    def create_model(cls, model_name: str, model_size: str, device: str = "cuda") -> torch.nn.Module:
        """Create a model with specified configuration"""
        
        if not MODEL_IMPORTS_AVAILABLE:
            raise ImportError("Model creation not available - missing imports")
        
        base_config = cls.MODEL_CONFIGS[model_size].copy()
        base_config.update({
            'vocab_size': 32000,
            'rms_norm_eps': 1e-6,
            'device': device,
            'dtype': torch.float32,
        })
        
        if "reversible" in model_name.lower():
            base_config.update({
                'attention_type': 'candidate_selection',
                'use_reversible': True,
                'reverse_thres': base_config['max_position_embeddings'] // 4,
                'candidate_pr_ratio': 0.5,
                'candidate_top_k': 40,
            })
            return create_modern_reversible_qwen3_model(**base_config)
        else:
            # Standard model (you may need to implement this)
            base_config.update({
                'attention_type': 'standard',
                'use_reversible': False,
            })
            # For now, create reversible with reversible disabled
            return create_modern_reversible_qwen3_model(**base_config)

class DatasetLoader:
    """Load and preprocess datasets for evaluation"""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_glue_task(self, task_name: str, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
        """Load GLUE task dataset"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        
        # Load dataset
        if task_name == "mnli":
            dataset = load_dataset("glue", task_name)
            train_dataset = dataset["train"]
            eval_dataset = dataset["validation_matched"]
        else:
            dataset = load_dataset("glue", task_name)
            train_dataset = dataset["train"]
            eval_dataset = dataset["validation"]
        
        # Limit samples if specified
        if max_samples and len(train_dataset) > max_samples:
            indices = random.sample(range(len(train_dataset)), max_samples)
            train_dataset = train_dataset.select(indices)
        
        # Preprocess
        train_dataset = self._preprocess_glue_dataset(train_dataset, task_name)
        eval_dataset = self._preprocess_glue_dataset(eval_dataset, task_name)
        
        return train_dataset, eval_dataset
    
    def _preprocess_glue_dataset(self, dataset: Dataset, task_name: str) -> Dataset:
        """Preprocess GLUE dataset"""
        
        def tokenize_function(examples):
            if task_name in ["mrpc", "qqp", "mnli", "qnli", "rte"]:
                # Sentence pair tasks
                key1, key2 = ("sentence1", "sentence2") if task_name != "qnli" else ("question", "sentence")
                result = self.tokenizer(
                    examples[key1],
                    examples[key2],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
            else:
                # Single sentence tasks
                key = "sentence" if task_name in ["sst2", "cola"] else "text"
                result = self.tokenizer(
                    examples[key],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
            
            if "label" in examples:
                result["labels"] = examples["label"]
            
            return result
        
        return dataset.map(tokenize_function, batched=True)
    
    def create_memory_stress_dataset(self, lengths: List[int], num_samples: int = 100) -> Dataset:
        """Create dataset for memory stress testing"""
        
        data = []
        for length in lengths:
            for _ in range(num_samples // len(lengths)):
                # Create random sequence of specified length
                sequence = " ".join([f"token_{i}" for i in range(length)])
                label = length % 2  # Simple classification based on length
                data.append({"text": sequence, "label": label})
        
        dataset = Dataset.from_list(data)
        
        def tokenize_function(examples):
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max(lengths),
                return_tensors="pt"
            )
            result["labels"] = examples["label"]
            return result
        
        return dataset.map(tokenize_function, batched=True)

class MemoryProfiler:
    """Profile memory usage during model operations"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.measurements = {}
    
    def measure_memory(self, operation_name: str):
        """Context manager for memory measurement"""
        return self._MemoryMeasurement(self, operation_name)
    
    class _MemoryMeasurement:
        def __init__(self, profiler, operation_name):
            self.profiler = profiler
            self.operation_name = operation_name
            self.start_memory = 0
            self.peak_memory = 0
        
        def __enter__(self):
            if self.profiler.device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self.start_memory = torch.cuda.memory_allocated()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.profiler.device == "cuda" and torch.cuda.is_available():
                self.peak_memory = torch.cuda.max_memory_allocated()
                memory_used = self.peak_memory - self.start_memory
                self.profiler.measurements[self.operation_name] = {
                    'start_memory': self.start_memory,
                    'peak_memory': self.peak_memory,
                    'memory_used': memory_used,
                    'memory_used_mb': memory_used / (1024 * 1024)
                }

class EfficiencyBenchmark:
    """Benchmark training and inference efficiency"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.memory_profiler = MemoryProfiler(device)
    
    def benchmark_training_step(self, model: torch.nn.Module, batch: Dict, 
                               num_iterations: int = 10) -> Dict[str, float]:
        """Benchmark training step efficiency"""
        
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            with torch.amp.autocast(self.device, enabled=torch.cuda.is_available()):
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else torch.mean(outputs.logits)
            loss.backward()
            optimizer.step()
        
        # Actual measurement
        with self.memory_profiler.measure_memory("training_step"):
            start_time = time.time()
            
            for _ in range(num_iterations):
                optimizer.zero_grad()
                with torch.amp.autocast(self.device, enabled=torch.cuda.is_available()):
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else torch.mean(outputs.logits)
                loss.backward()
                optimizer.step()
            
            end_time = time.time()
        
        avg_time_per_step = (end_time - start_time) / num_iterations
        memory_info = self.memory_profiler.measurements.get("training_step", {})
        
        return {
            'avg_training_time_ms': avg_time_per_step * 1000,
            'memory_usage_mb': memory_info.get('memory_used_mb', 0),
            'peak_memory_mb': memory_info.get('peak_memory', 0) / (1024 * 1024)
        }
    
    def benchmark_inference(self, model: torch.nn.Module, batch: Dict,
                          num_iterations: int = 50) -> Dict[str, float]:
        """Benchmark inference efficiency"""
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(**batch)
        
        # Actual measurement
        with self.memory_profiler.measure_memory("inference"):
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(**batch)
            
            end_time = time.time()
        
        avg_time_per_inference = (end_time - start_time) / num_iterations
        memory_info = self.memory_profiler.measurements.get("inference", {})
        
        return {
            'avg_inference_time_ms': avg_time_per_inference * 1000,
            'memory_usage_mb': memory_info.get('memory_used_mb', 0),
            'peak_memory_mb': memory_info.get('peak_memory', 0) / (1024 * 1024)
        }

class Qwen3EvaluationSuite:
    """Main evaluation suite for Qwen3 models"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = {}
        self.efficiency_benchmark = EfficiencyBenchmark(config.device)
        
        # Initialize tokenizer (you may need to adjust this)
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logger.warning(f"Could not load Qwen tokenizer: {e}")
                # Fallback to a generic tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.dataset_loader = DatasetLoader(self.tokenizer)
    
    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation suite"""
        
        logger.info("Starting Qwen3 Evaluation Suite")
        logger.info(f"Models: {self.config.model_names}")
        logger.info(f"Tasks: {self.config.tasks}")
        logger.info(f"Seeds: {self.config.seeds}")
        
        for seed in self.config.seeds:
            set_seed(seed)
            logger.info(f"Running evaluation with seed {seed}")
            
            for model_name in self.config.model_names:
                for model_size in self.config.model_sizes:
                    model_key = f"{model_name}_{model_size}_seed{seed}"
                    logger.info(f"Evaluating {model_key}")
                    
                    try:
                        # Create model
                        model = ModelFactory.create_model(model_name, model_size, self.config.device)
                        
                        # Run evaluations
                        model_results = self._evaluate_model(model, model_key)
                        self.results[model_key] = model_results
                        
                        # Clean up
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logger.error(f"Failed to evaluate {model_key}: {e}")
                        self.results[model_key] = {"error": str(e)}
        
        # Aggregate results
        aggregated_results = self._aggregate_results()
        
        # Save results
        self._save_results(aggregated_results)
        
        return aggregated_results
    
    def _evaluate_model(self, model: torch.nn.Module, model_key: str) -> Dict:
        """Evaluate a single model"""
        
        results = {
            'task_performance': {},
            'efficiency_metrics': {},
            'memory_profiles': {},
        }
        
        # Task performance evaluation
        for task_name in self.config.tasks:
            try:
                task_results = self._evaluate_task(model, task_name)
                results['task_performance'][task_name] = task_results
            except Exception as e:
                logger.error(f"Task {task_name} failed for {model_key}: {e}")
                results['task_performance'][task_name] = {"error": str(e)}
        
        # Efficiency benchmarking
        try:
            efficiency_results = self._benchmark_efficiency(model)
            results['efficiency_metrics'] = efficiency_results
        except Exception as e:
            logger.error(f"Efficiency benchmark failed for {model_key}: {e}")
            results['efficiency_metrics'] = {"error": str(e)}
        
        # Memory profiling
        try:
            memory_results = self._profile_memory_usage(model)
            results['memory_profiles'] = memory_results
        except Exception as e:
            logger.error(f"Memory profiling failed for {model_key}: {e}")
            results['memory_profiles'] = {"error": str(e)}
        
        return results
    
    def _evaluate_task(self, model: torch.nn.Module, task_name: str) -> Dict:
        """Evaluate model on a specific task"""
        
        task_info = TaskRegistry.get_task_info(task_name)
        
        if task_info['type'] == 'glue':
            return self._evaluate_glue_task(model, task_name)
        elif task_info['type'] == 'custom':
            return self._evaluate_custom_task(model, task_name)
        else:
            raise ValueError(f"Unknown task type: {task_info['type']}")
    
    def _evaluate_glue_task(self, model: torch.nn.Module, task_name: str) -> Dict:
        """Evaluate on GLUE task"""
        
        # Load dataset
        train_dataset, eval_dataset = self.dataset_loader.load_glue_task(
            task_name, self.config.max_samples_per_task
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/{task_name}",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            evaluation_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            logging_steps=50,
            fp16=self.config.mixed_precision and self.config.device == "cuda",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # Load metric
        task_info = TaskRegistry.get_task_info(task_name)
        metric = evaluate.load("glue", task_name)
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train and evaluate
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        eval_result = trainer.evaluate()
        
        return {
            'training_time': training_time,
            'train_loss': train_result.training_loss,
            'eval_metrics': eval_result,
            'num_train_samples': len(train_dataset),
            'num_eval_samples': len(eval_dataset),
        }
    
    def _evaluate_custom_task(self, model: torch.nn.Module, task_name: str) -> Dict:
        """Evaluate on custom task"""
        
        if task_name == "memory_stress":
            return self._evaluate_memory_stress(model)
        else:
            return {"error": f"Custom task {task_name} not implemented"}
    
    def _evaluate_memory_stress(self, model: torch.nn.Module) -> Dict:
        """Evaluate memory stress with long sequences"""
        
        dataset = self.dataset_loader.create_memory_stress_dataset(
            self.config.memory_test_lengths, num_samples=100
        )
        
        results = {}
        
        for length in self.config.memory_test_lengths:
            # Filter dataset for this length
            length_data = [item for item in dataset if len(item['input_ids']) <= length]
            
            if not length_data:
                continue
            
            try:
                # Test forward pass
                batch = {k: torch.stack([item[k] for item in length_data[:4]]).to(self.config.device) 
                        for k in length_data[0].keys() if k != 'labels'}
                
                with torch.no_grad():
                    start_time = time.time()
                    outputs = model(**batch)
                    forward_time = time.time() - start_time
                
                results[f"length_{length}"] = {
                    'forward_time_ms': forward_time * 1000,
                    'success': True,
                }
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[f"length_{length}"] = {
                        'success': False,
                        'error': 'OOM',
                    }
                else:
                    results[f"length_{length}"] = {
                        'success': False,
                        'error': str(e),
                    }
        
        return results
    
    def _benchmark_efficiency(self, model: torch.nn.Module) -> Dict:
        """Benchmark model efficiency"""
        
        # Create sample batch
        batch_size = self.config.batch_size
        seq_length = 512
        
        sample_batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)).to(self.config.device),
            'attention_mask': torch.ones((batch_size, seq_length)).to(self.config.device),
        }
        
        # Benchmark training
        training_metrics = self.efficiency_benchmark.benchmark_training_step(model, sample_batch)
        
        # Benchmark inference
        inference_metrics = self.efficiency_benchmark.benchmark_inference(model, sample_batch)
        
        return {
            'training': training_metrics,
            'inference': inference_metrics,
        }
    
    def _profile_memory_usage(self, model: torch.nn.Module) -> Dict:
        """Profile memory usage across different sequence lengths"""
        
        memory_profiles = {}
        
        for length in [256, 512, 1024, 2048]:
            try:
                batch = {
                    'input_ids': torch.randint(0, 1000, (1, length)).to(self.config.device),
                    'attention_mask': torch.ones((1, length)).to(self.config.device),
                }
                
                with self.efficiency_benchmark.memory_profiler.measure_memory(f"length_{length}"):
                    with torch.no_grad():
                        _ = model(**batch)
                
                memory_info = self.efficiency_benchmark.memory_profiler.measurements.get(f"length_{length}", {})
                memory_profiles[f"length_{length}"] = memory_info
                
            except Exception as e:
                memory_profiles[f"length_{length}"] = {"error": str(e)}
        
        return memory_profiles
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results across seeds and configurations"""
        
        aggregated = {
            'summary': {},
            'detailed': self.results,
            'comparisons': {},
        }
        
        # Group by model type
        reversible_results = {}
        standard_results = {}
        
        for key, result in self.results.items():
            if "reversible" in key:
                reversible_results[key] = result
            else:
                standard_results[key] = result
        
        # Calculate averages and comparisons
        if reversible_results and standard_results:
            aggregated['comparisons'] = self._compare_model_types(reversible_results, standard_results)
        
        # Summary statistics
        aggregated['summary'] = {
            'total_models_evaluated': len(self.results),
            'successful_evaluations': len([r for r in self.results.values() if 'error' not in r]),
            'failed_evaluations': len([r for r in self.results.values() if 'error' in r]),
        }
        
        return aggregated
    
    def _compare_model_types(self, reversible_results: Dict, standard_results: Dict) -> Dict:
        """Compare reversible vs standard models"""
        
        comparisons = {
            'performance_gaps': {},
            'efficiency_gaps': {},
            'memory_gaps': {},
        }
        
        # Extract metrics for comparison
        for task in self.config.tasks:
            reversible_scores = []
            standard_scores = []
            
            for result in reversible_results.values():
                if task in result.get('task_performance', {}):
                    task_result = result['task_performance'][task]
                    if 'eval_metrics' in task_result:
                        # Extract primary metric
                        eval_metrics = task_result['eval_metrics']
                        if 'eval_accuracy' in eval_metrics:
                            reversible_scores.append(eval_metrics['eval_accuracy'])
                        elif 'eval_f1' in eval_metrics:
                            reversible_scores.append(eval_metrics['eval_f1'])
            
            for result in standard_results.values():
                if task in result.get('task_performance', {}):
                    task_result = result['task_performance'][task]
                    if 'eval_metrics' in task_result:
                        eval_metrics = task_result['eval_metrics']
                        if 'eval_accuracy' in eval_metrics:
                            standard_scores.append(eval_metrics['eval_accuracy'])
                        elif 'eval_f1' in eval_metrics:
                            standard_scores.append(eval_metrics['eval_f1'])
            
            if reversible_scores and standard_scores:
                rev_avg = np.mean(reversible_scores)
                std_avg = np.mean(standard_scores)
                comparisons['performance_gaps'][task] = {
                    'reversible_avg': rev_avg,
                    'standard_avg': std_avg,
                    'gap': rev_avg - std_avg,
                    'relative_gap': (rev_avg - std_avg) / std_avg if std_avg != 0 else 0,
                }
        
        return comparisons
    
    def _save_results(self, results: Dict):
        """Save evaluation results"""
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        with open(output_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_path = output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Qwen3 Evaluation Suite - Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total models evaluated: {results['summary']['total_models_evaluated']}\n")
            f.write(f"Successful evaluations: {results['summary']['successful_evaluations']}\n")
            f.write(f"Failed evaluations: {results['summary']['failed_evaluations']}\n\n")
            
            if 'comparisons' in results and 'performance_gaps' in results['comparisons']:
                f.write("Performance Gaps (Reversible - Standard):\n")
                f.write("-" * 40 + "\n")
                for task, gap_info in results['comparisons']['performance_gaps'].items():
                    f.write(f"{task}: {gap_info['gap']:+.3f} ({gap_info['relative_gap']:+.1%})\n")
        
        logger.info(f"Results saved to {output_dir}")

def load_config(config_path: str) -> EvaluationConfig:
    """Load configuration from YAML file"""
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to EvaluationConfig
        return EvaluationConfig(**config_dict)
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return EvaluationConfig()

def main():
    parser = argparse.ArgumentParser(description='Qwen3 Evaluation Suite')
    parser.add_argument('--config', type=str, default='configs/default_eval.yaml',
                        help='Path to configuration file')
    parser.add_argument('--models', type=str, 
                        help='Comma-separated list of models (overrides config)')
    parser.add_argument('--tasks', type=str,
                        help='Comma-separated list of tasks (overrides config)')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick evaluation with reduced samples')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.models:
        config.model_names = [m.strip() for m in args.models.split(',')]
    if args.tasks:
        config.tasks = [t.strip() for t in args.tasks.split(',')]
    if args.output_dir:
        config.output_dir = args.output_dir
    config.device = args.device
    
    if args.quick:
        config.max_samples_per_task = 100
        config.num_epochs = 1
        config.seeds = [42]
        config.model_sizes = ["small"]
    
    # Run evaluation
    logger.info("Starting Qwen3 Evaluation Suite")
    logger.info(f"Configuration: {config}")
    
    suite = Qwen3EvaluationSuite(config)
    results = suite.run_full_evaluation()
    
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to {config.output_dir}")

if __name__ == "__main__":
    main()
