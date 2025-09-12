"""
DiZO-Style Benchmark Framework for Reversible vs Standard Qwen3
==============================================================

Adapts the DiZO training and evaluation framework to provide fair comparison
between reversible and standard Qwen3 models on the same datasets and scale.
"""

import os
import sys
import torch
import logging
from dataclasses import dataclass
from typing import Optional, Union, List
from transformers import TrainingArguments, AutoTokenizer, AutoConfig
from transformers.training_args import IntervalStrategy

# Add DiZO modules to path for reuse
sys.path.append('/home/yul23028/DiZO/large_models')

from tasks import get_task
from utils import count_time, set_seed
from metrics import calculate_metric
from trainer import OurTrainer
from dataset import FewShotDataset

# Import our reversible Qwen3 implementation
from qwen3_reversible_02_2 import create_reversible_qwen3_model

logger = logging.getLogger(__name__)

@dataclass
class Qwen3BenchmarkArguments(TrainingArguments):
    """Extended arguments for Qwen3 reversible vs standard comparison"""
    
    # ---- Model Configuration ----
    model_type: str = "reversible"  # "reversible" or "standard"
    hidden_size: int = 1024
    num_layers: int = 6
    num_heads: int = 8
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    
    # ---- Task Configuration ----
    task_name: str = "SST2"
    num_train: int = 1000
    num_dev: Optional[int] = None
    num_eval: Optional[int] = None
    max_length: int = 512
    
    # ---- Training Strategy ----
    trainer: str = "regular"  # "none" | "regular" | "zo"
    use_reversible: bool = True
    attention_type: str = "standard"  # "standard" | "native_sparse" | "candidate_selection"
    
    # ---- Memory Optimization ----
    gradient_checkpointing: bool = True
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "float16" | "bfloat16"
    
    # ---- Evaluation ----
    result_file: Optional[str] = None
    save_detailed_results: bool = True


class Qwen3Framework:
    """
    Framework for training and evaluating Qwen3 models (reversible vs standard)
    following DiZO's methodology for fair comparison
    """
    
    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()
        
    def load_model(self):
        """Load either reversible or standard Qwen3 model based on configuration"""
        
        with count_time(f"Loading {self.args.model_type} Qwen3 model"):
            
            # Create model configuration
            config_dict = {
                'hidden_size': self.args.hidden_size,
                'num_layers': self.args.num_layers,
                'num_attention_heads': self.args.num_heads,
                'vocab_size': self.args.vocab_size,
                'max_position_embeddings': self.args.max_position_embeddings,
                'rms_norm_eps': 1e-6,
                'tie_word_embeddings': False,
                'use_cache': False,  # Disable for consistent memory usage
            }
            
            if self.args.model_type == "reversible":
                # Create reversible model using our implementation
                model = create_reversible_qwen3_model(
                    vocab_size=self.args.vocab_size,
                    hidden_size=self.args.hidden_size,
                    num_layers=self.args.num_layers,
                    num_heads=self.args.num_heads,
                    use_reversible=True,
                    attention_type=self.args.attention_type,
                    seq_len=self.args.max_length
                )
                
            elif self.args.model_type == "standard":
                # Create standard model using our implementation (with reversible=False)
                model = create_reversible_qwen3_model(
                    vocab_size=self.args.vocab_size,
                    hidden_size=self.args.hidden_size,
                    num_layers=self.args.num_layers,
                    num_heads=self.args.num_heads,
                    use_reversible=False,
                    attention_type=self.args.attention_type,
                    seq_len=self.args.max_length
                )
            else:
                raise ValueError(f"Unknown model_type: {self.args.model_type}")
            
            # Enable gradient checkpointing if requested
            if self.args.gradient_checkpointing:
                if hasattr(model, 'enable_gradient_checkpointing'):
                    model.enable_gradient_checkpointing()
                elif hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
            
            model.eval()
            
        # Load tokenizer (using Qwen3 tokenizer as base)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B-Chat", use_fast=False)
        
        # Ensure tokenizer has required special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        return model, tokenizer
    
    def forward(self, input_ids, option_len=None, generation=False):
        """
        Forward pass following DiZO's methodology
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)
        
        if generation:
            # For generation tasks (not currently implemented)
            raise NotImplementedError("Generation tasks not yet implemented")
        else:
            with torch.inference_mode():
                self.model.eval()
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
            labels = input_ids[0, 1:]
            logits = logits[0, :-1]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            
            # Return only the option part
            return selected_log_probs[-option_len:]
    
    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Single prediction following DiZO's methodology
        """
        verbose = verbose or getattr(self.args, 'verbose', False)
        
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")
        
        # Encode prompt and candidates (simplified version)
        # This would need to be adapted to use DiZO's encode_prompt function
        encoded_candidates = []
        option_lens = []
        
        for candidate in eval_sample.candidates:
            # Simple encoding - in practice, use DiZO's template system
            prompt = f"Text: {eval_sample.data.get('sentence', '')} Label: {candidate}"
            encoded = self.tokenizer.encode(prompt, max_length=self.args.max_length, 
                                          truncation=True, add_special_tokens=True)
            encoded_candidates.append(encoded)
            option_lens.append(len(self.tokenizer.encode(str(candidate), add_special_tokens=False)))
        
        outputs = []
        
        # Calculate probabilities for all candidates
        for candidate_id, encoded_candidate in enumerate(encoded_candidates):
            try:
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                outputs.append({"log_probs": selected_log_probs})
            except Exception as e:
                logger.warning(f"Error processing candidate {candidate_id}: {e}")
                # Fallback to zero probability
                outputs.append({"log_probs": torch.tensor([0.0])})
        
        scores = [x['log_probs'].mean().item() for x in outputs]
        
        if verbose:
            logger.info(f"Prediction scores: {scores}")
        
        # Determine correct candidate ID
        if isinstance(eval_sample.correct_candidate, list):
            correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
        else:
            correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)
        
        # Return prediction
        from collections import namedtuple
        Prediction = namedtuple('Prediction', ['correct_candidate', 'predicted_candidate'])
        return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(torch.argmax(torch.tensor(scores))))
    
    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluation following DiZO's methodology
        """
        from tqdm import tqdm
        
        if one_train_set_per_eval_sample:
            logger.info(f"Evaluating {len(eval_samples)} samples with individual train sets")
        else:
            logger.info(f"Evaluating {len(eval_samples)} samples with {len(train_samples)} train samples")
        
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples, desc="Evaluating")):
            try:
                prediction = self.one_step_pred(
                    train_samples[eval_id] if one_train_set_per_eval_sample else train_samples,
                    eval_sample, 
                    verbose=False
                )
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Error evaluating sample {eval_id}: {e}")
                # Add a default incorrect prediction
                from collections import namedtuple
                Prediction = namedtuple('Prediction', ['correct_candidate', 'predicted_candidate'])
                predictions.append(Prediction(correct_candidate=0, predicted_candidate=1))
        
        # Calculate metrics
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        
        return metrics
    
    def train(self, train_samples, eval_samples):
        """
        Training following DiZO's methodology with HuggingFace Trainer
        """
        logger.info(f"Training {self.args.model_type} model on {len(train_samples)} samples")
        
        # Convert samples to HuggingFace dataset format
        # This would need to be adapted to use DiZO's dataset conversion
        from torch.utils.data import Dataset
        
        class SimpleDataset(Dataset):
            def __init__(self, samples, tokenizer, max_length):
                self.samples = samples
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.samples)
                
            def __getitem__(self, idx):
                sample = self.samples[idx]
                # Simple encoding - should use DiZO's proper encoding
                text = sample.data.get('sentence', '')
                label = sample.correct_candidate if isinstance(sample.correct_candidate, int) else 0
                
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        
        train_dataset = SimpleDataset(train_samples, self.tokenizer, self.args.max_length)
        eval_dataset = SimpleDataset(eval_samples, self.tokenizer, self.args.max_length)
        
        # Setup trainer
        trainer = OurTrainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Update model reference
        self.model = trainer.model


def run_qwen3_benchmark(
    model_types=["reversible", "standard"],
    tasks=["SST2", "CoLA", "BoolQ"],
    hidden_sizes=[512, 1024],
    num_layers_list=[4, 6],
    attention_types=["standard", "candidate_selection"]
):
    """
    Run comprehensive benchmark comparing reversible vs standard Qwen3
    """
    results = {}
    
    for model_type in model_types:
        for task_name in tasks:
            for hidden_size in hidden_sizes:
                for num_layers in num_layers_list:
                    for attention_type in attention_types:
                        
                        # Skip candidate selection for standard models (for now)
                        if model_type == "standard" and attention_type == "candidate_selection":
                            continue
                            
                        config_name = f"{model_type}_{task_name}_h{hidden_size}_l{num_layers}_{attention_type}"
                        logger.info(f"Running benchmark: {config_name}")
                        
                        try:
                            # Setup arguments
                            args = Qwen3BenchmarkArguments(
                                model_type=model_type,
                                task_name=task_name,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                attention_type=attention_type,
                                num_train=1000,
                                max_length=512,
                                output_dir=f"./results/{config_name}",
                                per_device_train_batch_size=4,
                                per_device_eval_batch_size=8,
                                num_train_epochs=3,
                                logging_steps=100,
                                eval_steps=500,
                                save_steps=1000,
                                load_best_model_at_end=True,
                                metric_for_best_model="accuracy",
                                greater_is_better=True,
                            )
                            
                            # Load task
                            task = get_task(task_name)
                            
                            # Create framework
                            framework = Qwen3Framework(args, task)
                            
                            # Get train/eval samples
                            train_samples = task.samples['train'][:args.num_train]
                            eval_samples = task.samples['test'][:200]  # Limit for faster evaluation
                            
                            if args.trainer != "none":
                                # Training mode
                                framework.train(train_samples, eval_samples[:100])  # Smaller dev set
                                metrics = framework.evaluate([], eval_samples)
                            else:
                                # Zero-shot evaluation
                                metrics = framework.evaluate(train_samples[:32], eval_samples)  # Few-shot
                            
                            results[config_name] = {
                                'metrics': metrics,
                                'config': {
                                    'model_type': model_type,
                                    'task_name': task_name,
                                    'hidden_size': hidden_size,
                                    'num_layers': num_layers,
                                    'attention_type': attention_type
                                }
                            }
                            
                            logger.info(f"Results for {config_name}: {metrics}")
                            
                        except Exception as e:
                            logger.error(f"Error running {config_name}: {e}")
                            results[config_name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Run benchmark
    print("Starting Qwen3 Reversible vs Standard Benchmark")
    print("=" * 60)
    
    results = run_qwen3_benchmark(
        model_types=["reversible", "standard"],
        tasks=["SST2"],  # Start with one task
        hidden_sizes=[512],  # Start with smaller size
        num_layers_list=[4],
        attention_types=["standard"]
    )
    
    # Save results
    import json
    with open("qwen3_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nBenchmark Results:")
    print("=" * 40)
    for config_name, result in results.items():
        if 'metrics' in result:
            print(f"{config_name}: {result['metrics']}")
        else:
            print(f"{config_name}: ERROR - {result.get('error', 'Unknown error')}")
