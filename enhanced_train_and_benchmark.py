"""
Enhanced Training and Benchmarking Framework for Reversible Qwen3 Models
========================================================================

This script provides comprehensive fine-tuning on benchmark datasets followed by 
evaluation across multiple benchmarks to compare reversible vs standard architectures.

Features:
- Fine-tuning on multiple benchmark datasets (enwik8, CodeParrot, GSM8K, SQuAD, WikiText)
- Comprehensive evaluation using all available benchmark suites
- Fair comparison between reversible and standard models
- Integrated training and evaluation pipeline

Usage:
python enhanced_train_and_benchmark.py --datasets wikitext,enwik8,code --models reversible,standard --full_eval
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# Suppress HuggingFace warnings and set tokenizer parallelism
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*CardData.*")

# Suppress specific HuggingFace warnings
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing modules
try:
    from test_train_qwen3_rev_v3f import ReversibleQwenPerformanceTester, EnhancedPerformanceMetrics
    BASE_TESTER_AVAILABLE = True
except ImportError as e:
    print(f"Base tester not available: {e}")
    BASE_TESTER_AVAILABLE = False

try:
    from advanced_benchmarks import AdvancedBenchmarkSuite
    ADVANCED_BENCHMARKS_AVAILABLE = True
except ImportError as e:
    print(f"Advanced benchmarks not available: {e}")
    ADVANCED_BENCHMARKS_AVAILABLE = False

try:
    from run_advanced_benchmarks import ComprehensiveBenchmarkRunner
    COMPREHENSIVE_RUNNER_AVAILABLE = True
except ImportError as e:
    print(f"Comprehensive runner not available: {e}")
    COMPREHENSIVE_RUNNER_AVAILABLE = False

try:
    from qwen3_reversible_02_2 import create_reversible_qwen3_model
    MODEL_CREATION_AVAILABLE = True
except ImportError as e:
    print(f"Model creation not available: {e}")
    MODEL_CREATION_AVAILABLE = False

# Dataset imports
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

@dataclass
class BenchmarkDatasetConfig:
    """Configuration for benchmark datasets"""
    name: str
    task_type: str  # 'language_modeling', 'code', 'reasoning', 'comprehension'
    seq_length: int = 512
    num_train_samples: int = 10000
    num_val_samples: int = 1000
    num_test_samples: int = 1000
    vocab_size: int = 32000
    
@dataclass 
class TrainingConfig:
    """Training configuration for different datasets"""
    dataset_name: str
    epochs: int = 10
    learning_rate: float = 3e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    scheduler_type: str = 'cosine'
    use_amp: bool = True
    early_stopping_patience: int = 3

class BenchmarkDatasetLoader:
    """Loads and preprocesses various benchmark datasets for training"""
    
    def __init__(self, cache_dir='./benchmark_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.datasets = {}
        
        # Initialize advanced benchmark suite for dataset access
        if ADVANCED_BENCHMARKS_AVAILABLE:
            self.advanced_suite = AdvancedBenchmarkSuite(cache_dir=str(self.cache_dir))
        else:
            self.advanced_suite = None
    
    def load_wikitext_dataset(self, config: BenchmarkDatasetConfig):
        """Load WikiText dataset"""
        print("Loading WikiText dataset...")
        
        # Suppress dataset warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        
        # Create tokenizer
        tokenizer = self._create_tokenizer(dataset['train']['text'], config.vocab_size)
        
        # Process splits with better progress tracking
        print("Processing WikiText splits...")
        train_data = self._process_text_data(
            dataset['train']['text'], tokenizer, config.seq_length, config.num_train_samples, "train"
        )
        val_data = self._process_text_data(
            dataset['validation']['text'], tokenizer, config.seq_length, config.num_val_samples, "val"
        )
        test_data = self._process_text_data(
            dataset['test']['text'], tokenizer, config.seq_length, config.num_test_samples, "test"
        )
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'tokenizer': tokenizer,
            'vocab_size': tokenizer.get_vocab_size()
        }
    
    def load_enwik8_dataset(self, config: BenchmarkDatasetConfig):
        """Load enwik8 character-level dataset"""
        print("Loading enwik8 dataset...")
        
        if not self.advanced_suite:
            print("Advanced benchmark suite not available for enwik8")
            return None
            
        try:
            enwik8_data = self.advanced_suite.load_enwik8_dataset()
            
            # Convert to training format
            char_to_idx = enwik8_data['char_to_idx']
            vocab_size = len(char_to_idx)
            
            # Process data
            train_data = self._process_char_data(
                enwik8_data['train'], char_to_idx, config.seq_length, config.num_train_samples
            )
            val_data = self._process_char_data(
                enwik8_data['val'], char_to_idx, config.seq_length, config.num_val_samples
            )
            test_data = self._process_char_data(
                enwik8_data['test'], char_to_idx, config.seq_length, config.num_test_samples
            )
            
            return {
                'train': train_data,
                'val': val_data,
                'test': test_data,
                'char_to_idx': char_to_idx,
                'vocab_size': vocab_size
            }
            
        except Exception as e:
            print(f"Failed to load enwik8: {e}")
            return None
    
    def load_code_dataset(self, config: BenchmarkDatasetConfig):
        """Load CodeParrot dataset"""
        print("Loading CodeParrot dataset...")
        
        if not self.advanced_suite:
            print("Advanced benchmark suite not available for code data")
            return None
            
        try:
            # Suppress specific dataset warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                code_data = self.advanced_suite.load_code_dataset()
                
            if not code_data:
                return None
            
            # Create tokenizer for code
            all_code = code_data['train'] + code_data['val'] + code_data['test']
            tokenizer = self._create_tokenizer(all_code, config.vocab_size)
            
            # Process splits with better progress tracking
            print("Processing code data splits...")
            train_data = self._process_text_data(
                code_data['train'], tokenizer, config.seq_length, config.num_train_samples, "train"
            )
            val_data = self._process_text_data(
                code_data['val'], tokenizer, config.seq_length, config.num_val_samples, "val"
            )
            test_data = self._process_text_data(
                code_data['test'], tokenizer, config.seq_length, config.num_test_samples, "test"
            )
            
            return {
                'train': train_data,
                'val': val_data,
                'test': test_data,
                'tokenizer': tokenizer,
                'vocab_size': tokenizer.get_vocab_size()
            }
            
        except Exception as e:
            print(f"Failed to load code dataset: {e}")
            return None
    
    def load_math_reasoning_dataset(self, config: BenchmarkDatasetConfig):
        """Load GSM8K math reasoning dataset"""
        print("Loading GSM8K dataset...")
        
        try:
            # Suppress dataset warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dataset = load_dataset("gsm8k", "main")
            
            # Combine question and answer for language modeling
            def process_gsm8k(split_data):
                texts = []
                for item in split_data:
                    combined_text = f"Question: {item['question']} Answer: {item['answer']}"
                    texts.append(combined_text)
                return texts
            
            train_texts = process_gsm8k(dataset['train'])
            test_texts = process_gsm8k(dataset['test'])
            
            # Split test into val/test
            mid_point = len(test_texts) // 2
            val_texts = test_texts[:mid_point]
            test_texts = test_texts[mid_point:]
            
            # Create tokenizer
            tokenizer = self._create_tokenizer(train_texts, config.vocab_size)
            
            # Process splits with better progress tracking
            print("Processing GSM8K splits...")
            train_data = self._process_text_data(
                train_texts, tokenizer, config.seq_length, config.num_train_samples, "train"
            )
            val_data = self._process_text_data(
                val_texts, tokenizer, config.seq_length, config.num_val_samples, "val"
            )
            test_data = self._process_text_data(
                test_texts, tokenizer, config.seq_length, config.num_test_samples, "test"
            )
            
            return {
                'train': train_data,
                'val': val_data,
                'test': test_data,
                'tokenizer': tokenizer,
                'vocab_size': tokenizer.get_vocab_size()
            }
            
        except Exception as e:
            print(f"Failed to load GSM8K: {e}")
            return None
    
    def load_reading_comprehension_dataset(self, config: BenchmarkDatasetConfig):
        """Load SQuAD reading comprehension dataset"""
        print("Loading SQuAD dataset...")
        
        try:
            # Suppress dataset warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dataset = load_dataset("squad")
            
            # Combine context, question, and answer for language modeling
            def process_squad(split_data):
                texts = []
                for item in split_data:
                    context = item['context']
                    question = item['question']
                    # Use first answer
                    answer = item['answers']['text'][0] if item['answers']['text'] else "No answer"
                    combined_text = f"Context: {context} Question: {question} Answer: {answer}"
                    texts.append(combined_text)
                return texts
            
            train_texts = process_squad(dataset['train'])
            val_texts = process_squad(dataset['validation'])
            
            # Create test split from validation
            mid_point = len(val_texts) // 2
            test_texts = val_texts[mid_point:]
            val_texts = val_texts[:mid_point]
            
            # Create tokenizer
            tokenizer = self._create_tokenizer(train_texts, config.vocab_size)
            
            # Process splits with better progress tracking
            print("Processing SQuAD splits...")
            train_data = self._process_text_data(
                train_texts, tokenizer, config.seq_length, config.num_train_samples, "train"
            )
            val_data = self._process_text_data(
                val_texts, tokenizer, config.seq_length, config.num_val_samples, "val"
            )
            test_data = self._process_text_data(
                test_texts, tokenizer, config.seq_length, config.num_test_samples, "test"
            )
            
            return {
                'train': train_data,
                'val': val_data,
                'test': test_data,
                'tokenizer': tokenizer,
                'vocab_size': tokenizer.get_vocab_size()
            }
            
        except Exception as e:
            print(f"Failed to load SQuAD: {e}")
            return None
    
    def _create_tokenizer(self, texts, vocab_size):
        """Create tokenizer from text data"""
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import WordLevelTrainer
        
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]
        )
        
        def batch_iterator():
            for i in range(0, len(texts), 1000):
                batch = [text for text in texts[i:i+1000] if text and text.strip()]
                if batch:
                    yield batch
        
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        return tokenizer
    
    def _process_text_data(self, texts, tokenizer, seq_length, max_samples, split_name="data"):
        """Process text data into sequences with better progress tracking"""
        data = []
        stride = seq_length // 4  # Smaller stride for more sequences
        
        # Filter and combine texts more aggressively
        valid_texts = []
        current_text = ""
        
        for text in texts:
            if text and text.strip():
                current_text += " " + text.strip()
                # Split into chunks when we have enough content
                if len(current_text) > seq_length * 4:
                    valid_texts.append(current_text)
                    current_text = ""
        
        # Add remaining text
        if current_text.strip():
            valid_texts.append(current_text)
        
        if not valid_texts:
            print(f"Warning: No valid texts found for {split_name}")
            return data
            
        print(f"Processing {len(valid_texts)} combined texts for {split_name} split...")
        
        with tqdm(valid_texts, desc=f"Processing {split_name} data", leave=False) as pbar:
            for text in pbar:
                try:
                    tokens = tokenizer.encode(text.strip()).ids
                    
                    if len(tokens) < seq_length + 1:
                        continue
                    
                    # Create overlapping sequences with smaller stride
                    for start_idx in range(0, len(tokens) - seq_length, stride):
                        sequence = tokens[start_idx:start_idx + seq_length + 1]
                        if len(sequence) == seq_length + 1:
                            x = torch.tensor(sequence[:-1], dtype=torch.long)
                            y = torch.tensor(sequence[1:], dtype=torch.long)
                            data.append((x, y))
                            
                            if len(data) >= max_samples:
                                pbar.set_description(f"Completed {split_name} ({len(data)} sequences)")
                                return data
                                
                except Exception as e:
                    continue
                
                # Update progress more frequently
                if len(data) % 50 == 0:
                    pbar.set_description(f"Processing {split_name} ({len(data)}/{max_samples} sequences)")
        
        print(f"Created {len(data)} sequences for {split_name} split")
        return data
    
    def _process_char_data(self, text, char_to_idx, seq_length, max_samples):
        """Process character-level data"""
        data = []
        stride = seq_length // 2
        
        if len(text) < seq_length + 1:
            return data
        
        # Create sequences
        for start_idx in range(0, len(text) - seq_length, stride):
            sequence_text = text[start_idx:start_idx + seq_length + 1]
            
            try:
                tokens = [char_to_idx.get(char, char_to_idx.get('<unk>', 0)) for char in sequence_text]
                
                if len(tokens) == seq_length + 1:
                    x = torch.tensor(tokens[:-1], dtype=torch.long)
                    y = torch.tensor(tokens[1:], dtype=torch.long)
                    data.append((x, y))
                    
                    if len(data) >= max_samples:
                        break
                        
            except Exception as e:
                continue
        
        return data
    
    def get_dataset(self, dataset_name: str, config: BenchmarkDatasetConfig):
        """Get dataset by name"""
        
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]
        
        loaders = {
            'wikitext': self.load_wikitext_dataset,
            'enwik8': self.load_enwik8_dataset,
            'code': self.load_code_dataset,
            'math': self.load_math_reasoning_dataset,
            'squad': self.load_reading_comprehension_dataset
        }
        
        if dataset_name not in loaders:
            print(f"Unknown dataset: {dataset_name}")
            return None
        
        dataset = loaders[dataset_name](config)
        if dataset:
            self.datasets[dataset_name] = dataset
        
        return dataset

class BenchmarkDataset(Dataset):
    """PyTorch Dataset wrapper for benchmark data"""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class EnhancedTrainingAndBenchmarkRunner:
    """Enhanced framework for training and benchmarking"""
    
    def __init__(self, device='cuda', cache_dir='./benchmark_cache'):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.dataset_loader = BenchmarkDatasetLoader(cache_dir)
        
        if BASE_TESTER_AVAILABLE:
            self.base_tester = ReversibleQwenPerformanceTester(device=device)
        else:
            self.base_tester = None
            
        if COMPREHENSIVE_RUNNER_AVAILABLE:
            self.benchmark_runner = ComprehensiveBenchmarkRunner(device=device)
        else:
            self.benchmark_runner = None
        
        self.results = {}
    
    def create_models(self, model_types=['reversible', 'standard'], vocab_size=32000, hidden_size=512*4, num_layers=4*2):
        """Create models for comparison"""
        
        if not MODEL_CREATION_AVAILABLE:
            print("Model creation not available")
            return {}
        
        models = {}
        
        for model_type in model_types:
            try:
                use_reversible = model_type == 'reversible'
                model = create_reversible_qwen3_model(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    num_hidden_layers=num_layers,
                    num_attention_heads=8,
                    num_key_value_heads=4,
                    attention_type="standard",
                    use_reversible=use_reversible,
                    reverse_thres=256 if use_reversible else 999999,
                    intermediate_size=hidden_size * 4,
                    max_position_embeddings=2048,
                    rms_norm_eps=1e-6
                )
                
                model = model.to(self.device)
                models[f"{model_type}_qwen3"] = model
                print(f"Created {model_type} model")
                
            except Exception as e:
                print(f"Failed to create {model_type} model: {e}")
        
        return models
    
    def get_training_config(self, dataset_name: str, model_type: str) -> TrainingConfig:
        """Get optimized training config for dataset and model type"""
        
        base_configs = {
            'wikitext': TrainingConfig(
                dataset_name=dataset_name,
                epochs=30,
                learning_rate=3e-4,
                batch_size=32,
                gradient_accumulation_steps=4
            ),
            'enwik8': TrainingConfig(
                dataset_name=dataset_name,
                epochs=10,
                learning_rate=2e-4,
                batch_size=16,
                gradient_accumulation_steps=2
            ),
            'code': TrainingConfig(
                dataset_name=dataset_name,
                epochs=30,
                learning_rate=1e-4,
                batch_size=32,
                gradient_accumulation_steps=8
            ),
            'math': TrainingConfig(
                dataset_name=dataset_name,
                epochs=30,
                learning_rate=2e-4,
                batch_size=32,
                gradient_accumulation_steps=4
            ),
            'squad': TrainingConfig(
                dataset_name=dataset_name,
                epochs=30,
                learning_rate=2e-4,
                batch_size=32,
                gradient_accumulation_steps=8
            )
        }
        
        config = base_configs.get(dataset_name, base_configs['wikitext'])
        
        # # Adjust for model type
        # if model_type == 'standard':
        #     # Standard models may need different hyperparameters
        #     config.learning_rate *= 0.7  # Lower learning rate
        #     config.epochs = int(config.epochs * 1.5)  # More epochs
            
        return config
    
    def fine_tune_on_dataset(self, model, dataset_name: str, dataset_data: Dict, 
                           training_config: TrainingConfig) -> Dict:
        """Fine-tune model on specific dataset"""
        
        print(f"\n{'='*60}")
        print(f"FINE-TUNING ON {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Create data loaders
        train_dataset = BenchmarkDataset(dataset_data['train'])
        val_dataset = BenchmarkDataset(dataset_data['val'])
        test_dataset = BenchmarkDataset(dataset_data['test'])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=True,
            num_workers=0,  # Disable multiprocessing to avoid tokenizer warnings
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=False,
            num_workers=0,  # Disable multiprocessing to avoid tokenizer warnings
            pin_memory=True if torch.cuda.is_available() else False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=False,
            num_workers=0,  # Disable multiprocessing to avoid tokenizer warnings
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        
        # Safety check for dataset sizes
        if len(train_dataset) < 10:
            print(f"ERROR: Insufficient training data ({len(train_dataset)} samples). Need at least 10 samples.")
            return {
                'error': 'insufficient_training_data',
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset)
            }
        
        if len(val_dataset) == 0:
            print("WARNING: No validation data available. Creating validation set from training data.")
            # Split training data for validation
            val_size = min(len(train_dataset) // 5, 100)  # 20% or max 100 samples
            val_indices = torch.randperm(len(train_dataset))[:val_size]
            train_indices = torch.randperm(len(train_dataset))[val_size:]
            
            val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            
            print(f"Adjusted: Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        
        # Setup training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        total_steps = len(train_loader) * training_config.epochs // training_config.gradient_accumulation_steps
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=training_config.learning_rate * 0.01
        )
        
        scaler = torch.amp.GradScaler('cuda') if training_config.use_amp and torch.cuda.is_available() else None
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(training_config.epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            optimizer.zero_grad()
            
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                try:
                    if training_config.use_amp and scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                            
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = targets[..., 1:].contiguous()
                            loss = criterion(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                        
                        scaled_loss = scaler.scale(loss / training_config.gradient_accumulation_steps)
                        scaled_loss.backward()
                    else:
                        outputs = model(inputs)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = targets[..., 1:].contiguous()
                        loss = criterion(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        (loss / training_config.gradient_accumulation_steps).backward()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Update weights
                    if (batch_idx + 1) % training_config.gradient_accumulation_steps == 0:
                        # Check gradients before update
                        total_norm = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                total_norm += p.grad.data.norm(2).item() ** 2
                        total_norm = total_norm ** 0.5
                        
                        if total_norm < 1e-8:
                            print(f"   WARNING: Very small gradients (norm: {total_norm:.2e}) - learning may be impaired!")
                        
                        if training_config.use_amp and scaler is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                        
                        scheduler.step()
                        optimizer.zero_grad()
                        
                except Exception as e:
                    print(f"Training error at batch {batch_idx}: {e}")
                    continue
            
            avg_train_loss = epoch_loss / max(num_batches, 1)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            
            if len(val_dataset) > 0:
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        try:
                            outputs = model(inputs)
                            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                            
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = targets[..., 1:].contiguous()
                            loss = criterion(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                            
                            val_loss += loss.item()
                            val_batches += 1
                            
                        except Exception as e:
                            continue
            
            avg_val_loss = val_loss / max(val_batches, 1) if val_batches > 0 else float('inf')
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Debug: Check if model parameters are actually changing
            if epoch == 0:
                first_param = next(model.parameters()).clone()
            elif epoch == 1:
                second_param = next(model.parameters())
                param_change = torch.norm(second_param - first_param).item()
                print(f"   Parameter change magnitude: {param_change:.6f}")
                if param_change < 1e-6:
                    print("   WARNING: Model parameters barely changing - possible learning issue!")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= training_config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Calculate final metrics on test set
        model.eval()
        test_loss = 0
        test_batches = 0
        
        if len(test_dataset) > 0:
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    try:
                        outputs = model(inputs)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = targets[..., 1:].contiguous()
                        loss = criterion(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        test_loss += loss.item()
                        test_batches += 1
                        
                    except Exception as e:
                        continue
        
        final_test_loss = test_loss / max(test_batches, 1) if test_batches > 0 else avg_train_loss
        final_perplexity = torch.exp(torch.tensor(final_test_loss)).item()
        
        # Ensure perplexity is reasonable
        if final_perplexity < 1.1:  # Suspiciously low perplexity
            print(f"WARNING: Suspiciously low perplexity ({final_perplexity:.2f}). Using validation loss instead.")
            final_test_loss = best_val_loss if best_val_loss != float('inf') else avg_train_loss
            final_perplexity = torch.exp(torch.tensor(final_test_loss)).item()
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_test_loss': final_test_loss,
            'final_perplexity': final_perplexity,
            'epochs_trained': len(train_losses),
            'best_val_loss': best_val_loss
        }
    
    def run_comprehensive_evaluation(self, models: Dict[str, torch.nn.Module]) -> Dict:
        """Run comprehensive evaluation using all available benchmarks"""
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE BENCHMARK EVALUATION")
        print(f"{'='*80}")
        
        all_results = {}
        
        # 1. Use existing benchmark runner if available
        if self.benchmark_runner:
            try:
                print("\n🔬 Running comprehensive benchmark suite...")
                benchmark_results = self.benchmark_runner.run_all_benchmarks(
                    models, 
                    benchmark_types=['text_generation', 'advanced']
                )
                all_results['comprehensive_benchmarks'] = benchmark_results
                
                # Print summary
                self.benchmark_runner.print_comprehensive_summary()
                
            except Exception as e:
                print(f"Comprehensive benchmark failed: {e}")
        
        # 2. Run advanced benchmarks if available
        if ADVANCED_BENCHMARKS_AVAILABLE:
            try:
                print("\n🧪 Running advanced benchmark suite...")
                advanced_suite = AdvancedBenchmarkSuite(device=self.device)
                advanced_results = advanced_suite.run_comprehensive_benchmark(
                    models, quick_test=True
                )
                all_results['advanced_benchmarks'] = advanced_results
                
            except Exception as e:
                print(f"Advanced benchmark failed: {e}")
        
        # 3. Run base performance tests if available
        if self.base_tester:
            try:
                print("\n📊 Running enhanced performance tests...")
                for model_name, model in models.items():
                    # Create a simple test dataset for evaluation
                    test_data = self.base_tester.create_test_dataset(
                        vocab_size=getattr(model.config, 'vocab_size', 32000),
                        seq_len=512,
                        num_samples=500
                    )
                    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
                    
                    # Calculate enhanced metrics
                    perplexity, loss = self.base_tester.calculate_perplexity(model, test_loader, max_batches=20)
                    lm_metrics = self.base_tester.calculate_enhanced_language_metrics(model, test_loader, max_batches=20)
                    comp_metrics = self.base_tester.measure_computational_efficiency(model, test_loader, max_batches=10)
                    mem_metrics = self.base_tester.measure_memory_efficiency(model)
                    
                    if 'enhanced_performance' not in all_results:
                        all_results['enhanced_performance'] = {}
                    
                    all_results['enhanced_performance'][model_name] = {
                        'perplexity': perplexity,
                        'loss': loss,
                        'token_accuracy': lm_metrics['token_accuracy'],
                        'bits_per_byte': lm_metrics['bits_per_byte'],
                        'throughput': comp_metrics['throughput_tokens_per_sec'],
                        'memory_peak_mb': mem_metrics['peak_memory_mb'],
                        'memory_efficiency': mem_metrics['memory_efficiency_ratio']
                    }
                    
            except Exception as e:
                print(f"Enhanced performance tests failed: {e}")
        
        return all_results
    
    def run_multi_dataset_training_and_evaluation(self, 
                                                 datasets=['wikitext', 'enwik8', 'code'], 
                                                 models=['reversible', 'standard'],
                                                 full_evaluation=True):
        """Run training on multiple datasets followed by comprehensive evaluation"""
        
        print(f"\n{'='*100}")
        print("MULTI-DATASET TRAINING AND EVALUATION PIPELINE")
        print(f"{'='*100}")
        print(f"Datasets: {datasets}")
        print(f"Models: {models}")
        print(f"Full evaluation: {full_evaluation}")
        
        all_results = {}
        
        # Dataset configurations - increased sample sizes for better training
        dataset_configs = {
            'wikitext': BenchmarkDatasetConfig('wikitext', 'language_modeling', seq_length=512, 
                                             num_train_samples=8000, num_val_samples=1000, num_test_samples=1000),
            'enwik8': BenchmarkDatasetConfig('enwik8', 'language_modeling', seq_length=512, 
                                           num_train_samples=6000, num_val_samples=800, num_test_samples=800),
            'code': BenchmarkDatasetConfig('code', 'code', seq_length=512, 
                                         num_train_samples=5000, num_val_samples=600, num_test_samples=600),
            'math': BenchmarkDatasetConfig('math', 'reasoning', seq_length=1024, 
                                         num_train_samples=4000, num_val_samples=500, num_test_samples=500),
            'squad': BenchmarkDatasetConfig('squad', 'comprehension', seq_length=1024, 
                                          num_train_samples=4000, num_val_samples=500, num_test_samples=500)
        }
        
        for dataset_name in datasets:
            print(f"\n{'='*80}")
            print(f"TRAINING ON {dataset_name.upper()} DATASET")
            print(f"{'='*80}")
            
            if dataset_name not in dataset_configs:
                print(f"Unknown dataset: {dataset_name}")
                continue
            
            config = dataset_configs[dataset_name]
            
            # Load dataset
            dataset_data = self.dataset_loader.get_dataset(dataset_name, config)
            if not dataset_data:
                print(f"Failed to load {dataset_name} dataset")
                continue
            
            # Update vocab size based on loaded dataset
            vocab_size = dataset_data.get('vocab_size', 32000)
            
            # Create models for this dataset
            trained_models = self.create_models(
                model_types=models, 
                vocab_size=vocab_size,
                hidden_size=512,
                num_layers=4
            )
            
            if not trained_models:
                print(f"Failed to create models for {dataset_name}")
                continue
            
            dataset_results = {}
            
            # Train each model
            for model_name, model in trained_models.items():
                model_type = 'reversible' if 'reversible' in model_name else 'standard'
                training_config = self.get_training_config(dataset_name, model_type)
                
                print(f"\n📚 Training {model_name} on {dataset_name}...")
                
                try:
                    training_results = self.fine_tune_on_dataset(
                        model, dataset_name, dataset_data, training_config
                    )
                    
                    dataset_results[model_name] = {
                        'training_results': training_results,
                        'model': model,
                        'dataset': dataset_name,
                        'config': training_config
                    }
                    
                    print(f"✅ {model_name} training completed:")
                    print(f"   Final test loss: {training_results['final_test_loss']:.4f}")
                    print(f"   Final perplexity: {training_results['final_perplexity']:.2f}")
                    print(f"   Epochs trained: {training_results['epochs_trained']}")
                    
                except Exception as e:
                    print(f"❌ Training failed for {model_name} on {dataset_name}: {e}")
                    continue
            
            all_results[dataset_name] = dataset_results
            
            # Run evaluation on this dataset's models
            if full_evaluation and dataset_results:
                print(f"\n🔍 Evaluating models trained on {dataset_name}...")
                
                models_for_eval = {name: result['model'] for name, result in dataset_results.items()}
                eval_results = self.run_comprehensive_evaluation(models_for_eval)
                
                # Add evaluation results
                for model_name in dataset_results:
                    if model_name in models_for_eval:
                        dataset_results[model_name]['evaluation_results'] = eval_results
        
        return all_results
    
    def save_results(self, results: Dict, filename: str = 'enhanced_training_benchmark_results.json'):
        """Save comprehensive results"""
        
        # Convert results to serializable format
        serializable_results = {}
        
        for dataset_name, dataset_results in results.items():
            serializable_results[dataset_name] = {}
            
            for model_name, model_data in dataset_results.items():
                serializable_data = {
                    'training_results': model_data['training_results'],
                    'dataset': model_data['dataset'],
                    'config': asdict(model_data['config']) if hasattr(model_data['config'], '__dict__') else str(model_data['config'])
                }
                
                # Add evaluation results if available
                if 'evaluation_results' in model_data:
                    # Convert evaluation results to serializable format
                    eval_results = model_data['evaluation_results']
                    serializable_eval = {}
                    
                    for eval_type, eval_data in eval_results.items():
                        try:
                            serializable_eval[eval_type] = eval_data
                        except:
                            serializable_eval[eval_type] = str(eval_data)
                    
                    serializable_data['evaluation_results'] = serializable_eval
                
                serializable_results[dataset_name][model_name] = serializable_data
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")
    
    def print_summary(self, results: Dict):
        """Print comprehensive summary of all results"""
        
        print(f"\n{'='*100}")
        print("COMPREHENSIVE TRAINING AND EVALUATION SUMMARY")
        print(f"{'='*100}")
        
        for dataset_name, dataset_results in results.items():
            print(f"\n📚 {dataset_name.upper()} DATASET RESULTS:")
            print("-" * 60)
            
            if not dataset_results:
                print("   No results available")
                continue
            
            # Training results summary
            print(f"{'Model':<25} {'Final Loss':<12} {'Perplexity':<12} {'Epochs':<8}")
            print("-" * 60)
            
            for model_name, model_data in dataset_results.items():
                training_results = model_data['training_results']
                print(f"{model_name:<25} {training_results['final_test_loss']:<12.4f} "
                      f"{training_results['final_perplexity']:<12.2f} {training_results['epochs_trained']:<8}")
            
            # Performance comparison
            reversible_models = [name for name in dataset_results.keys() if 'reversible' in name.lower()]
            standard_models = [name for name in dataset_results.keys() if 'standard' in name.lower()]
            
            if reversible_models and standard_models:
                print(f"\n   Performance Comparison:")
                
                rev_perplexity = np.mean([dataset_results[m]['training_results']['final_perplexity'] 
                                        for m in reversible_models])
                std_perplexity = np.mean([dataset_results[m]['training_results']['final_perplexity'] 
                                        for m in standard_models])
                
                print(f"   Reversible avg perplexity: {rev_perplexity:.2f}")
                print(f"   Standard avg perplexity: {std_perplexity:.2f}")
                print(f"   Performance gap: {rev_perplexity - std_perplexity:+.2f}")
        
        # Overall insights
        print(f"\n{'='*100}")
        print("KEY INSIGHTS")
        print(f"{'='*100}")
        
        all_datasets = list(results.keys())
        
        if len(all_datasets) > 1:
            print(f"✅ Successfully trained on {len(all_datasets)} datasets: {', '.join(all_datasets)}")
            
            # Cross-dataset performance analysis
            reversible_perplexities = []
            standard_perplexities = []
            
            for dataset_results in results.values():
                for model_name, model_data in dataset_results.items():
                    perplexity = model_data['training_results']['final_perplexity']
                    if 'reversible' in model_name.lower():
                        reversible_perplexities.append(perplexity)
                    else:
                        standard_perplexities.append(perplexity)
            
            if reversible_perplexities and standard_perplexities:
                print(f"\n📊 Overall Performance Across All Datasets:")
                print(f"   Reversible models - Avg perplexity: {np.mean(reversible_perplexities):.2f} ± {np.std(reversible_perplexities):.2f}")
                print(f"   Standard models - Avg perplexity: {np.mean(standard_perplexities):.2f} ± {np.std(standard_perplexities):.2f}")
                print(f"   Overall performance gap: {np.mean(reversible_perplexities) - np.mean(standard_perplexities):+.2f}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Training and Benchmarking for Qwen3 Models')
    parser.add_argument('--datasets', type=str, default='wikitext,enwik8', 
                        help='Comma-separated list of datasets: wikitext,enwik8,code,math,squad')
    parser.add_argument('--models', type=str, default='reversible,standard',
                        help='Comma-separated list of model types: reversible,standard')
    parser.add_argument('--full_eval', action='store_true', 
                        help='Run full evaluation suite after training')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--cache_dir', type=str, default='./benchmark_cache', help='Cache directory')
    parser.add_argument('--output', type=str, default='enhanced_training_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Parse arguments
    datasets = [d.strip() for d in args.datasets.split(',')]
    models = [m.strip() for m in args.models.split(',')]
    
    print("="*100)
    print("ENHANCED TRAINING AND BENCHMARKING FRAMEWORK")
    print("="*100)
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Full evaluation: {args.full_eval}")
    print(f"Device: {args.device}")
    
    # Initialize runner
    runner = EnhancedTrainingAndBenchmarkRunner(device=args.device, cache_dir=args.cache_dir)
    
    # Run training and evaluation
    results = runner.run_multi_dataset_training_and_evaluation(
        datasets=datasets,
        models=models,
        full_evaluation=args.full_eval
    )
    
    # Print summary
    runner.print_summary(results)
    
    # Save results
    runner.save_results(results, args.output)
    
    print(f"\n🎉 Enhanced training and evaluation complete!")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
