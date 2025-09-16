"""
Comprehensive Qwen3 vs DiZO Benchmark Suite
==========================================

This script provides a complete pipeline to:
1. Train reversible Qwen3 models using reversible_qwen3_02_2.py
2. Train standard Qwen3 models for comparison
3. Compare both against DiZO models on identical datasets and metrics
4. Use the same evaluation framework as DiZO (GLUE+, memory, advanced benchmarks)

Usage:
python comprehensive_qwen3_dizo_benchmark.py --scale small --datasets sst2,cola,mrpc --full_eval
python comprehensive_qwen3_dizo_benchmark.py --scale medium --datasets glue_basic --compare_dizo
python comprehensive_qwen3_dizo_benchmark.py --scale large --datasets all --full_eval --save_models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model creation functions
try:
    from qwen3_reversible_02_2 import create_reversible_qwen3_model, get_reversible_training_config
    MODEL_CREATION_AVAILABLE = True
    logger.info("✓ Reversible Qwen3 model creation available")
except ImportError as e:
    logger.error(f"✗ Model creation not available: {e}")
    MODEL_CREATION_AVAILABLE = False

# Import benchmark frameworks
try:
    from glue_plus_benchmark import GLUEPlusBenchmark, get_recommended_task_progression, GLUE_PLUS_TASKS
    GLUE_AVAILABLE = True
    logger.info("✓ GLUE+ benchmark available")
except ImportError as e:
    logger.warning(f"GLUE+ benchmark not available: {e}")
    GLUE_AVAILABLE = False

try:
    from memory_benchmark import MemoryBenchmark
    MEMORY_AVAILABLE = True
    logger.info("✓ Memory benchmark available")
except ImportError as e:
    logger.warning(f"Memory benchmark not available: {e}")
    MEMORY_AVAILABLE = False

try:
    from advanced_benchmarks import AdvancedBenchmarkSuite
    ADVANCED_AVAILABLE = True
    logger.info("✓ Advanced benchmark suite available")
except ImportError as e:
    logger.warning(f"Advanced benchmark suite not available: {e}")
    ADVANCED_AVAILABLE = False

try:
    from run_advanced_benchmarks import ComprehensiveBenchmarkRunner
    COMPREHENSIVE_AVAILABLE = True
    logger.info("✓ Comprehensive benchmark runner available")
except ImportError as e:
    logger.warning(f"Comprehensive benchmark runner not available: {e}")
    COMPREHENSIVE_AVAILABLE = False

# Import evaluation metrics
try:
    import evaluate
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoConfig
    from rouge_score import rouge_scorer
    import nltk
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
    METRICS_AVAILABLE = True
    logger.info("✓ Evaluation metrics available")
except ImportError as e:
    logger.warning(f"Evaluation metrics not available: {e}")
    METRICS_AVAILABLE = False

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    # Model configurations
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    max_seq_length: int = 2048
    
    # Training configurations
    batch_size: int = 16
    learning_rate: float = 2e-5  # Reduced learning rate for stability
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.1  # Add label smoothing
    
    # Evaluation configurations
    eval_batch_size: int = 32
    eval_steps: int = 500
    save_steps: int = 1000
    
    # Dataset configurations
    train_size: Optional[int] = None  # Use full dataset if None
    eval_size: Optional[int] = 2000   # Increased for better validation
    
    # Benchmark configurations
    run_glue: bool = True
    run_memory: bool = True
    run_advanced: bool = True
    run_generation: bool = True
    
    # Scale configurations
    scale: str = "small"  # small, medium, large
    
    def __post_init__(self):
        """Adjust configurations based on scale"""
        if self.scale == "small":
            self.num_layers = 6
            self.hidden_size = 512
            self.num_attention_heads = 8
            self.num_epochs = 2
            self.eval_size = 1500  # Increased validation size
        elif self.scale == "medium":
            self.num_layers = 12
            self.hidden_size = 768
            self.num_attention_heads = 12
            self.num_epochs = 3
            self.eval_size = 2000  # Increased validation size
        elif self.scale == "large":
            self.num_layers = 24
            self.hidden_size = 1024
            self.num_attention_heads = 16
            self.num_epochs = 5
            self.eval_size = 3000  # Increased validation size

class DiZODatasetCompatibilityLayer:
    """
    Dataset compatibility layer to match DiZO's data preprocessing and formats
    This ensures fair comparison by using identical data preprocessing
    """
    
    def __init__(self, tokenizer_name="bert-base-uncased", max_length=512):
        if METRICS_AVAILABLE:
            try:
                from transformers import AutoTokenizer
                # Try Qwen first only if explicitly requested
                if "qwen" in tokenizer_name.lower():
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
                        # Handle Qwen tokenizer pad token setup...
                        if self.tokenizer.pad_token is None:
                            vocab = self.tokenizer.get_vocab()
                            if vocab:
                                # Use exclamation mark as pad token for Qwen
                                if b'!' in vocab:
                                    self.tokenizer.pad_token = '!'
                                    self.tokenizer.pad_token_id = vocab[b'!']
                                else:
                                    self.tokenizer.pad_token = list(vocab.keys())[0].decode('utf-8', errors='ignore')
                                    self.tokenizer.pad_token_id = 0
                        logger.info(f"Using Qwen tokenizer with pad_token: {repr(self.tokenizer.pad_token)}")
                    except Exception as qwen_error:
                        logger.warning(f"Qwen tokenizer failed: {qwen_error}, falling back to BERT")
                        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    # Use standard tokenizer (BERT, etc.)
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
                
                logger.info(f"Tokenizer configured with pad_token: {repr(self.tokenizer.pad_token)}, pad_token_id: {self.tokenizer.pad_token_id}")
            except Exception as e:
                # Fallback to basic tokenizer
                self.tokenizer = None
                logger.warning(f"Using basic tokenization - install transformers for better results: {e}")
        else:
            self.tokenizer = None
            
        self.max_length = max_length
        self.datasets_cache = {}
    
    def load_glue_dataset(self, task_name: str, split: str = "train", limit: Optional[int] = None):
        """Load GLUE dataset in DiZO-compatible format"""
        if not METRICS_AVAILABLE:
            return None
            
        cache_key = f"{task_name}_{split}_{limit}"
        if cache_key in self.datasets_cache:
            return self.datasets_cache[cache_key]
        
        try:
            if task_name in GLUE_PLUS_TASKS:
                task_config = GLUE_PLUS_TASKS[task_name]
                dataset_name = task_config.dataset_name
                
                if task_name.startswith('superglue_'):
                    dataset = load_dataset(dataset_name.split('/')[0], dataset_name.split('/')[1], split=split)
                else:
                    dataset = load_dataset(dataset_name, split=split)
                
                if limit:
                    dataset = dataset.select(range(min(limit, len(dataset))))
                
                # Convert to compatible format
                processed_data = self._process_dataset(dataset, task_name, task_config)
                self.datasets_cache[cache_key] = processed_data
                return processed_data
                
        except Exception as e:
            logger.error(f"Failed to load {task_name} dataset: {e}")
            return None
    
    def _process_dataset(self, dataset, task_name: str, task_config):
        """Process dataset to match DiZO's format"""
        processed_examples = []
        
        for example in dataset:
            processed_example = self._process_example(example, task_name, task_config)
            if processed_example:
                processed_examples.append(processed_example)
        
        return processed_examples
    
    def _process_example(self, example, task_name: str, task_config):
        """Process individual example to match DiZO's format"""
        try:
            if task_name == 'sst2':
                text = example['sentence']
                label = example['label']
                
            elif task_name == 'cola':
                text = example['sentence']
                label = example['label']
                
            elif task_name == 'mrpc':
                text = f"{example['sentence1']} [SEP] {example['sentence2']}"
                label = example['label']
                
            elif task_name == 'qqp':
                text = f"{example['question1']} [SEP] {example['question2']}"
                label = example['label']
                
            elif task_name == 'mnli':
                text = f"{example['premise']} [SEP] {example['hypothesis']}"
                label = example['label']
                
            else:
                # Generic handling
                text_fields = [k for k in example.keys() if 'sentence' in k or 'text' in k or 'premise' in k or 'hypothesis' in k]
                if len(text_fields) == 1:
                    text = example[text_fields[0]]
                else:
                    text = " [SEP] ".join([example[f] for f in text_fields[:2]])
                label = example.get('label', 0)
            
            # Tokenize if tokenizer available
            if self.tokenizer:
                tokens = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return {
                    'input_ids': tokens['input_ids'].squeeze(),
                    'attention_mask': tokens['attention_mask'].squeeze(),
                    'label': torch.tensor(label, dtype=torch.long),
                    'text': text
                }
            else:
                return {
                    'text': text,
                    'label': torch.tensor(label, dtype=torch.long)
                }
                
        except Exception as e:
            logger.warning(f"Failed to process example: {e}")
            return None

class ModelTrainer:
    """Unified trainer for both reversible and standard models"""
    
    def __init__(self, config: BenchmarkConfig, device='cuda'):
        self.config = config
        # Handle device properly (could be string or torch.device)
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.data_loader = DiZODatasetCompatibilityLayer()
        
    def create_models(self) -> Dict[str, nn.Module]:
        """Create both reversible and standard Qwen3 models"""
        if not MODEL_CREATION_AVAILABLE:
            logger.error("Model creation not available!")
            return {}
        
        models = {}
        
        try:
            # Create reversible model
            logger.info("Creating reversible Qwen3 model...")
            reversible_model = create_reversible_qwen3_model(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_hidden_layers=self.config.num_layers,
                num_attention_heads=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_attention_heads // 2,
                attention_type="standard",
                use_reversible=True,
                reverse_thres=256,  # Use reversible for sequences > 256
                candidate_pr_ratio=0.7,
                candidate_top_k=32
            )
            
            # Initialize weights
            self._initialize_model_weights(reversible_model)
            models['reversible_qwen3'] = reversible_model.to(self.device)
            
            # Create standard model (same architecture, no reversible layers)
            logger.info("Creating standard Qwen3 model...")
            standard_model = create_reversible_qwen3_model(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_hidden_layers=self.config.num_layers,
                num_attention_heads=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_attention_heads // 2,
                attention_type="standard",
                use_reversible=False,  # No reversible layers
                reverse_thres=999999,  # Never use reversible
                candidate_pr_ratio=0.7,
                candidate_top_k=32
            )
            
            self._initialize_model_weights(standard_model)
            models['standard_qwen3'] = standard_model.to(self.device)
            
            # Log model information
            for name, model in models.items():
                param_count = sum(p.numel() for p in model.parameters())
                logger.info(f"Created {name}: {param_count:,} parameters")
                
        except Exception as e:
            logger.error(f"Failed to create models: {e}")
            
        return models
    
    def _initialize_model_weights(self, model):
        """Initialize model weights properly"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        
        model.apply(init_weights)
    
    def train_model_on_task(self, model: nn.Module, task_name: str, model_name: str) -> Dict:
        """Train a model on a specific GLUE task"""
        logger.info(f"Training {model_name} on {task_name}")
        
        # Load dataset
        train_data = self.data_loader.load_glue_dataset(task_name, 'train', self.config.train_size)
        val_data = self.data_loader.load_glue_dataset(task_name, 'validation', self.config.eval_size)
        
        if not train_data or not val_data:
            logger.error(f"Failed to load data for {task_name}")
            return {}
        
        logger.info(f"Dataset sizes - Train: {len(train_data)}, Validation: {len(val_data)}")
        
        # Check label distribution
        train_labels = [item['label'].item() if torch.is_tensor(item['label']) else item['label'] for item in train_data]
        val_labels = [item['label'].item() if torch.is_tensor(item['label']) else item['label'] for item in val_data]
        
        from collections import Counter
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        logger.info(f"Train label distribution: {dict(train_dist)}")
        logger.info(f"Validation label distribution: {dict(val_dist)}")
        
        # Create task-specific model (add classification head)
        task_config = GLUE_PLUS_TASKS[task_name]
        classifier = self._create_classification_model(model, task_config.num_labels)
        
        # Create data loaders first
        train_loader = self._create_dataloader(train_data, self.config.batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_data, self.config.eval_batch_size, shuffle=False)
        
        # Setup training
        optimizer = optim.AdamW(
            classifier.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Add learning rate scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio
        )
        
        # Training loop with early stopping
        best_metric = 0
        patience = 3
        patience_counter = 0
        training_results = {
            'train_losses': [],
            'val_metrics': [],
            'best_metric': 0,
            'epochs_trained': 0
        }
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_loss = self._train_epoch(classifier, train_loader, optimizer, scheduler)
            training_results['train_losses'].append(train_loss)
            
            # Evaluate
            val_metric = self._evaluate_model(classifier, val_loader, task_config.metric)
            training_results['val_metrics'].append(val_metric)
            
            # Early stopping logic
            if val_metric > best_metric:
                best_metric = val_metric
                training_results['best_metric'] = best_metric
                patience_counter = 0
            else:
                patience_counter += 1
            
            training_results['epochs_trained'] = epoch + 1
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val {task_config.metric}: {val_metric:.4f}, Best: {best_metric:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        return training_results
    
    def _create_classification_model(self, base_model: nn.Module, num_labels: int):
        """Create classification model with task-specific head"""
        class QwenClassificationWrapper(nn.Module):
            def __init__(self, qwen_model, hidden_size, num_labels):
                super().__init__()
                self.qwen_model = qwen_model
                self.dropout = nn.Dropout(0.3)  # Increased dropout
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.LayerNorm(hidden_size // 2),  # Added normalization
                    nn.ReLU(),
                    nn.Dropout(0.2),  # Additional dropout
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.LayerNorm(hidden_size // 4),  # Added normalization
                    nn.ReLU(), 
                    nn.Dropout(0.1),  # Final dropout
                    nn.Linear(hidden_size // 4, num_labels)
                )
                
                # Initialize classifier weights
                self._init_classifier_weights()
            
            def _init_classifier_weights(self):
                """Initialize classifier head weights"""
                for module in self.classifier.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
            
            def forward(self, input_ids, attention_mask=None):
                # Get output from Qwen model
                outputs = self.qwen_model(input_ids)
                
                # Extract hidden states (use last token or mean pooling)
                if isinstance(outputs, dict):
                    hidden_states = outputs['hidden_states']  # [batch_size, seq_len, hidden_size]
                else:
                    hidden_states = outputs
                
                # Use mean pooling over sequence length
                if attention_mask is not None:
                    # Apply attention mask for mean pooling
                    attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                    hidden_states = hidden_states * attention_mask
                    pooled = hidden_states.sum(dim=1) / attention_mask.sum(dim=1)
                else:
                    # Simple mean pooling without mask
                    pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
                
                # Apply dropout and classification head
                pooled = self.dropout(pooled)
                logits = self.classifier(pooled)
                
                return logits
        
        return QwenClassificationWrapper(base_model, self.config.hidden_size, num_labels).to(self.device)
    
    def _create_dataloader(self, data: List, batch_size: int, shuffle: bool = True):
        """Create PyTorch DataLoader from processed data"""
        class TaskDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        return DataLoader(TaskDataset(data), batch_size=batch_size, shuffle=shuffle)
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler=None) -> float:
        """Train model for one epoch"""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            # Handle batched dictionary format (default PyTorch collate_fn)
            if isinstance(batch, dict) and 'input_ids' in batch:
                # Tokenized input - batch is already collated as dict
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device) 
                labels = batch['label'].to(self.device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)(outputs, labels)
            elif isinstance(batch, dict) and 'text' in batch:
                # Text input (fallback)
                texts = batch['text']  # This is a list
                labels = batch['label'].to(self.device)
                
                # Simple text to token conversion (for demo)
                input_ids = torch.randint(0, self.config.vocab_size, (len(texts), self.config.max_seq_length)).to(self.device)
                outputs = model(input_ids)
                loss = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)(outputs, labels)
            else:
                logger.warning(f"Unexpected batch format: {type(batch)}")
                if isinstance(batch, dict):
                    logger.warning(f"Batch keys: {list(batch.keys())}")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            optimizer.step()
            
            # Step learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def _evaluate_model(self, model: nn.Module, val_loader: DataLoader, metric_name: str) -> float:
        """Evaluate model on validation set"""
        model.eval()
        predictions = []
        true_labels = []
        
        # Get device from model parameters
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = self.device
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle batched dictionary format
                if isinstance(batch, dict) and 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(model_device)
                    attention_mask = batch['attention_mask'].to(model_device)
                    labels = batch['label']  # Keep on CPU for comparison
                    
                    outputs = model(input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs, dim=-1).cpu()
                elif isinstance(batch, dict) and 'text' in batch:
                    # Text input (fallback)
                    texts = batch['text']  # List of strings
                    labels = batch['label']  # Keep on CPU for comparison
                    
                    input_ids = torch.randint(0, self.config.vocab_size, (len(texts), self.config.max_seq_length)).to(model_device)
                    outputs = model(input_ids)
                    preds = torch.argmax(outputs, dim=-1).cpu()
                else:
                    continue
                
                # Convert to numpy for consistent handling
                if isinstance(preds, torch.Tensor):
                    preds = preds.numpy()
                if isinstance(labels, torch.Tensor):
                    labels = labels.numpy()
                
                predictions.extend(preds.tolist() if hasattr(preds, 'tolist') else list(preds))
                true_labels.extend(labels.tolist() if hasattr(labels, 'tolist') else list(labels))
        
        # Calculate metric
        if len(predictions) == 0:
            return 0.0
            
        # Debug print for first few predictions
        if len(predictions) > 5:
            logger.debug(f"Sample predictions: {predictions[:5]}")
            logger.debug(f"Sample true labels: {true_labels[:5]}")
            
        if metric_name == 'accuracy':
            correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
            accuracy = correct / len(predictions)
            logger.debug(f"Validation: {correct}/{len(predictions)} correct = {accuracy:.4f}")
            return accuracy
        elif metric_name == 'f1':
            # Simplified F1 calculation
            tp = sum(p == 1 and t == 1 for p, t in zip(predictions, true_labels))
            fp = sum(p == 1 and t == 0 for p, t in zip(predictions, true_labels))
            fn = sum(p == 0 and t == 1 for p, t in zip(predictions, true_labels))
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        else:
            # Default to accuracy
            correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
            return correct / len(predictions)

class ComprehensiveQwen3DiZOBenchmark:
    """Main benchmark orchestrator"""
    
    def __init__(self, config: BenchmarkConfig, device='cuda'):
        self.config = config
        
        # Handle device properly (could be string or torch.device)
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        # Validate device availability
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = torch.device('cpu')
        
        self.trainer = ModelTrainer(config, self.device)
        self.results = {}
        
        # Initialize benchmark suites
        if GLUE_AVAILABLE:
            # Use BERT tokenizer for GLUE+ benchmarks for compatibility
            # The actual model training can still use Qwen tokenizer separately
            self.glue_benchmark = GLUEPlusBenchmark(tokenizer_name="bert-base-uncased")
        if MEMORY_AVAILABLE:
            self.memory_benchmark = MemoryBenchmark()
        if ADVANCED_AVAILABLE:
            self.advanced_benchmark = AdvancedBenchmarkSuite(device=str(self.device))
        
        logger.info(f"ComprehensiveQwen3DiZOBenchmark initialized with device: {self.device}")
    
    def run_full_benchmark(self, tasks: List[str] = None, compare_dizo: bool = False) -> Dict:
        """Run complete benchmark suite"""
        logger.info("Starting comprehensive Qwen3 vs DiZO benchmark")
        
        # Default tasks
        if tasks is None:
            tasks = ['sst2', 'cola', 'mrpc']  # Start with basic tasks
        
        # Create models
        models = self.trainer.create_models()
        if not models:
            logger.error("Failed to create models!")
            return {}
        
        # Phase 1: Task-specific training and evaluation
        logger.info("Phase 1: Task-specific training")
        task_results = {}
        
        for task_name in tasks:
            if task_name not in GLUE_PLUS_TASKS:
                logger.warning(f"Unknown task: {task_name}")
                continue
                
            task_results[task_name] = {}
            
            for model_name, model in models.items():
                try:
                    training_result = self.trainer.train_model_on_task(model, task_name, model_name)
                    task_results[task_name][model_name] = training_result
                    logger.info(f"Completed {model_name} on {task_name}")
                except Exception as e:
                    logger.error(f"Failed to train {model_name} on {task_name}: {e}")
        
        self.results['task_specific'] = task_results
        
        # Phase 2: Comprehensive benchmarks using existing framework
        logger.info("Phase 2: Comprehensive benchmark evaluation")
        
        # Run GLUE+ benchmark
        if self.config.run_glue and GLUE_AVAILABLE:
            try:
                glue_results = self.glue_benchmark.run_full_benchmark(models, task_subset=tasks)
                self.results['glue_plus'] = glue_results
                logger.info("Completed GLUE+ benchmark")
            except Exception as e:
                logger.error(f"GLUE+ benchmark failed: {e}")
        
        # Run memory benchmark
        if self.config.run_memory and MEMORY_AVAILABLE:
            try:
                memory_results = self.memory_benchmark.run_memory_benchmark(models)
                self.results['memory'] = memory_results
                logger.info("Completed memory benchmark")
            except Exception as e:
                logger.error(f"Memory benchmark failed: {e}")
        
        # Run advanced benchmarks
        if self.config.run_advanced and ADVANCED_AVAILABLE:
            try:
                advanced_results = self.advanced_benchmark.run_comprehensive_benchmark(models, quick_test=True)
                self.results['advanced'] = advanced_results
                logger.info("Completed advanced benchmark")
            except Exception as e:
                logger.error(f"Advanced benchmark failed: {e}")
        
        # Phase 3: DiZO comparison (if requested)
        if compare_dizo:
            logger.info("Phase 3: DiZO model comparison")
            dizo_comparison = self._compare_with_dizo_results()
            self.results['dizo_comparison'] = dizo_comparison
        
        return self.results
    
    def _compare_with_dizo_results(self) -> Dict:
        """Compare results with known DiZO performance"""
        # Based on DiZO's published results or logs
        dizo_sst2_accuracy = 0.936  # From DiZO logs
        
        comparison = {}
        
        if 'task_specific' in self.results and 'sst2' in self.results['task_specific']:
            sst2_results = self.results['task_specific']['sst2']
            
            for model_name, result in sst2_results.items():
                if 'best_metric' in result:
                    model_accuracy = result['best_metric']
                    comparison[model_name] = {
                        'accuracy': model_accuracy,
                        'dizo_accuracy': dizo_sst2_accuracy,
                        'difference': model_accuracy - dizo_sst2_accuracy,
                        'relative_performance': model_accuracy / dizo_sst2_accuracy
                    }
        
        return comparison
    
    def print_comprehensive_summary(self):
        """Print detailed summary of all benchmark results"""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE QWEN3 vs DiZO BENCHMARK RESULTS")
        print("="*80)
        
        # Task-specific results
        if 'task_specific' in self.results:
            print("\n📊 TASK-SPECIFIC TRAINING RESULTS")
            print("-" * 50)
            
            for task_name, task_results in self.results['task_specific'].items():
                print(f"\n{task_name.upper()}:")
                for model_name, result in task_results.items():
                    if 'best_metric' in result:
                        print(f"  {model_name}: {result['best_metric']:.3f} (best), {result['epochs_trained']} epochs")
        
        # GLUE+ results
        if 'glue_plus' in self.results and GLUE_AVAILABLE:
            print("\n🧠 GLUE+ BENCHMARK RESULTS")
            print("-" * 50)
            self.glue_benchmark.print_benchmark_summary()
        
        # Memory benchmark results
        if 'memory' in self.results and MEMORY_AVAILABLE:
            print("\n💾 MEMORY BENCHMARK RESULTS")
            print("-" * 50)
            self.memory_benchmark.print_memory_benchmark_summary()
        
        # Advanced benchmark results
        if 'advanced' in self.results and ADVANCED_AVAILABLE:
            print("\n🚀 ADVANCED BENCHMARK RESULTS")
            print("-" * 50)
            self.advanced_benchmark.print_advanced_summary(self.results['advanced'])
        
        # DiZO comparison
        if 'dizo_comparison' in self.results:
            print("\n⚖️  DIZO COMPARISON RESULTS")
            print("-" * 50)
            
            for model_name, comparison in self.results['dizo_comparison'].items():
                rel_perf = comparison['relative_performance']
                status = "✓ Better" if rel_perf > 1.0 else "⚠ Worse" if rel_perf < 0.95 else "≈ Similar"
                print(f"  {model_name}: {comparison['accuracy']:.3f} vs DiZO {comparison['dizo_accuracy']:.3f} "
                      f"({comparison['difference']:+.3f}) {status}")
        
        # Summary statistics
        print("\n📈 SUMMARY STATISTICS")
        print("-" * 50)
        
        if 'task_specific' in self.results:
            all_reversible_scores = []
            all_standard_scores = []
            
            for task_results in self.results['task_specific'].values():
                for model_name, result in task_results.items():
                    if 'best_metric' in result:
                        if 'reversible' in model_name:
                            all_reversible_scores.append(result['best_metric'])
                        else:
                            all_standard_scores.append(result['best_metric'])
            
            if all_reversible_scores and all_standard_scores:
                rev_avg = np.mean(all_reversible_scores)
                std_avg = np.mean(all_standard_scores)
                print(f"  Average Reversible Performance: {rev_avg:.3f}")
                print(f"  Average Standard Performance: {std_avg:.3f}")
                print(f"  Performance Gap: {rev_avg - std_avg:+.3f}")
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"qwen3_dizo_benchmark_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Comprehensive Qwen3 vs DiZO Benchmark")
    
    parser.add_argument('--scale', choices=['small', 'medium', 'large'], default='small',
                        help='Model scale (affects size and training time)')
    
    parser.add_argument('--datasets', type=str, default='sst2,cola,mrpc',
                        help='Comma-separated list of datasets or "glue_basic", "glue_all", "all"')
    
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides scale default)')
    
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size (overrides scale default)')
    
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides default)')
    
    parser.add_argument('--full_eval', action='store_true',
                        help='Run all benchmark suites (GLUE+, memory, advanced)')
    
    parser.add_argument('--compare_dizo', action='store_true',
                        help='Include comparison with DiZO results')
    
    parser.add_argument('--save_models', action='store_true',
                        help='Save trained models')
    
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                        help='Output directory for results and models')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    return parser.parse_args()

def main():
    """Main benchmark execution"""
    args = parse_arguments()
    
    # Create config
    config = BenchmarkConfig(scale=args.scale)
    
    # Override config with command line arguments
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    if args.full_eval:
        config.run_glue = True
        config.run_memory = True
        config.run_advanced = True
        config.run_generation = True
    
    # Parse datasets
    if args.datasets == 'glue_basic':
        tasks = ['sst2', 'cola', 'mrpc']
    elif args.datasets == 'glue_all':
        tasks = list(GLUE_PLUS_TASKS.keys())[:8]  # All standard GLUE
    elif args.datasets == 'all':
        tasks = list(GLUE_PLUS_TASKS.keys())
    else:
        tasks = args.datasets.split(',')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize benchmark
    benchmark = ComprehensiveQwen3DiZOBenchmark(config, device=args.device)
    
    # Run benchmark
    logger.info(f"Starting benchmark with scale={args.scale}, tasks={tasks}")
    results = benchmark.run_full_benchmark(tasks=tasks, compare_dizo=args.compare_dizo)
    
    # Print summary
    benchmark.print_comprehensive_summary()
    
    # Save results
    results_file = output_dir / f"results_{args.scale}_{int(time.time())}.json"
    benchmark.save_results(str(results_file))
    
    logger.info("Benchmark completed successfully!")

if __name__ == "__main__":
    main()