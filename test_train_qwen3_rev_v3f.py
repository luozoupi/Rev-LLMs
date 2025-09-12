"""
Performance Testing Framework for Qwen3 Reversible Models - Enhanced Benchmarks
===============================================================================

Added comprehensive benchmarks: token accuracy, throughput, memory scaling, 
training stability, and bits-per-byte metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

def load_wikitext_data(subset="wikitext-103-raw-v1"):
    """Load WikiText dataset instead of synthetic data"""
    print("Loading WikiText dataset...")
    dataset = load_dataset("wikitext", subset)
    print(f"Loaded WikiText: Train={len(dataset['train'])}, Val={len(dataset['validation'])}, Test={len(dataset['test'])}")
    return dataset

def create_wikitext_tokenizer(dataset, vocab_size=32000):
    """Create tokenizer from WikiText data"""
    print("Creating WikiText tokenizer...")
    
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
        texts_yielded = 0
        for i in range(0, len(dataset["train"]["text"]), 1000):
            batch_texts = []
            for text in dataset["train"]["text"][i:i+1000]:
                if text and text.strip() and len(text.strip()) > 10:
                    batch_texts.append(text.strip())
            if batch_texts:
                yield batch_texts
                texts_yielded += len(batch_texts)
        print(f"Used {texts_yielded} texts for tokenizer training")
    
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    return tokenizer

class WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_split, tokenizer, seq_length=512, stride=256):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.unk_id = tokenizer.token_to_id("<unk>")
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")
        
        self.ignore_index = self.pad_id if self.pad_id is not None else 0
        
        print(f"Special tokens - pad: {self.pad_id}, unk: {self.unk_id}, bos: {self.bos_id}, eos: {self.eos_id}")
        
        self.examples = []
        
        valid_texts = [text for text in dataset_split["text"] if text and text.strip() and len(text.strip()) > 20]
        print(f"Processing {len(valid_texts)} valid texts from {len(dataset_split['text'])} total texts...")
        
        for text in tqdm(valid_texts, desc="Processing WikiText"):
            clean_text = text.strip().replace('\n', ' ')
            if len(clean_text) < 50:
                continue
                
            tokens = tokenizer.encode(clean_text).ids
            
            if len(tokens) < seq_length + 1:
                continue
            
            for start_idx in range(0, len(tokens) - seq_length, stride):
                sequence = tokens[start_idx:start_idx + seq_length + 1]
                if len(sequence) == seq_length + 1:
                    self.examples.append(sequence)
        
        print(f"Created WikiText dataset with {len(self.examples)} sequences")
        
        if len(self.examples) == 0:
            raise ValueError("No valid sequences created! Check tokenization and sequence length.")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        y = torch.where(y == self.pad_id, torch.tensor(-100, dtype=torch.long), y)
        
        return x, y

@dataclass
class EnhancedPerformanceMetrics:
    """Enhanced performance metrics with additional benchmarks"""
    
    # Basic Language Modeling
    model_name: str
    perplexity: float
    loss: float
    token_accuracy: float
    bits_per_byte: float
    
    # Memory & Compute Efficiency  
    memory_peak_mb: float
    memory_reserved_mb: float
    memory_efficiency_ratio: float
    throughput_tokens_per_sec: float
    time_per_token_ms: float
    estimated_flops_per_token: float
    
    # Training Dynamics
    training_time_per_epoch: float
    gradient_norm: float
    convergence_rate: float
    training_stability: float
    overfitting_gap: float
    epochs_to_converge: int
    
    # Memory Scaling
    memory_scaling_slope: float
    
    # Model-specific
    attention_sparsity: Optional[float] = None
    pruning_ratio: Optional[float] = None
    reversible_layers_used: Optional[int] = None

class ReversibleQwenPerformanceTester:
    """Enhanced performance testing with comprehensive benchmarks"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
    def create_test_dataset(self, vocab_size=50000, seq_len=512, num_samples=1000):
        """Create synthetic dataset for testing"""
        
        weights = np.array([1.0 / (i + 1) ** 0.8 for i in range(vocab_size)])
        weights /= weights.sum()
        
        data = []
        for _ in range(num_samples):
            sequence = np.random.choice(vocab_size, size=seq_len, p=weights)
            targets = np.roll(sequence, -1)
            
            x = torch.tensor(sequence, dtype=torch.long)
            y = torch.tensor(targets, dtype=torch.long)
            y[-1] = -100
            
            data.append((x, y))
        
        return data
    
    def get_wikitext_training_config(self):
        """Optimized config for WikiText training"""
        return {
            'epochs': 8,
            'learning_rate': 1e-4,
            'min_learning_rate': 1e-6,
            'scheduler_type': 'cosine',
            'warmup_ratio': 0.1,
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'gradient_clip_norm': 1.0,
            'gradient_accumulation_steps': 2,
            'use_amp': True,
            'amp_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            'early_stopping_patience': 10,
            'early_stopping_min_delta': 0.0001,
            'betas': (0.9, 0.99),
            'eps': 1e-8
        }
    
    def get_optimized_training_config(self, model_type="reversible"):
        """Get optimized training configuration based on model type"""
        
        # Match older v202_2f behavior: shorter training for reversible, longer & lower LR for non-reversible
        base_config = self.get_wikitext_training_config()
        return base_config
        # if "reversible" in model_type.lower():
        #     # Keep defaults (epochs=8, lr=3e-4, patience=4)
        #     return base_config
        # else:
        #     cfg = dict(base_config)
        #     cfg.update({
        #         'epochs': 30,
        #         'learning_rate': 1e-4,
        #         'early_stopping_patience': 10,
        #         'betas': (0.9, 0.999)
        #     })
        #     return cfg
        
    def initialize_model_weights(self, model, model_type="reversible"):
        """Proper initialization for different model types"""
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                if "reversible" in model_type.lower():
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)
                else:
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)
                    
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                    
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                std = 0.01 if "reversible" in model_type.lower() else 0.02
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()
                
        model.apply(init_weights)
        return model
    
    def create_scheduler(self, optimizer, config, steps_per_epoch):
        """Create appropriate learning rate scheduler"""
        
        total_steps = config['epochs'] * steps_per_epoch
        
        if config['scheduler_type'] == 'onecycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config['learning_rate'],
                total_steps=total_steps,
                pct_start=config['warmup_ratio'],
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1000.0,
                cycle_momentum=True
            )
        elif config['scheduler_type'] == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=config['min_learning_rate']
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
            
        return scheduler
    
    def setup_models_for_comparison(self, vocab_size=10000, hidden_size=512, 
                                    num_layers=4, num_heads=8):
            """Setup different model configurations for comparison"""
            
            from qwen3_reversible_02_2 import create_reversible_qwen3_model
            
            models = {}
            
            # Focus on pure reversible vs standard comparison
            configs = [
                ("reversible_standard", "standard", True),
                ("non_reversible_standard", "standard", False)
            ]
            
            for name, attention_type, use_reversible in configs:
                try:
                    model = create_reversible_qwen3_model(
                        vocab_size=vocab_size,
                        hidden_size=hidden_size,
                        num_hidden_layers=num_layers,
                        num_attention_heads=num_heads,
                        num_key_value_heads=num_heads // 2,
                        attention_type=attention_type,
                        use_reversible=use_reversible,
                        reverse_thres=256 if use_reversible else 999999,
                        # Add missing parameters from working version
                        candidate_pr_ratio=0.7,
                        candidate_top_k=32,
                        intermediate_size=hidden_size * 4,
                        max_position_embeddings=2048,
                        rms_norm_eps=1e-6
                    )
                    
                    model = self.initialize_model_weights(model, name)
                    model = model.to(self.device)
                    models[name] = model
                    
                    param_count = sum(p.numel() for p in model.parameters())
                    print(f"Created {name} model ({param_count:,} params)")
                    
                except Exception as e:
                    print(f"Failed to create {name}: {e}")
                    
            return models
    
    
    # NEW: Enhanced benchmarking methods
    def measure_computational_efficiency(self, model, dataloader, max_batches=30):
        """Measure throughput and computational efficiency"""
        
        model.eval()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        total_tokens = 0
        total_time = 0
        
        with torch.no_grad():
            overall_start = time.time()
            
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                    
                inputs = inputs.to(self.device)
                batch_start = time.time()
                
                outputs = model(inputs)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                batch_time = time.time() - batch_start
                
                total_tokens += inputs.numel()
                total_time += batch_time
        
        throughput = total_tokens / total_time if total_time > 0 else 0
        time_per_token = (total_time / total_tokens * 1000) if total_tokens > 0 else 0
        
        # Estimate FLOPs (simplified approximation)
        total_params = sum(p.numel() for p in model.parameters())
        estimated_flops_per_token = 6 * total_params  # Forward pass approximation
        
        return {
            'throughput_tokens_per_sec': throughput,
            'time_per_token_ms': time_per_token,
            'estimated_flops_per_token': estimated_flops_per_token
        }
    
    def calculate_enhanced_language_metrics(self, model, dataloader, max_batches=30):
        """Calculate token accuracy and bits-per-byte"""
        
        model.eval()
        
        # Token accuracy tracking
        correct_predictions = 0
        total_predictions = 0
        
        # Bits per byte tracking
        total_nll_bits = 0
        total_chars = 0
        
        criterion_sum = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                try:
                    outputs = model(inputs)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    
                    # Token accuracy calculation
                    predictions = torch.argmax(shift_logits, dim=-1)
                    mask = (shift_labels != -100)
                    correct = (predictions == shift_labels) & mask
                    
                    correct_predictions += correct.sum().item()
                    total_predictions += mask.sum().item()
                    
                    # Bits per byte calculation
                    nll = criterion_sum(shift_logits.view(-1, shift_logits.size(-1)), 
                                      shift_labels.view(-1))
                    
                    total_nll_bits += nll.item() / np.log(2)  # Convert to bits
                    total_chars += mask.sum().item()
                    
                except Exception as e:
                    print(f"Enhanced metrics error: {e}")
                    continue
        
        token_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        bits_per_byte = total_nll_bits / total_chars if total_chars > 0 else float('inf')
        
        return {
            'token_accuracy': token_accuracy,
            'bits_per_byte': bits_per_byte
        }
    
    def analyze_memory_scaling(self, model, seq_lengths=[128, 256, 512, 1024]):
        """Analyze how memory scales with sequence length"""
        
        memory_scaling = []
        model.eval()
        
        for seq_len in seq_lengths:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            test_input = torch.randint(0, min(1000, getattr(model.config, 'vocab_size', 1000)), 
                                     (1, seq_len)).to(self.device)
            
            try:
                with torch.no_grad():
                    outputs = model(test_input)
                
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
                else:
                    peak_memory = 0
                    
                memory_scaling.append({
                    'seq_length': seq_len,
                    'peak_memory_mb': peak_memory,
                    'memory_per_token': peak_memory / seq_len if seq_len > 0 else 0
                })
                
                print(f"  Seq {seq_len}: {peak_memory:.1f} MB ({peak_memory/seq_len:.3f} MB/token)")
                
            except Exception as e:
                print(f"Memory test failed for seq_len {seq_len}: {e}")
                memory_scaling.append({
                    'seq_length': seq_len,
                    'peak_memory_mb': float('inf'),
                    'memory_per_token': float('inf')
                })
        
        # Calculate memory scaling slope
        if len(memory_scaling) > 1:
            seq_lens = [m['seq_length'] for m in memory_scaling]
            memories = [m['peak_memory_mb'] for m in memory_scaling if m['peak_memory_mb'] != float('inf')]
            
            if len(memories) > 1:
                slope = np.polyfit(seq_lens[:len(memories)], memories, 1)[0]
            else:
                slope = float('inf')
        else:
            slope = 0
        
        return {
            'memory_scaling_data': memory_scaling,
            'memory_scaling_slope': slope
        }
    
    def measure_memory_efficiency(self, model):
        """Calculate memory efficiency metrics"""
        
        total_params = sum(p.numel() for p in model.parameters())
        param_memory = total_params * 4 / (1024**2)  # MB for fp32
        
        test_input = torch.randint(0, min(1000, getattr(model.config, 'vocab_size', 1000)), 
                                 (1, 512)).to(self.device)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            outputs = model(test_input)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            peak_memory = param_memory
        
        memory_efficiency_ratio = peak_memory / param_memory if param_memory > 0 else 1.0
        
        return {
            'parameter_memory_mb': param_memory,
            'peak_memory_mb': peak_memory,
            'memory_efficiency_ratio': memory_efficiency_ratio
        }
    
    def measure_training_stability(self, training_curves):
        """Measure training stability and convergence characteristics"""
        
        if not training_curves.get('train_losses') or len(training_curves['train_losses']) < 3:
            return {
                'convergence_rate': 0.0,
                'training_stability': 0.0,
                'overfitting_gap': 0.0,
                'epochs_to_converge': 0
            }
        
        train_losses = np.array(training_curves['train_losses'])
        val_losses = np.array(training_curves.get('val_losses', []))
        
        # Convergence rate (loss improvement per epoch)
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        convergence_rate = (initial_loss - final_loss) / len(train_losses)
        
        # Training stability (inverse of loss variance)
        loss_variance = np.var(train_losses)
        training_stability = 1.0 / (loss_variance + 1e-8)
        
        # Overfitting gap
        if len(val_losses) > 0 and len(val_losses) == len(train_losses):
            overfitting_gap = val_losses[-1] - train_losses[-1]
        else:
            overfitting_gap = 0.0
        
        return {
            'convergence_rate': float(convergence_rate),
            'training_stability': float(training_stability),
            'overfitting_gap': float(overfitting_gap),
            'epochs_to_converge': len(train_losses)
        }
    
    def fine_tune_model(self, model, train_loader, val_loader, max_epochs=50):
        """Enhanced fine-tuning with additional metric collection"""
        
        model_type = "reversible" if hasattr(model, 'use_reversible') and getattr(model, 'use_reversible', False) else "standard"
        config = self.get_optimized_training_config(model_type)
        
        print(f"Training config: LR={config['learning_rate']}, Epochs={min(max_epochs, config['epochs'])}")
        print(f"Data loaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        
        if len(val_loader) == 0:
            print("CRITICAL: Validation loader is empty!")
            return self._create_empty_results()
        
        # Setup optimizer and scheduler
        if config['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                betas=config['betas'],
                eps=config['eps']
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        scheduler = self.create_scheduler(optimizer, config, len(train_loader))
        scaler = GradScaler() if config['use_amp'] else None
        criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        
        # Training tracking
        train_losses = []
        val_losses = []
        epoch_times = []
        memory_usage = []
        gradient_norms = []
        
        # NEW: Enhanced tracking
        token_accuracies = []
        throughput_measurements = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        epochs_to_run = min(max_epochs, config['epochs'])
        
        for epoch in range(epochs_to_run):
            model.train()
            epoch_start = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            total_train_loss = 0
            num_train_batches = 0
            total_grad_norm = 0
            
            # NEW: Token accuracy tracking during training
            train_correct = 0
            train_total = 0
            
            optimizer.zero_grad()
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                try:
                    if config['use_amp'] and scaler is not None:
                        with autocast(dtype=config['amp_dtype']):
                            outputs = model(inputs)
                            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                            
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = targets[..., 1:].contiguous()
                            loss = criterion(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                        
                        scaled_loss = scaler.scale(loss / config['gradient_accumulation_steps'])
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
                        
                        (loss / config['gradient_accumulation_steps']).backward()
                    
                    total_train_loss += loss.item()
                    num_train_batches += 1
                    
                    # NEW: Calculate token accuracy during training
                    with torch.no_grad():
                        predictions = torch.argmax(shift_logits, dim=-1)
                        mask = (shift_labels != -100)
                        correct = (predictions == shift_labels) & mask
                        train_correct += correct.sum().item()
                        train_total += mask.sum().item()
                    
                    # Update weights
                    if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                        if config['use_amp'] and scaler is not None:
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), config['gradient_clip_norm']
                            )
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), config['gradient_clip_norm']
                            )
                            optimizer.step()
                        
                        scheduler.step()
                        optimizer.zero_grad()
                        total_grad_norm += grad_norm.item()
                    
                    if batch_idx % 30 == 0:
                        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config['learning_rate']
                        train_acc = train_correct / train_total * 100 if train_total > 0 else 0
                        print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss: {loss.item():.4f}, "
                              f"Acc: {train_acc:.1f}%, LR: {current_lr:.6f}")
                        
                except Exception as e:
                    print(f"Training error at batch {batch_idx}: {e}")
                    continue
            
            avg_train_loss = total_train_loss / max(num_train_batches, 1)
            avg_grad_norm = total_grad_norm / max(num_train_batches // config['gradient_accumulation_steps'], 1)
            epoch_time = time.time() - epoch_start
            epoch_token_accuracy = train_correct / train_total if train_total > 0 else 0
            
            # Record memory usage
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
                reserved_memory = torch.cuda.max_memory_reserved() / (1024**2)
                memory_usage.append({'peak': peak_memory, 'reserved': reserved_memory})
            else:
                memory_usage.append({'peak': 0, 'reserved': 0})
            
            # Enhanced validation loop
            model.eval()
            val_loss = 0
            val_batches = 0
            val_tokens = 0
            val_correct = 0
            val_total = 0
            
            print(f"Starting validation with {len(val_loader)} batches...")
            
            with torch.no_grad():
                for val_batch_idx, (inputs, targets) in enumerate(val_loader):
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
                        
                        # Token accuracy for validation
                        predictions = torch.argmax(shift_logits, dim=-1)
                        mask = (shift_labels != -100)
                        correct = (predictions == shift_labels) & mask
                        
                        valid_tokens = mask.sum().item()
                        
                        if not torch.isnan(loss) and not torch.isinf(loss) and valid_tokens > 0:
                            val_loss += loss.item()
                            val_correct += correct.sum().item()
                            val_total += valid_tokens
                            val_tokens += valid_tokens
                            val_batches += 1
                        
                    except Exception as e:
                        print(f"Validation error at batch {val_batch_idx}: {e}")
                        continue
            
            # Calculate averages
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            val_token_accuracy = val_correct / val_total if val_total > 0 else 0
            
            print(f"Validation: {val_batches} batches, {val_tokens} tokens, "
                  f"loss: {avg_val_loss:.4f}, acc: {val_token_accuracy*100:.1f}%")
            
            # Record metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            epoch_times.append(epoch_time)
            gradient_norms.append(avg_grad_norm)
            token_accuracies.append(val_token_accuracy)
            
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config['learning_rate']
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Train Acc: {epoch_token_accuracy*100:.1f}%, "
                  f"Val Acc: {val_token_accuracy*100:.1f}%, Time: {epoch_time:.2f}s, "
                  f"Memory: {memory_usage[-1]['peak']:.1f}MB, LR: {current_lr:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epoch_times': epoch_times,
            'memory_usage': memory_usage,
            'gradient_norms': gradient_norms,
            'token_accuracies': token_accuracies
        }
    
    def _create_empty_results(self):
        """Create empty results for failed cases"""
        return {
            'train_losses': [],
            'val_losses': [],
            'epoch_times': [],
            'memory_usage': [],
            'gradient_norms': [],
            'token_accuracies': []
        }
    
    def calculate_perplexity(self, model, dataloader, max_batches=30):
        """Enhanced perplexity calculation with better error handling"""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        valid_batches = 0
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        
        print(f"Calculating perplexity with up to {max_batches} batches...")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                try:
                    outputs = model(inputs)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    
                    loss = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    valid_tokens = (shift_labels.view(-1) != -100).sum().item()
                    
                    if not torch.isnan(loss) and not torch.isinf(loss) and valid_tokens > 0:
                        total_loss += loss.item()
                        total_tokens += valid_tokens
                        valid_batches += 1
                    
                except Exception as e:
                    print(f"Perplexity calculation error at batch {batch_idx}: {e}")
                    continue
        
        print(f"Perplexity calculation: {valid_batches} valid batches, {total_tokens} total tokens")
        
        if total_tokens == 0 or valid_batches == 0:
            print("No valid tokens found for perplexity calculation!")
            return float('inf'), float('inf')
        
        avg_loss = total_loss / total_tokens
        avg_loss = min(avg_loss, 20.0)  # Clamp to prevent overflow
        
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"Final perplexity: {perplexity:.2f} (avg_loss: {avg_loss:.4f})")
        
        return perplexity, avg_loss
    
    def debug_dataset(self, dataset, name="dataset"):
        """Debug dataset to understand structure"""
        print(f"Debugging {name}...")
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample_x, sample_y = dataset[0]
            print(f"Sample input shape: {sample_x.shape}")
            print(f"Sample target shape: {sample_y.shape}")
            print(f"Sample input tokens (first 10): {sample_x[:10].tolist()}")
            print(f"Sample target tokens (first 10): {sample_y[:10].tolist()}")
            
            ignore_count = (sample_y == -100).sum().item()
            print(f"Ignore tokens (-100) in sample: {ignore_count}")
            
            print(f"Token range - input: [{sample_x.min().item()}, {sample_x.max().item()}]")
            print(f"Token range - target: [{sample_y.min().item()}, {sample_y.max().item()}]")
    
    def run_comprehensive_test(self, seq_len=512, vocab_size=32000, use_wikitext=True):
        """Enhanced comprehensive test with all benchmarks"""
        
        print("="*60)
        print("ENHANCED REVERSIBLE QWEN3 COMPREHENSIVE BENCHMARKS")
        print("="*60)
        
        if use_wikitext:
            try:
                dataset = load_wikitext_data()
                tokenizer = create_wikitext_tokenizer(dataset, vocab_size)
                actual_vocab_size = tokenizer.get_vocab_size()
                print(f"Actual vocabulary size: {actual_vocab_size}")
                
                print("Creating WikiText datasets...")
                train_dataset = WikiTextDataset(dataset['train'], tokenizer, seq_len, stride=seq_len//4)
                val_dataset = WikiTextDataset(dataset['validation'], tokenizer, seq_len, stride=seq_len//4) 
                test_dataset = WikiTextDataset(dataset['test'], tokenizer, seq_len, stride=seq_len//4)
                
                self.debug_dataset(train_dataset, "train")
                self.debug_dataset(val_dataset, "validation")
                self.debug_dataset(test_dataset, "test")
                
                train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=False)
                test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=False)
                
                vocab_size = actual_vocab_size
                
            except Exception as e:
                print(f"WikiText loading failed: {e}")
                print("Falling back to synthetic data...")
                use_wikitext = False
        
        if not use_wikitext:
            train_data = self.create_test_dataset(vocab_size, seq_len, 500)
            val_data = self.create_test_dataset(vocab_size, seq_len, 100)
            test_data = self.create_test_dataset(vocab_size, seq_len, 200)
            
            train_loader = DataLoader(train_data, batch_size=4, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
        
        # Setup models
        print("Setting up models...")
        # Make model size comparable to older v202_2f defaults (hidden_size=1024, num_layers=6, heads=8)
        models = self.setup_models_for_comparison(
            vocab_size=vocab_size,
            hidden_size=1024*2,
            num_layers=6+2,
            num_heads=8
        )
        
        if not models:
            print("No models created successfully!")
            return None
        
        # Test each model with enhanced benchmarks
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*50}")
            print(f"ENHANCED TESTING: {model_name}")
            print(f"{'='*50}")
            
            try:
                # 1. Baseline perplexity and language metrics
                print("1. Calculating baseline language modeling metrics...")
                baseline_ppl, baseline_loss = self.calculate_perplexity(model, test_loader, max_batches=10)
                baseline_lm_metrics = self.calculate_enhanced_language_metrics(model, test_loader, max_batches=10)
                
                # 2. Computational efficiency
                print("2. Measuring computational efficiency...")
                comp_metrics = self.measure_computational_efficiency(model, test_loader, max_batches=5)
                
                # 3. Memory efficiency and scaling
                print("3. Analyzing memory efficiency...")
                mem_metrics = self.measure_memory_efficiency(model)
                
                print("4. Testing memory scaling...")
                scaling_metrics = self.analyze_memory_scaling(model, seq_lengths=[128, 256, 512])
                
                # 5. Training with enhanced tracking
                print("5. Fine-tuning with enhanced metrics...")
                training_results = self.fine_tune_model(model, train_loader, val_loader, max_epochs=30)
                
                if not training_results['val_losses']:
                    print(f"No validation results for {model_name}")
                    continue
                
                # 6. Training stability analysis
                print("6. Analyzing training stability...")
                stability_metrics = self.measure_training_stability(training_results)
                
                # 7. Post-training metrics
                print("7. Final language modeling metrics...")
                final_ppl, final_loss = self.calculate_perplexity(model, test_loader, max_batches=10)
                final_lm_metrics = self.calculate_enhanced_language_metrics(model, test_loader, max_batches=10)
                
                # Combine into enhanced metrics
                enhanced_metrics = EnhancedPerformanceMetrics(
                    model_name=model_name,
                    perplexity=final_ppl,
                    loss=final_loss,
                    token_accuracy=final_lm_metrics['token_accuracy'],
                    bits_per_byte=final_lm_metrics['bits_per_byte'],
                    
                    memory_peak_mb=max([m['peak'] for m in training_results['memory_usage']]) if training_results['memory_usage'] else 0,
                    memory_reserved_mb=max([m['reserved'] for m in training_results['memory_usage']]) if training_results['memory_usage'] else 0,
                    memory_efficiency_ratio=mem_metrics['memory_efficiency_ratio'],
                    throughput_tokens_per_sec=comp_metrics['throughput_tokens_per_sec'],
                    time_per_token_ms=comp_metrics['time_per_token_ms'],
                    estimated_flops_per_token=comp_metrics['estimated_flops_per_token'],
                    
                    training_time_per_epoch=np.mean(training_results['epoch_times']) if training_results['epoch_times'] else 0,
                    gradient_norm=np.mean(training_results['gradient_norms']) if training_results['gradient_norms'] else 0,
                    convergence_rate=stability_metrics['convergence_rate'],
                    training_stability=stability_metrics['training_stability'],
                    overfitting_gap=stability_metrics['overfitting_gap'],
                    epochs_to_converge=stability_metrics['epochs_to_converge'],
                    
                    memory_scaling_slope=scaling_metrics['memory_scaling_slope']
                )
                
                improvement = baseline_ppl - final_ppl if baseline_ppl != float('inf') and final_ppl != float('inf') else 0
                
                all_results[model_name] = {
                    'enhanced_metrics': enhanced_metrics,
                    'training_curves': training_results,
                    'baseline_perplexity': baseline_ppl,
                    'improvement': improvement,
                    'memory_scaling_data': scaling_metrics['memory_scaling_data']
                }
                
                print(f"COMPLETED: {model_name}")
                print(f"  Perplexity: {baseline_ppl:.2f} -> {final_ppl:.2f} (improvement: {improvement:.2f})")
                print(f"  Token Accuracy: {final_lm_metrics['token_accuracy']*100:.1f}%")
                print(f"  Throughput: {comp_metrics['throughput_tokens_per_sec']:.0f} tokens/sec")
                print(f"  Memory Efficiency: {mem_metrics['memory_efficiency_ratio']:.2f}x")
                print(f"  Memory Scaling: {scaling_metrics['memory_scaling_slope']:.2f} MB/token")
                
            except Exception as e:
                print(f"FAILED: {model_name} - {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_results
    
    
    
    def create_enhanced_visualization(self, results):
        """Create comprehensive enhanced visualization"""
        
        if not results:
            print("No results to visualize")
            return None
        
        model_names = list(results.keys())
        
        # Create enhanced visualization
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle('Enhanced Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract enhanced metrics
        enhanced_metrics = {name: result['enhanced_metrics'] for name, result in results.items()}
        
        # 1. Language Modeling Performance
        perplexities = [enhanced_metrics[name].perplexity for name in model_names]
        token_accuracies = [enhanced_metrics[name].token_accuracy * 100 for name in model_names]
        bpb_scores = [enhanced_metrics[name].bits_per_byte for name in model_names]
        
        # Filter out infinite values for plotting
        finite_perplexities = [p if p != float('inf') else 0 for p in perplexities]
        finite_bpb = [b if b != float('inf') else 0 for b in bpb_scores]
        
        axes[0, 0].bar(model_names, finite_perplexities)
        axes[0, 0].set_title('Perplexity (Lower = Better)')
        axes[0, 0].set_ylabel('Perplexity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(model_names, token_accuracies)
        axes[0, 1].set_title('Token Accuracy (Higher = Better)')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[0, 2].bar(model_names, finite_bpb)
        axes[0, 2].set_title('Bits per Byte (Lower = Better)')
        axes[0, 2].set_ylabel('BPB')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 2. Computational Efficiency
        throughputs = [enhanced_metrics[name].throughput_tokens_per_sec for name in model_names]
        memory_peaks = [enhanced_metrics[name].memory_peak_mb for name in model_names]
        memory_efficiency = [enhanced_metrics[name].memory_efficiency_ratio for name in model_names]
        
        axes[0, 3].bar(model_names, throughputs)
        axes[0, 3].set_title('Throughput (Higher = Better)')
        axes[0, 3].set_ylabel('Tokens/sec')
        axes[0, 3].tick_params(axis='x', rotation=45)
        
        axes[1, 0].bar(model_names, memory_peaks)
        axes[1, 0].set_title('Peak Memory (Lower = Better)')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(model_names, memory_efficiency)
        axes[1, 1].set_title('Memory Efficiency (Lower = Better)')
        axes[1, 1].set_ylabel('Peak/Parameter Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 3. Training Dynamics
        convergence_rates = [enhanced_metrics[name].convergence_rate for name in model_names]
        training_stability = [enhanced_metrics[name].training_stability for name in model_names]
        overfitting_gaps = [enhanced_metrics[name].overfitting_gap for name in model_names]
        
        axes[1, 2].bar(model_names, convergence_rates)
        axes[1, 2].set_title('Convergence Rate (Higher = Better)')
        axes[1, 2].set_ylabel('Loss decrease/epoch')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        axes[1, 3].bar(model_names, training_stability)
        axes[1, 3].set_title('Training Stability (Higher = Better)')
        axes[1, 3].set_ylabel('Stability Score')
        axes[1, 3].tick_params(axis='x', rotation=45)
        
        axes[2, 0].bar(model_names, overfitting_gaps)
        axes[2, 0].set_title('Overfitting Gap (Lower = Better)')
        axes[2, 0].set_ylabel('Val Loss - Train Loss')
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # 4. Memory Scaling and Additional Metrics
        scaling_slopes = [enhanced_metrics[name].memory_scaling_slope for name in model_names]
        finite_slopes = [s if s != float('inf') else 0 for s in scaling_slopes]
        
        axes[2, 1].bar(model_names, finite_slopes)
        axes[2, 1].set_title('Memory Scaling (Lower = Better)')
        axes[2, 1].set_ylabel('MB per token')
        axes[2, 1].tick_params(axis='x', rotation=45)
        
        # 5. Training Loss Curves
        for name in model_names:
            if 'training_curves' in results[name]:
                curves = results[name]['training_curves']
                if curves['train_losses'] and curves['val_losses']:
                    axes[2, 2].plot(curves['train_losses'], label=f'{name} (train)')
                    axes[2, 2].plot(curves['val_losses'], '--', label=f'{name} (val)')
        axes[2, 2].set_title('Training Curves')
        axes[2, 2].set_ylabel('Loss')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].legend()
        
        # 6. Token Accuracy Over Training
        for name in model_names:
            if 'training_curves' in results[name]:
                curves = results[name]['training_curves']
                if curves.get('token_accuracies'):
                    epochs = range(len(curves['token_accuracies']))
                    accuracies = [acc * 100 for acc in curves['token_accuracies']]
                    axes[2, 3].plot(epochs, accuracies, label=name)
        axes[2, 3].set_title('Token Accuracy Over Training')
        axes[2, 3].set_ylabel('Accuracy (%)')
        axes[2, 3].set_xlabel('Epoch')
        axes[2, 3].legend()
        
        plt.tight_layout()
        return fig
    
    def print_enhanced_summary(self, results):
        """Print comprehensive summary with all benchmarks"""
        
        if not results:
            print("No results to summarize")
            return
        
        print("\n" + "="*100)
        print("ENHANCED BENCHMARK SUMMARY")
        print("="*100)
        
        # Create formatted table
        headers = ['Model', 'Perplexity', 'Token Acc%', 'BPB', 'Throughput', 'Memory MB', 
                  'Mem Eff', 'Convergence', 'Stability', 'Mem Scaling']
        
        print(f"{'Model':<20} {'PPL':<8} {'Acc%':<6} {'BPB':<8} {'Tok/s':<8} {'Mem':<8} {'Eff':<6} {'Conv':<8} {'Stab':<8} {'Scale':<8}")
        print("-" * 95)
        
        for name, result in results.items():
            metrics = result['enhanced_metrics']
            
            ppl = f"{metrics.perplexity:.1f}" if metrics.perplexity != float('inf') else "inf"
            acc = f"{metrics.token_accuracy*100:.1f}"
            bpb = f"{metrics.bits_per_byte:.2f}" if metrics.bits_per_byte != float('inf') else "inf"
            thr = f"{metrics.throughput_tokens_per_sec:.0f}"
            mem = f"{metrics.memory_peak_mb:.0f}"
            eff = f"{metrics.memory_efficiency_ratio:.1f}"
            conv = f"{metrics.convergence_rate:.3f}"
            stab = f"{metrics.training_stability:.1f}"
            scale = f"{metrics.memory_scaling_slope:.2f}" if metrics.memory_scaling_slope != float('inf') else "inf"
            
            print(f"{name:<20} {ppl:<8} {acc:<6} {bpb:<8} {thr:<8} {mem:<8} {eff:<6} {conv:<8} {stab:<8} {scale:<8}")
        
        print("="*100)
        print("\nMETRIC EXPLANATIONS:")
        print("PPL = Perplexity (language modeling quality, lower better)")
        print("Acc% = Token prediction accuracy (higher better)")  
        print("BPB = Bits per byte (compression efficiency, lower better)")
        print("Tok/s = Inference throughput (higher better)")
        print("Mem = Peak memory usage in MB (lower better)")
        print("Eff = Memory efficiency ratio (lower better)")
        print("Conv = Convergence rate (loss improvement/epoch, higher better)")
        print("Stab = Training stability (higher better)")
        print("Scale = Memory scaling slope MB/token (lower better)")
        
        # Performance comparison insights
        print("\n" + "="*100)
        print("KEY INSIGHTS:")
        print("="*100)
        
        reversible_models = [name for name in results.keys() if 'reversible' in name.lower()]
        standard_models = [name for name in results.keys() if 'reversible' not in name.lower()]
        
        if reversible_models and standard_models:
            # Compare best of each type
            best_reversible = min(reversible_models, 
                                key=lambda x: results[x]['enhanced_metrics'].perplexity)
            best_standard = min(standard_models,
                              key=lambda x: results[x]['enhanced_metrics'].perplexity)
            
            rev_metrics = results[best_reversible]['enhanced_metrics']
            std_metrics = results[best_standard]['enhanced_metrics']
            
            print(f"BEST REVERSIBLE ({best_reversible}) vs BEST STANDARD ({best_standard}):")
            print(f"  Perplexity: {rev_metrics.perplexity:.2f} vs {std_metrics.perplexity:.2f}")
            print(f"  Token Accuracy: {rev_metrics.token_accuracy*100:.1f}% vs {std_metrics.token_accuracy*100:.1f}%")
            print(f"  Throughput: {rev_metrics.throughput_tokens_per_sec:.0f} vs {std_metrics.throughput_tokens_per_sec:.0f} tokens/sec")
            print(f"  Memory Efficiency: {rev_metrics.memory_efficiency_ratio:.2f}x vs {std_metrics.memory_efficiency_ratio:.2f}x")
            print(f"  Memory Scaling: {rev_metrics.memory_scaling_slope:.2f} vs {std_metrics.memory_scaling_slope:.2f} MB/token")
            print(f"  Training Stability: {rev_metrics.training_stability:.2f} vs {std_metrics.training_stability:.2f}")
        
        # Memory efficiency insights
        best_memory = min(results.keys(), key=lambda x: results[x]['enhanced_metrics'].memory_peak_mb)
        best_throughput = max(results.keys(), key=lambda x: results[x]['enhanced_metrics'].throughput_tokens_per_sec)
        best_accuracy = max(results.keys(), key=lambda x: results[x]['enhanced_metrics'].token_accuracy)
        
        print(f"\nBEST PERFORMERS:")
        print(f"  Memory Efficiency: {best_memory}")
        print(f"  Throughput: {best_throughput}")
        print(f"  Token Accuracy: {best_accuracy}")
    
    def save_enhanced_results(self, results, filename='enhanced_qwen_results.json'):
        """Save enhanced results with all benchmarks.

        Accepts either:
        - A dict of per-model results: {model_name: {enhanced_metrics, ...}}
        - A final_report dict with key 'basic_performance' holding the above
        """
        
        if not results:
            print("No results to save")
            return
        
        # Detect if this is a final_report bundle
        if isinstance(results, dict) and 'basic_performance' in results and 'enhanced_metrics' not in results:
            base = results.get('basic_performance', {}) or {}
            serializable_results = {}
            for name, result in base.items():
                if not result or 'enhanced_metrics' not in result:
                    continue
                serializable_results[name] = {
                    'enhanced_metrics': getattr(result['enhanced_metrics'], '__dict__', result['enhanced_metrics']),
                    'baseline_perplexity': result.get('baseline_perplexity'),
                    'improvement': result.get('improvement'),
                    'memory_scaling_data': result.get('memory_scaling_data'),
                    'training_curves': {
                        k: v for k, v in result.get('training_curves', {}).items()
                        if k != 'memory_usage'
                    }
                }
            payload = {
                'basic_performance': serializable_results,
                'scaling_analysis': results.get('scaling_analysis', {}),
                'benchmark_summary': results.get('benchmark_summary', {})
            }
        else:
            # Assume it's already a per-model map
            serializable_results = {}
            for name, result in results.items():
                if not result or 'enhanced_metrics' not in result:
                    continue
                serializable_results[name] = {
                    'enhanced_metrics': getattr(result['enhanced_metrics'], '__dict__', result['enhanced_metrics']),
                    'baseline_perplexity': result.get('baseline_perplexity'),
                    'improvement': result.get('improvement'),
                    'memory_scaling_data': result.get('memory_scaling_data'),
                    'training_curves': {
                        k: v for k, v in result.get('training_curves', {}).items()
                        if k != 'memory_usage'
                    }
                }
            payload = serializable_results
        
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        
        print(f"Enhanced results saved to {filename}")


# MAIN TESTING SCRIPT
# ===================

if __name__ == "__main__":
    # Import sophisticated benchmarks
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from glue_plus_benchmark import GLUEPlusBenchmark, get_recommended_task_progression
        from memory_benchmark import MemoryBenchmark, get_recommended_memory_task_sequence
        BENCHMARKS_AVAILABLE = True
    except ImportError as e:
        print(f"Advanced benchmarks not available: {e}")
        BENCHMARKS_AVAILABLE = False
    
    # Create enhanced tester
    tester = ReversibleQwenPerformanceTester()
    
    print("="*80)
    print("COMPREHENSIVE REVERSIBLE QWEN3 EVALUATION SUITE")
    print("="*80)
    
    # Phase 1: Basic Performance Benchmarking
    print("\n🔬 PHASE 1: Basic Performance & Memory Efficiency")
    print("-" * 60)
    
    basic_results = tester.run_comprehensive_test(
        seq_len=1024,  # Match v202_2f (256*4)
        vocab_size=16000*2,
        use_wikitext=True
    )
    
    if basic_results:
        print("\n📊 Basic Performance Summary:")
        tester.print_enhanced_summary(basic_results)
        
        # Create and save basic visualization
        fig = tester.create_enhanced_visualization(basic_results)
        if fig:
            plt.savefig('phase1_basic_performance.png', dpi=300, bbox_inches='tight')
            print("Phase 1 visualization saved!")
        
        # Phase 2: GLUE+ Language Understanding Benchmark (opt-in via env)
        RUN_GLUE_PLUS = BENCHMARKS_AVAILABLE and os.environ.get('RUN_GLUE_PLUS', '0') == '1'
        if RUN_GLUE_PLUS:
            print("\n🧠 PHASE 2: GLUE+ Language Understanding")
            print("-" * 60)
            
            # Use base LM models; GLUE evaluator will derive logits from LM output safely
            models_for_glue = tester.setup_models_for_comparison(
                vocab_size=16000, hidden_size=768, num_layers=6, num_heads=12
            )
            
            if models_for_glue:
                glue_benchmark = GLUEPlusBenchmark()
                
                # Start with core GLUE tasks
                task_progression = get_recommended_task_progression()
                
                print("Starting with Phase 1 GLUE tasks...")
                glue_results = glue_benchmark.run_full_benchmark(
                    models_for_glue, 
                    device=tester.device,
                    task_subset=task_progression['phase_1_basic']
                )
                
                if glue_results:
                    print("\n📈 GLUE+ Results:")
                    glue_benchmark.print_benchmark_summary()
                    
                    # Save GLUE results
                    with open('glue_plus_results.json', 'w') as f:
                        json.dump(glue_results, f, indent=2, default=str)
        
        # Phase 3: Long-Range Memory Benchmark (opt-in via env)
        RUN_MEMORY_BENCH = BENCHMARKS_AVAILABLE and os.environ.get('RUN_MEMORY_BENCH', '0') == '1'
        if RUN_MEMORY_BENCH:
            print("\n🧮 PHASE 3: Long-Range Memory & Context")
            print("-" * 60)
            
            memory_benchmark = MemoryBenchmark()
            
            # Use original models (not classification variants)
            original_models = tester.setup_models_for_comparison(
                vocab_size=1000, hidden_size=512, num_layers=4, num_heads=8
            )
            
            if original_models:
                memory_task_sequence = get_recommended_memory_task_sequence()
                
                print("Running comprehensive memory tasks...")
                memory_results = memory_benchmark.run_memory_benchmark(
                    original_models, device=tester.device
                )
                
                if memory_results:
                    print("\n🎯 Memory Benchmark Results:")
                    memory_benchmark.print_memory_benchmark_summary()
                    
                    # Save memory results
                    with open('memory_benchmark_results.json', 'w') as f:
                        json.dump(memory_results, f, indent=2, default=str)
        
        # Phase 4: Scaling Analysis
        print("\n📈 PHASE 4: Scaling & Efficiency Analysis")
        print("-" * 60)
        
        scaling_results = {}
        
        # Test different model sizes
        size_configs = [
            {'name': 'Small', 'hidden_size': 256, 'num_layers': 4, 'num_heads': 4},
            {'name': 'Medium', 'hidden_size': 512, 'num_layers': 6, 'num_heads': 8},
            {'name': 'Large', 'hidden_size': 768, 'num_layers': 8, 'num_heads': 12},
        ]
        
        for config in size_configs:
            print(f"\nTesting {config['name']} models...")
            
            try:
                size_models = tester.setup_models_for_comparison(
                    vocab_size=8000,
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    num_heads=config['num_heads']
                )
                
                if size_models:
                    # Quick evaluation on synthetic data
                    test_data = tester.create_test_dataset(8000, 512, 100)
                    test_loader = DataLoader(test_data, batch_size=2, shuffle=False)
                    
                    size_results = {}
                    for model_name, model in size_models.items():
                        try:
                            # Measure computational efficiency
                            comp_metrics = tester.measure_computational_efficiency(
                                model, test_loader, max_batches=10
                            )
                            
                            # Measure memory scaling
                            scaling_metrics = tester.analyze_memory_scaling(
                                model, seq_lengths=[128, 256, 512]
                            )
                            
                            size_results[model_name] = {
                                'computational': comp_metrics,
                                'memory_scaling': scaling_metrics,
                                'parameters': sum(p.numel() for p in model.parameters()),
                                'model_size': config['name']
                            }
                            
                            print(f"  {model_name}: {comp_metrics['throughput_tokens_per_sec']:.0f} tok/s, "
                                  f"{scaling_metrics['memory_scaling_slope']:.2f} MB/tok")
                            
                        except Exception as e:
                            print(f"  {model_name}: Failed - {e}")
                    
                    scaling_results[config['name']] = size_results
                    
            except Exception as e:
                print(f"Failed to test {config['name']} models: {e}")
        
        # Phase 5: Final Comprehensive Report
        print("\n📋 PHASE 5: Comprehensive Analysis Report")
        print("=" * 80)
        
        # Combine all results
        final_report = {
            'basic_performance': basic_results,
            'scaling_analysis': scaling_results,
            'benchmark_summary': {
                'total_phases': 5,
                'models_tested': list(basic_results.keys()) if basic_results else [],
                'benchmarks_completed': []
            }
        }
        
        if BENCHMARKS_AVAILABLE:
            if 'glue_results' in locals():
                final_report['glue_plus'] = glue_results
                final_report['benchmark_summary']['benchmarks_completed'].append('GLUE+')
            
            if 'memory_results' in locals():
                final_report['memory_benchmark'] = memory_results
                final_report['benchmark_summary']['benchmarks_completed'].append('Memory')
        
        # Save comprehensive results
        tester.save_enhanced_results(final_report, 'comprehensive_evaluation_results.json')
        
        # Print final summary
        print("\n🎉 EVALUATION COMPLETE!")
        print("-" * 40)
        print(f"✅ Phases completed: 5/5")
        print(f"✅ Models tested: {len(final_report['benchmark_summary']['models_tested'])}")
        print(f"✅ Benchmarks: {', '.join(final_report['benchmark_summary']['benchmarks_completed'])}")
        print(f"✅ Results saved to: comprehensive_evaluation_results.json")
        
        # Performance insights
        if basic_results:
            reversible_models = [name for name in basic_results.keys() if 'reversible' in name.lower()]
            standard_models = [name for name in basic_results.keys() if 'reversible' not in name.lower()]
            
            if reversible_models and standard_models:
                print(f"\n🔍 KEY INSIGHTS:")
                
                # Memory efficiency
                rev_memory = np.mean([basic_results[name]['enhanced_metrics'].memory_peak_mb for name in reversible_models])
                std_memory = np.mean([basic_results[name]['enhanced_metrics'].memory_peak_mb for name in standard_models])
                memory_savings = (std_memory - rev_memory) / std_memory * 100
                
                print(f"💾 Memory Efficiency: {memory_savings:.1f}% reduction with reversible models")
                
                # Performance comparison
                rev_ppl = np.mean([basic_results[name]['enhanced_metrics'].perplexity for name in reversible_models 
                                  if basic_results[name]['enhanced_metrics'].perplexity != float('inf')])
                std_ppl = np.mean([basic_results[name]['enhanced_metrics'].perplexity for name in standard_models
                                  if basic_results[name]['enhanced_metrics'].perplexity != float('inf')])
                
                if rev_ppl != float('inf') and std_ppl != float('inf'):
                    ppl_diff = ((rev_ppl - std_ppl) / std_ppl) * 100
                    print(f"🎯 Language Modeling: {abs(ppl_diff):.1f}% {'worse' if ppl_diff > 0 else 'better'} perplexity")
                
                # Throughput comparison
                rev_throughput = np.mean([basic_results[name]['enhanced_metrics'].throughput_tokens_per_sec for name in reversible_models])
                std_throughput = np.mean([basic_results[name]['enhanced_metrics'].throughput_tokens_per_sec for name in standard_models])
                throughput_diff = ((rev_throughput - std_throughput) / std_throughput) * 100
                
                print(f"⚡ Throughput: {abs(throughput_diff):.1f}% {'faster' if throughput_diff > 0 else 'slower'}")
        
        print(f"\n📊 Visualizations saved:")
        print(f"  - phase1_basic_performance.png")
        if 'fig' in locals():
            plt.show()
    
    else:
        print("❌ No results generated - check errors above")