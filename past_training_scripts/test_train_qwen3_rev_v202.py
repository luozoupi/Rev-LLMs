"""
Performance Testing Framework for Qwen3 Reversible Models - FIXED VERSION v203
===============================================================================

Fixed issues with validation loss calculation and perplexity computation
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
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
    def batch_iterator():
        yielded = 0
        for i in range(0, len(dataset['train']['text']), 1000):
            batch = [t for t in dataset['train']['text'][i:i+1000] if t and t.strip()]
            if not batch:
                continue
            yield batch
            yielded += len(batch)
            if yielded > 500000:
                break
    tokenizer.train_from_iterator(batch_iterator(), trainer)
    return tokenizer


class PackedTextDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids: List[int], seq_len: int, stride: Optional[int] = None):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.data = []
        need = seq_len + 1
        for start in range(0, len(token_ids) - need + 1, self.stride):
            window = token_ids[start:start+need]
            if len(window) == need:
                self.data.append(window)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        span = self.data[idx]
        x = torch.tensor(span[:-1], dtype=torch.long)
        y = torch.tensor(span[1:], dtype=torch.long)
        y[-1] = -100
        return x, y


def build_wikitext_packed(dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", tokenizer_name="gpt2", seq_len=512, stride=None, min_val_windows=50):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset(dataset_name, dataset_config)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": tok.eos_token or "<|pad|>"})
    def tokenize_split(split):
        if split not in ds:
            return None
        texts = [t for t in ds[split]['text'] if t and t.strip()]
        corpus = "\n".join(texts)
        ids = tok(corpus, add_special_tokens=False, return_attention_mask=False).input_ids
        if len(ids) < seq_len + 10:
            return PackedTextDataset([], seq_len)
        stride_eff = stride or seq_len // 2
        return PackedTextDataset(ids, seq_len, stride_eff)
    train_ds = tokenize_split("train")
    val_ds = tokenize_split("validation") or tokenize_split("test")
    test_ds = tokenize_split("test")
    def ensure(ds_obj, name, fallback):
        if ds_obj is None or len(ds_obj) == 0:
            print(f"⚠️  {name} empty; sampling fallback windows")
            n = min(min_val_windows, len(fallback))
            if n == 0:
                raise RuntimeError("No data available")
            idxs = torch.linspace(0, len(fallback)-1, n).long().tolist()
            return torch.utils.data.Subset(fallback, idxs)
        return ds_obj
    train_ds = ensure(train_ds, "train", train_ds)
    val_ds = ensure(val_ds, "val", train_ds)
    test_ds = ensure(test_ds, "test", train_ds)
    print(f"Packed windows -> train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")
    return train_ds, val_ds, test_ds, len(tok)

@dataclass
class PerformanceMetrics:
    """Store comprehensive performance metrics"""
    model_name: str
    perplexity: float
    loss: float
    accuracy: float
    memory_peak_mb: float
    memory_reserved_mb: float
    training_time_per_epoch: float
    inference_time_per_token: float
    gradient_norm: float
    
    # Model-specific metrics
    attention_sparsity: Optional[float] = None
    pruning_ratio: Optional[float] = None
    reversible_layers_used: Optional[int] = None

class ReversibleQwenPerformanceTester:
    """Comprehensive performance testing for Reversible Qwen3 models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
    def create_test_dataset(self, vocab_size=50000, seq_len=512, num_samples=1000):
        """Create synthetic dataset for testing"""
        
        # Create random sequences with realistic token distribution
        weights = np.array([1.0 / (i + 1) ** 0.8 for i in range(vocab_size)])
        weights /= weights.sum()
        
        data = []
        for _ in range(num_samples):
            sequence = np.random.choice(vocab_size, size=seq_len, p=weights)
            # Create targets (next token prediction) with ignore index for padding
            targets = np.roll(sequence, -1)
            
            # Convert to tensors and set last token to ignore index
            x = torch.tensor(sequence, dtype=torch.long)
            y = torch.tensor(targets, dtype=torch.long)
            y[-1] = -100  # Set last token to ignore index
            
            data.append((x, y))
        
        return data
    
    def get_wikitext_training_config(self):
        """Optimized config for WikiText training"""
        return {
            'epochs': 15,  # More epochs for real data
            'learning_rate': 0.0003,  # Moderate LR for WikiText
            'min_learning_rate': 1e-6,
            'scheduler_type': 'cosine',  # Cosine works well for language modeling
            'warmup_ratio': 0.1,
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'gradient_clip_norm': 1.0,
            'gradient_accumulation_steps': 2,  # Accumulate for effective larger batch
            'use_amp': True,
            'amp_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            'early_stopping_patience': 5,  # More patience for WikiText
            'early_stopping_min_delta': 0.001,
            'betas': (0.9, 0.95),
            'eps': 1e-8
        }
    
    def get_optimized_training_config(self, model_type="reversible"):
        """Get optimized training configuration based on model type"""
        
        if "reversible" in model_type.lower():
            return self.get_wikitext_training_config()
        else:
            # Standard model config
            return {
                'epochs': 30,
                'learning_rate': 0.0001,
                'min_learning_rate': 1e-6,
                'scheduler_type': 'cosine',
                'warmup_ratio': 0.05,
                'optimizer': 'adamw',
                'weight_decay': 0.01,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'gradient_clip_norm': 1.0,
                'gradient_accumulation_steps': 1,
                'early_stopping_patience': 3,
                'early_stopping_min_delta': 0.001,
                'use_amp': True,
                'amp_dtype': torch.float16,
            }
        
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
                torch.nn.init.normal_(m.weight, std=std)
        
        model.apply(init_weights)
        
        # Special handling for reversible blocks
        for module in model.modules():
            if hasattr(module, 'grad_scale'):
                torch.nn.init.constant_(module.grad_scale, 0.1)
                
        return model
    
    def setup_models_for_comparison(self, vocab_size=10000, hidden_size=768, 
                                  num_layers=6, num_heads=12):
        """Setup different model configurations for comparison"""
        
        from qwen3_reversible_02 import create_reversible_qwen3_model
        
        models = {}
        
        # FIXED: Updated model configurations with better parameters
        configs = [
            ("reversible_standard", "standard", True),
            ("reversible_candidate", "candidate_selection", True), 
            ("reversible_sparse", "native_sparse", True),
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
                    candidate_pr_ratio=0.7,
                    candidate_top_k=32
                )
                
                model = self.initialize_model_weights(model, name)
                model = model.to(self.device)
                models[name] = model
                
                param_count = sum(p.numel() for p in model.parameters())
                print(f"✅ Created {name} model ({param_count:,} params)")
                
            except Exception as e:
                print(f"❌ Failed to create {name}: {e}")
                
        return models
    
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
    
    def fine_tune_model(self, model, train_loader, val_loader, max_epochs=50):
        """FIXED: Fine-tune model with proper validation handling"""
        
        model_type = "reversible" if hasattr(model, 'use_reversible') and getattr(model, 'use_reversible', False) else "standard"
        config = self.get_optimized_training_config(model_type)
        
        print(f"🔧 Training config: {config}")
        print(f"📊 Data loaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        
        # DEBUGGING: Check validation loader
        if len(val_loader) == 0:
            print("❌ CRITICAL: Validation loader is empty!")
            return self._create_empty_results()
            
        # Test validation loader
        try:
            val_sample = next(iter(val_loader))
            print(f"✅ Validation sample shapes: {val_sample[0].shape}, {val_sample[1].shape}")
        except Exception as e:
            print(f"❌ CRITICAL: Cannot iterate validation loader: {e}")
            return self._create_empty_results()
        
        # Setup optimizer
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
        
        # FIXED: Use consistent loss function with proper ignore_index
        criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        
        # Training tracking
        train_losses = []
        val_losses = []
        epoch_times = []
        memory_usage = []
        gradient_norms = []
        
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
            
            optimizer.zero_grad()
            
            # Training loop
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                try:
                    if config['use_amp'] and scaler is not None:
                        with autocast(dtype=config['amp_dtype']):
                            outputs = model(inputs)
                            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                            
                            # Next token prediction loss
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
                    
                    if batch_idx % 20 == 0:
                        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config['learning_rate']
                        print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                        
                except Exception as e:
                    print(f"❌ Training error at batch {batch_idx}: {e}")
                    continue
            
            avg_train_loss = total_train_loss / max(num_train_batches, 1)
            avg_grad_norm = total_grad_norm / max(num_train_batches // config['gradient_accumulation_steps'], 1)
            epoch_time = time.time() - epoch_start
            
            # Record memory usage
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
                reserved_memory = torch.cuda.max_memory_reserved() / (1024**2)
                memory_usage.append({'peak': peak_memory, 'reserved': reserved_memory})
            else:
                memory_usage.append({'peak': 0, 'reserved': 0})
            
            # FIXED: Validation loop with debugging
            model.eval()
            val_loss = 0
            val_batches = 0
            val_tokens = 0
            
            print(f"🔍 Starting validation with {len(val_loader)} batches...")
            
            with torch.no_grad():
                for val_batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    try:
                        outputs = model(inputs)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        
                        # FIXED: Consistent loss calculation
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = targets[..., 1:].contiguous()
                        
                        # Calculate loss with same criterion as training
                        loss = criterion(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        # Count valid tokens (not ignored)
                        valid_tokens = (shift_labels.view(-1) != -100).sum().item()
                        
                        if not torch.isnan(loss) and not torch.isinf(loss) and valid_tokens > 0:
                            val_loss += loss.item()
                            val_tokens += valid_tokens
                            val_batches += 1
                        else:
                            print(f"⚠️ Skipping validation batch {val_batch_idx}: loss={loss.item()}, valid_tokens={valid_tokens}")
                        
                    except Exception as e:
                        print(f"❌ Validation error at batch {val_batch_idx}: {e}")
                        continue
            
            # FIXED: Proper validation loss calculation
            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
            else:
                print("❌ CRITICAL: No valid validation batches processed!")
                avg_val_loss = float('inf')
            
            print(f"📈 Validation: {val_batches} batches processed, {val_tokens} tokens, avg_loss: {avg_val_loss:.4f}")
            
            # Record metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            epoch_times.append(epoch_time)
            gradient_norms.append(avg_grad_norm)
            
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config['learning_rate']
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s, "
                  f"Memory: {memory_usage[-1]['peak']:.1f}MB, "
                  f"Grad Norm: {avg_grad_norm:.4f}, LR: {current_lr:.6f}")
            
            # Early stopping with patience
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1} (patience {config['early_stopping_patience']})")
                    break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epoch_times': epoch_times,
            'memory_usage': memory_usage,
            'gradient_norms': gradient_norms
        }
    
    def _create_empty_results(self):
        """Create empty results for failed cases"""
        return {
            'train_losses': [],
            'val_losses': [],
            'epoch_times': [],
            'memory_usage': [],
            'gradient_norms': []
        }
    
    def calculate_perplexity(self, model, dataloader, max_batches=50):
        """FIXED: Calculate perplexity with proper error handling"""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        valid_batches = 0
        
        # Use same criterion as training for consistency
        criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        
        print(f"🔍 Calculating perplexity with {min(len(dataloader), max_batches)} batches...")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                try:
                    outputs = model(inputs)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Calculate loss for perplexity (next token prediction)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    
                    loss = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # Count valid tokens (not ignored)
                    valid_tokens = (shift_labels.view(-1) != -100).sum().item()
                    
                    if not torch.isnan(loss) and not torch.isinf(loss) and valid_tokens > 0:
                        total_loss += loss.item()
                        total_tokens += valid_tokens
                        valid_batches += 1
                    else:
                        print(f"⚠️ Skipping perplexity batch {batch_idx}: loss={loss.item()}, valid_tokens={valid_tokens}")
                    
                except Exception as e:
                    print(f"❌ Perplexity calculation error at batch {batch_idx}: {e}")
                    continue
        
        print(f"📊 Perplexity calculation: {valid_batches} valid batches, {total_tokens} total tokens")
        
        if total_tokens == 0 or valid_batches == 0:
            print("❌ No valid tokens found for perplexity calculation!")
            return float('inf'), float('inf')
        
        avg_loss = total_loss / total_tokens
        
        # Clamp loss to prevent overflow in exp
        avg_loss = min(avg_loss, 20.0)  # exp(20) ≈ 485M, reasonable upper bound
        
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"📈 Avg loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return perplexity, avg_loss
    
    def debug_dataset(self, dataset, name="dataset"):
        """Debug dataset to understand structure"""
        print(f"🔍 Debugging {name}...")
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample_x, sample_y = dataset[0]
            print(f"Sample input shape: {sample_x.shape}")
            print(f"Sample target shape: {sample_y.shape}")
            print(f"Sample input tokens (first 10): {sample_x[:10].tolist()}")
            print(f"Sample target tokens (first 10): {sample_y[:10].tolist()}")
            
            # Check for ignore index usage
            ignore_count = (sample_y == -100).sum().item()
            print(f"Ignore tokens (-100) in sample: {ignore_count}")
            
            # Check token range
            print(f"Token range - input: [{sample_x.min().item()}, {sample_x.max().item()}]")
            print(f"Token range - target: [{sample_y.min().item()}, {sample_y.max().item()}]")
    
    def run_comprehensive_test(self, seq_len=512, vocab_size=32000, use_wikitext=True):
        """FIXED: Run comprehensive test with debugging"""
        
        print("="*60)
        print("COMPREHENSIVE REVERSIBLE QWEN3 ON WIKITEXT - FIXED v203")
        print("="*60)
        
        if use_wikitext:
            try:
                # Load and pack WikiText dataset
                print("Creating packed WikiText datasets...")
                train_dataset, val_dataset, test_dataset, vocab_size = build_wikitext_packed(
                    dataset_name="wikitext",
                    dataset_config="wikitext-2-raw-v1",
                    tokenizer_name="gpt2",
                    seq_len=seq_len,
                    stride=seq_len//4,
                    min_val_windows=50
                )
                
                # Debug datasets
                self.debug_dataset(train_dataset, "train")
                self.debug_dataset(val_dataset, "validation")
                self.debug_dataset(test_dataset, "test")
                
                # FIXED: Smaller batch sizes for WikiText
                train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=False)
                test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=False)
                
                print(f"🔧 Using vocabulary size: {vocab_size}")
                
            except Exception as e:
                print(f"❌ WikiText loading failed: {e}")
                print("🔄 Falling back to synthetic data...")
                use_wikitext = False
        
        if not use_wikitext:
            # Fallback to synthetic data
            train_data = self.create_test_dataset(vocab_size, seq_len, 500)
            val_data = self.create_test_dataset(vocab_size, seq_len, 100)
            test_data = self.create_test_dataset(vocab_size, seq_len, 200)
            
            train_loader = DataLoader(train_data, batch_size=4, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
        
        # Setup models
        print("Setting up models...")
        models = self.setup_models_for_comparison(
            vocab_size=vocab_size,
            hidden_size=512,  # Smaller for testing
            num_layers=4,     # Fewer layers for faster testing
            num_heads=8
        )
        
        if not models:
            print("❌ No models created successfully!")
            return None
        
        # Test each model
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*40}")
            print(f"Testing {model_name}")
            print(f"{'='*40}")
            
            try:
                # Calculate baseline perplexity with debugging
                print("📊 Calculating baseline perplexity...")
                baseline_ppl, baseline_loss = self.calculate_perplexity(model, test_loader, max_batches=10)
                print(f"📈 Baseline perplexity: {baseline_ppl:.2f}")
                
                # # Skip training if baseline is problematic
                # if baseline_ppl == float('inf') or baseline_ppl > 10000:
                #     print(f"⚠️ Baseline perplexity too high ({baseline_ppl}), skipping training for {model_name}")
                #     continue
                
                # Fine-tune model
                print("🚀 Starting fine-tuning...")
                training_results = self.fine_tune_model(model, train_loader, val_loader, max_epochs=30)
                
                if not training_results['val_losses']:
                    print(f"❌ No validation results for {model_name}")
                    continue
                
                # Calculate post-training perplexity
                print("📊 Calculating post-training perplexity...")
                final_ppl, final_loss = self.calculate_perplexity(model, test_loader, max_batches=10)
                print(f"📈 Final perplexity: {final_ppl:.2f}")
                
                # Collect metrics
                metrics = PerformanceMetrics(
                    model_name=model_name,
                    perplexity=final_ppl,
                    loss=final_loss,
                    accuracy=0.0,
                    memory_peak_mb=max([m['peak'] for m in training_results['memory_usage']]) if training_results['memory_usage'] else 0,
                    memory_reserved_mb=max([m['reserved'] for m in training_results['memory_usage']]) if training_results['memory_usage'] else 0,
                    training_time_per_epoch=np.mean(training_results['epoch_times']) if training_results['epoch_times'] else 0,
                    inference_time_per_token=0.0,
                    gradient_norm=np.mean(training_results['gradient_norms']) if training_results['gradient_norms'] else 0
                )
                
                improvement = baseline_ppl - final_ppl if baseline_ppl != float('inf') and final_ppl != float('inf') else 0
                
                all_results[model_name] = {
                    'metrics': metrics,
                    'training_curves': training_results,
                    'baseline_perplexity': baseline_ppl,
                    'improvement': improvement
                }
                
                print(f"✅ {model_name} completed successfully")
                print(f"   Improvement: {improvement:.2f} perplexity reduction")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_results
    
    def visualize_results(self, results):
        """Create comprehensive visualization of results"""
        
        if not results:
            print("❌ No results to visualize")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        model_names = list(results.keys())
        
        # Filter out infinite perplexities for visualization
        valid_results = {name: result for name, result in results.items() 
                        if result['metrics'].perplexity != float('inf')}
        
        if not valid_results:
            print("❌ No valid results to visualize (all perplexities are infinite)")
            return None
        
        valid_names = list(valid_results.keys())
        
        # 1. Perplexity Comparison
        perplexities = [valid_results[name]['metrics'].perplexity for name in valid_names]
        axes[0, 0].bar(valid_names, perplexities)
        axes[0, 0].set_title('Final Perplexity (Lower is Better)')
        axes[0, 0].set_ylabel('Perplexity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Memory Usage
        memory_usage = [results[name]['metrics'].memory_peak_mb for name in model_names]
        axes[0, 1].bar(model_names, memory_usage)
        axes[0, 1].set_title('Peak Memory Usage')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Training Time
        train_times = [results[name]['metrics'].training_time_per_epoch for name in model_names]
        axes[0, 2].bar(model_names, train_times)
        axes[0, 2].set_title('Training Time per Epoch')
        axes[0, 2].set_ylabel('Time (seconds)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Perplexity Improvement
        improvements = [valid_results[name]['improvement'] for name in valid_names]
        axes[1, 0].bar(valid_names, improvements)
        axes[1, 0].set_title('Perplexity Improvement')
        axes[1, 0].set_ylabel('PPL Reduction')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Training Loss Curves
        for name in model_names:
            if 'training_curves' in results[name]:
                curves = results[name]['training_curves']
                if curves['train_losses'] and curves['val_losses']:
                    axes[1, 1].plot(curves['train_losses'], label=f'{name} (train)')
                    axes[1, 1].plot(curves['val_losses'], '--', label=f'{name} (val)')
        axes[1, 1].set_title('Training Curves')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        
        # 6. Gradient Norm
        grad_norms = [results[name]['metrics'].gradient_norm for name in model_names]
        axes[1, 2].bar(model_names, grad_norms)
        axes[1, 2].set_title('Average Gradient Norm')
        axes[1, 2].set_ylabel('Gradient Norm')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def save_results(self, results, filename='reversible_qwen_wikitext_results.json'):
        """Save results to file"""
        
        if not results:
            print("❌ No results to save")
            return
        
        serializable_results = {}
        for name, result in results.items():
            serializable_results[name] = {
                'metrics': result['metrics'].__dict__,
                'baseline_perplexity': result['baseline_perplexity'],
                'improvement': result['improvement'],
                'training_curves': {
                    k: v for k, v in result['training_curves'].items()
                    if k != 'memory_usage'
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")


# DEBUGGING FUNCTIONS
# ===================

def debug_data_loader(loader, name="loader"):
    """Debug data loader to check for issues"""
    print(f"🔍 Debugging {name}...")
    print(f"Loader length: {len(loader)}")
    
    try:
        sample_batch = next(iter(loader))
        inputs, targets = sample_batch
        print(f"✅ Sample batch shapes: inputs={inputs.shape}, targets={targets.shape}")
        print(f"Sample input range: [{inputs.min().item()}, {inputs.max().item()}]")
        print(f"Sample target range: [{targets.min().item()}, {targets.max().item()}]")
        print(f"Ignore tokens in targets: {(targets == -100).sum().item()}")
    except Exception as e:
        print(f"❌ Cannot iterate {name}: {e}")

def debug_model_output(model, sample_input):
    """Debug model output structure"""
    print("🔍 Debugging model output...")
    
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(sample_input)
            
            if isinstance(outputs, dict):
                print(f"✅ Model output is dict with keys: {list(outputs.keys())}")
                if 'logits' in outputs:
                    logits = outputs['logits']
                    print(f"Logits shape: {logits.shape}")
                    print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            else:
                print(f"✅ Model output is tensor with shape: {outputs.shape}")
                print(f"Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                
        except Exception as e:
            print(f"❌ Model forward pass failed: {e}")


# MAIN TESTING SCRIPT
# ===================

if __name__ == "__main__":
    # Create tester
    tester = ReversibleQwenPerformanceTester()
    
    print("🧪 Running FIXED WikiText training test...")
    
    # Test with WikiText
    results = tester.run_comprehensive_test(
        seq_len=256*4,  # Smaller for debugging
        vocab_size=16000,  # Smaller vocab for faster testing
        use_wikitext=True
    )
    
    if results:
        print("\n" + "="*60)
        print("SUMMARY OF WIKITEXT RESULTS")
        print("="*60)
        
        for name, result in results.items():
            metrics = result['metrics']
            print(f"\n{name.upper()}:")
            print(f"  Final Perplexity: {metrics.perplexity:.2f}")
            print(f"  Peak Memory: {metrics.memory_peak_mb:.1f} MB")
            print(f"  Training Time/Epoch: {metrics.training_time_per_epoch:.1f}s")
            print(f"  Perplexity Improvement: {result['improvement']:.2f}")
            print(f"  Gradient Norm: {metrics.gradient_norm:.4f}")
        
        # Create visualization
        fig = tester.visualize_results(results)
        if fig:
            plt.savefig('reversible_qwen_wikitext_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("📊 Visualization saved!")
        
        # Save results
        tester.save_results(results)
        
        print("\n✅ WikiText training test completed!")
        
    else:
        print("❌ No results generated - check errors above")