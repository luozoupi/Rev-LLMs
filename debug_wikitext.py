#!/usr/bin/env python3
"""
Debug script to test WikiText-2 loading and tokenization
"""
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset

def debug_wikitext_loading():
    """Debug WikiText-2 loading step by step"""
    
    print("=== STEP 1: Loading WikiText-2 Dataset ===")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        print(f"✓ Successfully loaded WikiText-2")
        print(f"  Train: {len(dataset['train'])} examples")
        print(f"  Validation: {len(dataset['validation'])} examples")
        print(f"  Test: {len(dataset['test'])} examples")
        
        # Sample some texts to see what we're working with
        print("\n=== STEP 2: Examining Sample Texts ===")
        train_texts = dataset['train']['text']
        
        # Check for empty or short texts
        empty_count = sum(1 for text in train_texts if not text or not text.strip())
        short_count = sum(1 for text in train_texts if text and text.strip() and len(text.strip()) < 20)
        valid_count = sum(1 for text in train_texts if text and text.strip() and len(text.strip()) >= 20)
        
        print(f"  Empty texts: {empty_count}")
        print(f"  Short texts (<20 chars): {short_count}")
        print(f"  Valid texts (>=20 chars): {valid_count}")
        
        # Show some sample valid texts
        valid_texts = [text for text in train_texts if text and text.strip() and len(text.strip()) >= 20]
        print(f"\nFirst 5 valid texts:")
        for i, text in enumerate(valid_texts[:5]):
            print(f"  {i+1}. [{len(text)} chars] {text[:100]}...")
        
        print("\n=== STEP 3: Creating BPE Tokenizer ===")
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=1000,  # Small vocab for testing
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
            min_frequency=1  # Very low threshold for testing
        )
        
        def batch_iterator():
            texts_yielded = 0
            all_texts = []
            
            # Collect valid texts
            for text in train_texts:
                if text and text.strip() and len(text.strip()) > 5:
                    clean_text = text.strip().replace('\n', ' ').replace('\t', ' ')
                    clean_text = ' '.join(clean_text.split())
                    if len(clean_text) > 10:
                        all_texts.append(clean_text)
            
            print(f"    Found {len(all_texts)} valid texts for tokenizer training")
            
            # Yield in batches
            for i in range(0, len(all_texts), 100):
                batch = all_texts[i:i+100]
                if batch:
                    yield batch
                    texts_yielded += len(batch)
            
            print(f"    Used {texts_yielded} texts for tokenizer training")
        
        print("  Training tokenizer...")
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        print(f"  ✓ Tokenizer created with vocab size: {tokenizer.get_vocab_size()}")
        
        # Test tokenization on a sample
        if valid_texts:
            sample_text = valid_texts[0][:200]  # Take first 200 chars
            encoded = tokenizer.encode(sample_text)
            print(f"  Sample tokenization:")
            print(f"    Text: {sample_text}")
            print(f"    Tokens: {encoded.tokens[:20]}...")  # First 20 tokens
            print(f"    Token IDs: {encoded.ids[:20]}...")  # First 20 IDs
            print(f"    Total tokens: {len(encoded.ids)}")
        
        print("\n=== STEP 4: Testing Dataset Creation ===")
        
        class TestWikiTextDataset(Dataset):
            def __init__(self, dataset_split, tokenizer, seq_length=64):  # Shorter for testing
                self.tokenizer = tokenizer
                self.seq_length = seq_length
                
                self.pad_id = tokenizer.token_to_id("<pad>")
                self.unk_id = tokenizer.token_to_id("<unk>")
                print(f"    Special tokens - pad: {self.pad_id}, unk: {self.unk_id}")
                
                self.examples = []
                
                valid_texts = [text for text in dataset_split["text"] if text and text.strip() and len(text.strip()) > 10]
                print(f"    Processing {len(valid_texts)} valid texts...")
                
                for i, text in enumerate(valid_texts[:50]):  # Only process first 50 for testing
                    clean_text = text.strip().replace('\n', ' ').replace('\t', ' ')
                    clean_text = ' '.join(clean_text.split())
                    
                    if len(clean_text) < 20:
                        continue
                        
                    tokens = tokenizer.encode(clean_text).ids
                    print(f"      Text {i}: {len(clean_text)} chars -> {len(tokens)} tokens")
                    
                    if len(tokens) >= seq_length + 1:
                        # Create one sequence from this text
                        sequence = tokens[:seq_length + 1]
                        self.examples.append(sequence)
                        print(f"        ✓ Added sequence of length {len(sequence)}")
                    elif len(tokens) >= seq_length // 2:  # At least half the length
                        # Pad to required length
                        padded_tokens = tokens + [self.pad_id] * (seq_length + 1 - len(tokens))
                        self.examples.append(padded_tokens)
                        print(f"        ✓ Added padded sequence (original: {len(tokens)}, padded: {len(padded_tokens)})")
                    else:
                        print(f"        ✗ Text too short: {len(tokens)} tokens")
                
                print(f"    Created dataset with {len(self.examples)} sequences")
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                tokens = self.examples[idx]
                x = torch.tensor(tokens[:-1], dtype=torch.long)
                y = torch.tensor(tokens[1:], dtype=torch.long)
                y = torch.where(y == self.pad_id, torch.tensor(-100, dtype=torch.long), y)
                return {"input_ids": x, "labels": y}
        
        # Test dataset creation
        try:
            train_dataset = TestWikiTextDataset(dataset['train'], tokenizer, seq_length=64)
            print(f"  ✓ Successfully created train dataset with {len(train_dataset)} examples")
            
            if len(train_dataset) > 0:
                sample = train_dataset[0]
                print(f"  Sample data shape: input={sample['input_ids'].shape}, labels={sample['labels'].shape}")
                print(f"  Sample input tokens: {sample['input_ids'][:10].tolist()}")
                print(f"  Sample label tokens: {sample['labels'][:10].tolist()}")
            
        except Exception as e:
            print(f"  ✗ Dataset creation failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n=== SUMMARY ===")
        print("✓ WikiText-2 loading successful")
        print("✓ Tokenizer creation successful")
        print("✓ Dataset creation successful" if 'train_dataset' in locals() and len(train_dataset) > 0 else "✗ Dataset creation failed")
        
    except Exception as e:
        print(f"✗ Error during WikiText debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_wikitext_loading()