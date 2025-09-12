"""
Enhanced Training and Benchmarking Framework - Usage Guide
=========================================================

This guide shows how to use the new enhanced framework for training and benchmarking
reversible vs standard Qwen3 models on multiple benchmark datasets.

## Quick Start

### 1. Basic Training and Evaluation
Train on WikiText and enwik8, evaluate with full benchmark suite:

```bash
python enhanced_train_and_benchmark.py --datasets wikitext,enwik8 --models reversible,standard --full_eval
```

### 2. Code-Focused Training
Train specifically on code dataset:

```bash
python enhanced_train_and_benchmark.py --datasets code --models reversible,standard --full_eval
```

### 3. Multi-Domain Training
Train on all available datasets:

```bash
python enhanced_train_and_benchmark.py --datasets wikitext,enwik8,code,math,squad --models reversible,standard --full_eval
```

### 4. Evaluation-Only Mode
Use existing models with the comprehensive benchmark runner:

```bash
python run_advanced_benchmarks.py --models path/to/reversible_model.pt,path/to/standard_model.pt --tasks all
```

## Framework Architecture

### Dataset Support
- **WikiText**: General language modeling (word-level)
- **enwik8**: Character-level language modeling (challenging)
- **CodeParrot**: Code understanding and generation
- **GSM8K**: Mathematical reasoning
- **SQuAD**: Reading comprehension

### Model Types
- **Reversible**: Memory-efficient reversible transformers
- **Standard**: Traditional transformer architecture

### Benchmark Categories
1. **Text Generation**: BLEU/ROUGE metrics
2. **Language Understanding**: GLUE+ tasks
3. **Memory Efficiency**: Long-range dependency tests
4. **Advanced Tasks**: Domain-specific benchmarks

## Key Features

### 1. Fair Comparison Framework
- Same training procedure for both model types
- Dataset-specific hyperparameter optimization
- Consistent evaluation metrics

### 2. Comprehensive Evaluation
- Multiple benchmark suites
- Memory efficiency analysis
- Training stability metrics
- Cross-dataset performance comparison

### 3. Flexible Pipeline
- Train on single or multiple datasets
- Evaluation-only mode for existing models
- Configurable model architectures

## Expected Workflow

1. **Dataset Loading**: Automatic download and preprocessing
2. **Model Creation**: Both reversible and standard models
3. **Training**: Fine-tuning on each dataset with optimized configs
4. **Evaluation**: Comprehensive benchmarking across all metrics
5. **Analysis**: Performance comparison and insights

## Output Files

- `enhanced_training_results.json`: Complete training and evaluation results
- `comprehensive_benchmark_results.json`: Benchmark-only results
- Training curves and visualizations (if matplotlib available)

## Example Results Structure

```json
{
  "wikitext": {
    "reversible_qwen3": {
      "training_results": {
        "final_test_loss": 2.45,
        "final_perplexity": 11.6,
        "epochs_trained": 8
      },
      "evaluation_results": {
        "comprehensive_benchmarks": {...},
        "advanced_benchmarks": {...}
      }
    },
    "standard_qwen3": {
      "training_results": {
        "final_test_loss": 2.52,
        "final_perplexity": 12.4,
        "epochs_trained": 12
      },
      "evaluation_results": {...}
    }
  }
}
```

## Performance Insights

The framework automatically provides:

- Cross-dataset performance comparison
- Reversible vs standard architecture analysis
- Memory efficiency metrics
- Training convergence characteristics
- Task-specific performance breakdowns

## Requirements

- PyTorch with CUDA support
- Transformers library
- Datasets library
- Advanced benchmark modules (advanced_benchmarks.py, etc.)
- Sufficient GPU memory for training

## Troubleshooting

### Memory Issues
- Reduce batch size in training configs
- Use gradient accumulation for effective larger batches
- Enable mixed precision training (AMP)

### Dataset Loading Failures
- Check internet connection for dataset downloads
- Verify cache directory permissions
- Some datasets may require HuggingFace authentication

### Model Creation Errors
- Ensure qwen3_reversible_02_2.py is available
- Check CUDA availability for GPU training
- Verify model configuration parameters

## Advanced Usage

### Custom Dataset Configuration
Modify `BenchmarkDatasetConfig` in the script:

```python
custom_config = BenchmarkDatasetConfig(
    name='custom',
    task_type='language_modeling',
    seq_length=1024,
    num_train_samples=5000,
    vocab_size=32000
)
```

### Custom Training Configuration
Adjust `TrainingConfig` for specific needs:

```python
custom_training = TrainingConfig(
    dataset_name='custom',
    epochs=15,
    learning_rate=1e-4,
    batch_size=4,
    gradient_accumulation_steps=8
)
```

This framework provides a comprehensive solution for training and evaluating 
reversible vs standard transformer architectures across multiple domains and tasks.
