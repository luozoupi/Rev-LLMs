# Qwen3 Evaluation Suite

A comprehensive evaluation framework for comparing Reversible vs Standard Qwen3 models, inspired by DiZO's evaluation methodology.

## 🎯 Overview

This evaluation suite provides:

- **Comprehensive Benchmarking**: GLUE tasks, custom memory stress tests, and domain-specific evaluations
- **Statistical Analysis**: Rigorous statistical testing with confidence intervals and effect sizes
- **Memory Efficiency Analysis**: Memory scaling analysis and efficiency comparisons
- **Zero-Order Optimization**: DiZO-style memory-efficient training with variance reduction
- **Detailed Reporting**: Automated report generation with visualizations

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to the evaluation suite directory
cd Qwen3_Evaluation_Suite

# Install required dependencies
pip install torch transformers datasets evaluate
pip install scipy scikit-learn matplotlib seaborn
pip install pyyaml tqdm numpy pandas
```

### 2. Quick Test

```bash
# Run quick test to verify everything works
python quick_test.py
```

### 3. Run Evaluation

```bash
# Quick evaluation (fast, minimal tasks)
python run_evaluation.py --config configs/quick_eval.yaml

# Comprehensive evaluation (full suite)
python run_evaluation.py --config configs/comprehensive_eval.yaml

# Custom evaluation
python run_evaluation.py --models reversible_qwen3,standard_qwen3 --tasks sst2,cola --device cuda
```

## 📁 Directory Structure

```
Qwen3_Evaluation_Suite/
├── run_evaluation.py              # Main evaluation script
├── zero_order_optimization.py     # DiZO-style ZO optimization
├── analysis_and_comparison.py     # Statistical analysis & reporting
├── quick_test.py                  # Test suite
├── configs/
│   ├── default_eval.yaml         # Default configuration
│   ├── quick_eval.yaml           # Fast testing configuration
│   └── comprehensive_eval.yaml   # Full evaluation configuration
└── README.md                     # This file
```

## 🔧 Configuration

### Model Configurations

The suite supports different model sizes:

- **Small**: 512d hidden, 8 layers (~13M parameters)
- **Medium**: 1024d hidden, 16 layers (~110M parameters)  
- **Large**: 2048d hidden, 24 layers (~775M parameters)

### Task Configurations

- **GLUE Tasks**: SST-2, CoLA, MRPC, RTE, QNLI, QQP, MNLI
- **Custom Tasks**: Memory stress tests, arithmetic reasoning
- **Memory Tests**: Sequence lengths from 256 to 8192 tokens

### Example Configuration

```yaml
# configs/custom_eval.yaml
model_names:
  - "reversible_qwen3"
  - "standard_qwen3"

model_sizes:
  - "small"
  - "medium"

tasks:
  - "sst2"
  - "cola"
  - "memory_stress"

num_epochs: 3
learning_rate: 3.0e-4
batch_size: 16
device: "cuda"
seeds: [42, 123, 456]
```

## 📊 Evaluation Features

### 1. Performance Comparison

- **Statistical Testing**: Welch's t-test, Mann-Whitney U, Kolmogorov-Smirnov
- **Effect Size Analysis**: Cohen's d with magnitude interpretation
- **Confidence Intervals**: 95% confidence intervals for all metrics
- **Multiple Seeds**: Robust evaluation with statistical significance

### 2. Memory Analysis

- **Scaling Analysis**: Power-law fitting for memory vs sequence length
- **Efficiency Comparison**: Memory usage comparison at different lengths
- **Maximum Capacity**: Find maximum supported sequence lengths
- **Memory Profiling**: Detailed memory usage tracking

### 3. Training Efficiency

- **Convergence Analysis**: Training dynamics and convergence rates
- **Time Profiling**: Training and inference speed comparison
- **Zero-Order Optimization**: Memory-efficient gradient estimation
- **Resource Utilization**: GPU memory and compute efficiency

### 4. Advanced Features

- **DiZO-style ZO Training**: Memory-efficient zero-order optimization
- **Gradient Checkpointing**: Memory optimization during training
- **Mixed Precision**: Automatic mixed precision support
- **Visualization**: Automated plot generation for all metrics

## 📈 Results and Reporting

### Automated Reports

The suite generates:

1. **Comprehensive Report**: Markdown report with statistical analysis
2. **Performance Plots**: Bar charts comparing model performance
3. **Memory Scaling Plots**: Log-log plots of memory usage
4. **Training Dynamics**: Loss curves and training time analysis
5. **Statistical Tables**: Detailed tables with significance testing

### Example Output

```
Qwen3 Evaluation Report: Reversible vs Standard Models
====================================================

## Executive Summary
- Total Models Evaluated: 12
- Successful Evaluations: 11
- Failed Evaluations: 1
- Success Rate: 91.7%

## Performance Comparison
| Task | Reversible | Standard | Gap    | Effect Size | Significant |
|------|------------|----------|--------|-------------|-------------|
| SST2 | 0.847      | 0.851    | -0.004 | -0.12       | ✗           |
| COLA | 0.762      | 0.748    | +0.014 | +0.31       | ✓           |

## Memory Analysis
- Reversible models achieve average memory savings of 15.3%
- Maximum sequence length: Reversible (4096) vs Standard (2048)
- Memory scaling: Reversible (1.45x) vs Standard (1.89x)
```

## 🛠️ Advanced Usage

### Custom Model Integration

```python
from run_evaluation import ModelFactory

# Register custom model
class CustomModelFactory(ModelFactory):
    @classmethod
    def create_model(cls, model_name, model_size, device):
        if model_name == "my_custom_model":
            return MyCustomModel(config)
        return super().create_model(model_name, model_size, device)
```

### Zero-Order Optimization

```python
from zero_order_optimization import create_zo_trainer_for_qwen3

# Create ZO trainer
trainer = create_zo_trainer_for_qwen3(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    enhanced_zo=True,
    zo_eps=1e-3,
    learning_rate=1e-3
)

# Train with zero-order optimization
results = trainer.train()
```

### Custom Analysis

```python
from analysis_and_comparison import ComprehensiveReportGenerator

# Generate custom report
report_generator = ComprehensiveReportGenerator(output_dir="my_results")
report_path = report_generator.generate_full_report(evaluation_results)
```

## 🔍 Comparison with DiZO

| Feature | DiZO | Qwen3 Evaluation Suite |
|---------|------|------------------------|
| Zero-Order Optimization | ✓ | ✓ (Enhanced with variance reduction) |
| Memory Efficiency | ✓ | ✓ (Detailed scaling analysis) |
| Statistical Testing | Basic | ✓ (Comprehensive with effect sizes) |
| Model Architecture | OPT/LLaMA | ✓ (Reversible Qwen3) |
| Visualization | Limited | ✓ (Comprehensive plots) |
| Reproducibility | ✓ | ✓ (Multiple seeds, fixed configs) |

## 📋 Requirements

### Core Dependencies

```txt
torch>=1.13.0
transformers>=4.21.0
datasets>=2.0.0
evaluate>=0.4.0
scipy>=1.9.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
pyyaml>=6.0
tqdm>=4.64.0
numpy>=1.21.0
pandas>=1.4.0
```

### Optional Dependencies

```txt
rouge-score        # For ROUGE metrics
nltk              # For BLEU metrics
accelerate        # For multi-GPU training
wandb             # For experiment tracking
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure parent directory models are available
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/Rev-LLMs"
   ```

2. **CUDA Out of Memory**: Reduce batch size or use CPU
   ```yaml
   batch_size: 4
   device: "cpu"
   ```

3. **Dataset Loading**: Some GLUE tasks require internet connection
   ```bash
   # Pre-download datasets
   python -c "from datasets import load_dataset; load_dataset('glue', 'sst2')"
   ```

4. **Tokenizer Issues**: The suite handles most tokenizer problems automatically
   ```python
   # Manual tokenizer setup if needed
   tokenizer.pad_token = tokenizer.eos_token
   ```

## 🤝 Contributing

1. **Add New Tasks**: Extend `TaskRegistry` in `run_evaluation.py`
2. **Add Models**: Extend `ModelFactory.create_model()`
3. **Add Metrics**: Extend analysis modules
4. **Add Visualizations**: Extend `VisualizationEngine`

### Example: Adding Custom Task

```python
# In run_evaluation.py
class TaskRegistry:
    CUSTOM_TASKS = {
        'my_task': {'description': 'My custom task'},
        # ... existing tasks
    }
    
    def _evaluate_custom_task(self, model, task_name):
        if task_name == "my_task":
            # Implement custom evaluation
            return custom_evaluation_results
```

## 📚 References

- [DiZO Paper](https://arxiv.org/abs/2310.07177) - Memory-Efficient Zero-Order Optimization
- [Reversible Networks](https://arxiv.org/abs/1707.04585) - Reversible Residual Networks
- [GLUE Benchmark](https://gluebenchmark.com/) - General Language Understanding Evaluation

## 📄 License

This evaluation suite follows the same license as the parent project.

---

**Happy Evaluating! 🎉**

For questions or issues, please check the troubleshooting section or create an issue in the repository.
