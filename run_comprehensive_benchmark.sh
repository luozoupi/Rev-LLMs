#!/bin/bash

# Comprehensive DiZO OPT vs Qwen3 Benchmark Runner
# Usage: ./run_comprehensive_benchmark.sh [scale] [tasks] [models]

SCALE=${1:-small}
TASKS=${2:-sst2,cola}
MODELS=${3:-all}

echo "Running comprehensive benchmark..."
echo "Scale: $SCALE"
echo "Tasks: $TASKS" 
echo "Models: $MODELS"

python comprehensive_dizo_qwen3_benchmark.py \
    --scale $SCALE \
    --tasks $TASKS \
    --models $MODELS \
    --full_eval \
    --output_dir ./benchmark_results_$(date +%Y%m%d_%H%M%S)

echo "Benchmark completed!"
