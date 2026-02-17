#!/bin/bash

# Fine-tune Qwen2.5 on your classification data
# This script sets up and runs the fine-tuning process

echo "============================================================"
echo "🚀 QWEN2.5 FINE-TUNING FOR DOCUMENT CLASSIFICATION"
echo "============================================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please activate your conda/venv first."
    exit 1
fi

# Install fine-tuning dependencies
echo ""
echo "📦 Installing fine-tuning dependencies..."
.venv/bin/pip install -q torch transformers datasets peft tqdm numpy pandas

# Check GPU availability
echo ""
echo "🖥️  Checking GPU availability..."
.venv/bin/python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

# Run fine-tuning
echo ""
echo "🔥 Starting fine-tuning..."
echo "This may take 30-60 minutes depending on your GPU"
echo ""

.venv/bin/python finetune_qwen.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Fine-tuning complete!"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "1. Deploy the model: .venv/bin/python deploy_classifier.py"
    echo "2. Test with your documents"
    echo ""
else
    echo ""
    echo "❌ Fine-tuning failed. Check the errors above."
    exit 1
fi
