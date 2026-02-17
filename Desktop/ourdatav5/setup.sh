#!/bin/bash
# Setup script for Qwen2.5 Classifier

echo "=================================================="
echo "🚀 QWEN2.5 CLASSIFIER SETUP"
echo "=================================================="
echo ""

# Check Python
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install -q --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "=================================================="
echo "✅ SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Run quick start guide:"
echo "     python quickstart.py"
echo ""
echo "  3. Fine-tune model:"
echo "     python train_model.py"
echo ""
echo "  4. Test inference:"
echo "     python inference.py --model models/qwen2.5-classifier --text 'test'"
echo ""
