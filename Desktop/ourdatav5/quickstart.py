#!/usr/bin/env python3
"""
Quick start guide and installation helper for Qwen2.5 Classifier
"""

import sys
import subprocess
from pathlib import Path

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def check_requirements():
    """Check if required packages are installed"""
    print_section("🔍 CHECKING REQUIREMENTS")
    
    packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'datasets': 'datasets',
        'bitsandbytes': 'bitsandbytes (GPU optimization)',
    }
    
    missing = []
    for module, display_name in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        response = input("\nInstall missing packages? (y/n): ")
        if response.lower() == 'y':
            install_packages(missing)
    else:
        print(f"\n✅ All requirements satisfied!")

def install_packages(packages):
    """Install missing packages"""
    print("\n📦 Installing packages...")
    
    for package in packages:
        print(f"  Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"    ✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"    ✗ Failed to install {package}")

def show_usage():
    """Show usage instructions"""
    print_section("📚 QUICK START GUIDE")
    
    print("1️⃣  PREPARE DATA (Already Done!)")
    print("   Train data:  processed_dataset/train.jsonl (65 samples)")
    print("   Test data:   processed_dataset/test.jsonl (28 samples)")
    
    print("\n2️⃣  FINE-TUNE MODEL")
    print("   python train_model.py")
    print("   ⏱️  Time: ~30 min on GPU, hours on CPU")
    print("   💾 Output: models/qwen2.5-classifier/")
    
    print("\n3️⃣  CLASSIFY DOCUMENTS")
    print("   Single text:")
    print("     python inference.py --model models/qwen2.5-classifier \\")
    print("       --text 'Document text...'")
    print("")
    print("   From file:")
    print("     python inference.py --model models/qwen2.5-classifier \\")
    print("       --file document.txt")
    print("")
    print("   Batch classify:")
    print("     python inference.py --model models/qwen2.5-classifier \\")
    print("       --batch ./documents/ --output results.json")
    
    print("\n4️⃣  EXPORT MODEL")
    print("   To ZIP:")
    print("     python export.py --format zip --output export.zip")
    print("")
    print("   To USB Drive:")
    print("     python export.py --format usb --drive /Volumes/USB_DRIVE")
    print("")
    print("   To GitHub:")
    print("     python export.py --format github --repo https://github.com/user/repo")

def show_docker():
    """Show Docker instructions"""
    print_section("🐳 DOCKER DEPLOYMENT")
    
    dockerfile = '''FROM nvidia/cuda:12.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip
RUN pip install torch transformers datasets bitsandbytes

WORKDIR /app
COPY . /app

ENV CUDA_VISIBLE_DEVICES=0

CMD ["python3", "inference.py"]
'''
    
    print("Dockerfile:")
    print(dockerfile)
    
    print("Build & Run:")
    print("  docker build -t qwen-classifier .")
    print("  docker run --gpus all qwen-classifier --text 'document text'")

def show_api_example():
    """Show API deployment example"""
    print_section("🌐 REST API DEPLOYMENT")
    
    api_code = '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import load_model, classify_document
import torch

app = FastAPI(title="Qwen2.5 Classifier API")

# Load model on startup
model, tokenizer = load_model("models/qwen2.5-classifier")

class DocumentRequest(BaseModel):
    text: str = None
    file_path: str = None

class ClassificationResponse(BaseModel):
    classification_level: str
    sub_level: str
    impact_level: str
    justification: str

@app.post("/classify", response_model=ClassificationResponse)
async def classify_endpoint(request: DocumentRequest):
    """Classify a document"""
    try:
        if request.text:
            text = request.text
        elif request.file_path:
            with open(request.file_path, 'r') as f:
                text = f.read()
        else:
            raise HTTPException(status_code=400, detail="Provide text or file_path")
        
        response = classify_document(model, tokenizer, text)
        
        # Parse response
        result = parse_classification(response)
        return ClassificationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "qwen2.5-classifier"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    print("Save as: api_server.py")
    print(api_code)
    print("\nRun:")
    print("  pip install fastapi uvicorn")
    print("  python api_server.py")
    print("\nTest:")
    print("  curl -X POST http://localhost:8000/classify \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"text\": \"Your document text\"}'")

def show_troubleshooting():
    """Show troubleshooting guide"""
    print_section("🔧 TROUBLESHOOTING")
    
    issues = {
        "Out of Memory": [
            "• Reduce BATCH_SIZE in train_model.py",
            "• Use 8-bit quantization (enabled by default)",
            "• Use gradient accumulation",
            "• Run on GPU with more VRAM",
        ],
        "Slow Training": [
            "• Use GPU (NVIDIA CUDA compatible)",
            "• Reduce max_length in preprocessing",
            "• Increase batch_size if memory allows",
        ],
        "Model Download Issues": [
            "• Check internet connection",
            "• Ensure 50GB+ free disk space",
            "• Login to Hugging Face: huggingface-cli login",
            "• Set HF_HOME environment variable",
        ],
        "Slow Inference": [
            "• Use GPU acceleration",
            "• Reduce max_new_tokens (default 512)",
            "• Use model quantization",
        ],
    }
    
    for issue, solutions in issues.items():
        print(f"{issue}:")
        for solution in solutions:
            print(f"  {solution}")
        print()

def show_file_structure():
    """Show project file structure"""
    print_section("📁 PROJECT STRUCTURE")
    
    structure = """
project/
├── processed_dataset/
│   ├── train.jsonl           ← 65 training samples
│   ├── test.jsonl            ← 28 test samples
│   └── dataset_combined_balanced.csv
│
├── models/
│   └── qwen2.5-classifier/   ← Trained model (created after train_model.py)
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.model
│       └── training_config.json
│
├── train_model.py            ← Run: python train_model.py
├── inference.py              ← Run: python inference.py --help
├── export.py                 ← Run: python export.py --help
├── quickstart.py             ← This file
└── README.md                 ← Full documentation
"""
    
    print(structure)

def main():
    print("\n")
    print("  " + "="*66)
    print("  🚀 QWEN2.5 GOVERNMENT DATA CLASSIFIER - QUICK START")
    print("  " + "="*66)
    
    # Show file structure
    show_file_structure()
    
    # Check requirements
    check_requirements()
    
    # Show options
    print_section("📋 WHAT WOULD YOU LIKE TO DO?")
    print("  1. View Quick Start Guide")
    print("  2. View Docker Deployment")
    print("  3. View API Deployment")
    print("  4. View Troubleshooting")
    print("  5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        show_usage()
    elif choice == "2":
        show_docker()
    elif choice == "3":
        show_api_example()
    elif choice == "4":
        show_troubleshooting()
    elif choice == "5":
        print("\nGoodbye! 👋")
        return
    else:
        print("\n❌ Invalid choice")
        return
    
    print_section("📚 NEXT STEPS")
    print("  1. Install requirements:")
    print("     pip install -q torch transformers datasets bitsandbytes")
    print("")
    print("  2. Fine-tune the model:")
    print("     python train_model.py")
    print("")
    print("  3. Test classification:")
    print("     python inference.py --model models/qwen2.5-classifier --text 'test'")
    print("")
    print("  4. Export your model:")
    print("     python export.py --format zip --output model-export.zip")
    print("")
    print("✅ Full command reference: python inference.py --help")
    print()

if __name__ == "__main__":
    main()
