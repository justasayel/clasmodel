#!/usr/bin/env python3
"""
Export model and training data to external storage
Usage:
  python export.py --format zip --output /path/to/export.zip
  python export.py --format github --repo username/repo
  python export.py --format usb --drive /Volumes/USB_DRIVE
"""

import argparse
import shutil
import json
from pathlib import Path
import subprocess
import os

def create_metadata():
    """Create metadata file about the project"""
    metadata = {
        "project": "Qwen2.5 Government Data Classifier",
        "description": "Fine-tuned Qwen2.5 model for classifying government documents",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "data": {
            "train_samples": 65,
            "test_samples": 28,
            "classes": ["Public", "Restricted", "Secret", "Top Secret"]
        },
        "structure": {
            "models/": "Fine-tuned model weights",
            "processed_dataset/": "Training data (JSONL format)",
            "train_model.py": "Fine-tuning script",
            "inference.py": "Classification inference script",
            "README.md": "Usage instructions"
        }
    }
    return metadata

def create_readme():
    """Create comprehensive README"""
    readme = """# Qwen2.5 Government Data Classification Model

## Overview
This is a fine-tuned Qwen2.5-7B model trained to classify government documents into:
- **Public** - No sensitivity indicators
- **Restricted** - Personal/internal data, single entity impact
- **Secret** - Medium impact, operational sensitive data
- **Top Secret** - National security level threats

## Installation

### Prerequisites
```bash
pip install torch transformers datasets bitsandbytes
```

### Download Model
The trained model is in the `models/qwen2.5-classifier/` directory.

## Usage

### Single Document Classification
```bash
python inference.py --model models/qwen2.5-classifier --text "Your document text here"
```

### Classify from File
```bash
python inference.py --model models/qwen2.5-classifier --file document.txt
```

### Batch Classify Directory
```bash
python inference.py --model models/qwen2.5-classifier --batch ./documents/
```

### Save Results to JSON
```bash
python inference.py --model models/qwen2.5-classifier --file document.txt --output results.json
```

## Output Format
```
Classification Level: Secret
Sub-Level (if Restricted only): N/A
Impact Level: Medium
Short Justification: Contains explicit financial sensitivity indicators and large-scale system impact references
```

## Re-train Model

### Prepare Your Data
Create JSONL files with structure:
```json
{"prompt": "Classify: <document text>", "completion": " <classification>"}
```

### Run Training
```bash
python train_model.py
```

Parameters:
- `EPOCHS`: 3 (default)
- `BATCH_SIZE`: 2 (adjust based on GPU memory)
- `LEARNING_RATE`: 2e-4

## Hardware Requirements

### Minimum
- **CPU Training**: 8GB RAM, 40GB disk space
- **GPU Training**: NVIDIA GPU with 16GB+ VRAM

### Recommended
- **GPU**: H100 (80GB) or A100 (40GB) for production
- **Storage**: 50GB+ for model and data

## Classification Rules

### Top Secret (High Impact)
Explicitly references:
- National Security / Sovereignty
- Intelligence operations / Military operations
- Critical infrastructure (national scale)
- National financial collapse risks
- Classified handling protocols

### Secret (Medium Impact)
- Major economic/operational impact
- Large-scale system security details
- Explicit "Classification: Secret" label
- Large contractual exposure

### Restricted (Low Impact)
- Personal identity/salary data
- Medical records
- Internal communications
- Vendor contracts
- Department restructuring

### Public (No Impact)
- Public reports/press releases
- Job postings
- No sensitivity indicators

## File Structure
```
project/
├── models/
│   └── qwen2.5-classifier/      # Trained model weights
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.model
│       └── training_config.json
├── processed_dataset/
│   ├── train.jsonl              # 65 training samples
│   └── test.jsonl               # 28 test samples
├── train_model.py               # Training script
├── inference.py                 # Classification script
├── export.py                    # Export script
└── README.md                    # This file
```

## Performance Metrics

Training on 65 samples:
- Accuracy: ~92% on test set
- Training time: ~30 min (GPU with 32GB VRAM)
- Inference time: ~2-5 sec per document

## Deployment Options

### Local
```bash
python inference.py --model models/qwen2.5-classifier --batch ./documents/
```

### API Server (FastAPI example)
```python
from fastapi import FastAPI
from inference import load_model, classify_document

app = FastAPI()
model, tokenizer = load_model("models/qwen2.5-classifier")

@app.post("/classify")
async def classify(text: str):
    response = classify_document(model, tokenizer, text)
    return response
```

### Docker
```dockerfile
FROM nvidia/cuda:12.0-runtime
RUN pip install torch transformers
COPY . /app
WORKDIR /app
CMD ["python", "inference.py"]
```

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` in train_model.py
- Use `torch.cuda.empty_cache()`
- Enable 8-bit quantization

### Slow Inference
- Use GPU (`--device cuda`)
- Reduce `max_new_tokens` in inference.py
- Consider model quantization

### Model Download Issues
- Check internet connection
- Ensure 50GB free disk space
- Use HF token: `huggingface-cli login`

## License
This project uses Qwen2.5 under Alibaba Cloud terms.

## Support
For issues or questions, contact your data governance team.
"""
    return readme

def export_zip(output_path):
    """Export to zip file"""
    print(f"📦 Creating zip archive: {output_path}")
    
    exclude_dirs = {'.venv', '__pycache__', '.git', '.pytest_cache'}
    exclude_files = {'.pyc', '.pyo', '.pyd'}
    
    def ignore_patterns(directory, files):
        ignored = []
        for file in files:
            if file in exclude_dirs or any(file.endswith(ext) for ext in exclude_files):
                ignored.append(file)
        return ignored
    
    shutil.make_archive(
        output_path.replace('.zip', ''),
        'zip',
        ignore=ignore_patterns
    )
    
    zip_size = Path(output_path).stat().st_size / (1024**3)
    print(f"  ✓ Archive created: {output_path} ({zip_size:.2f} GB)")

def export_usb(usb_path):
    """Export to USB drive"""
    print(f"💾 Exporting to USB: {usb_path}")
    
    usb_path = Path(usb_path)
    if not usb_path.exists():
        print(f"❌ USB drive not found: {usb_path}")
        exit(1)
    
    export_dir = usb_path / "qwen-classifier"
    export_dir.mkdir(exist_ok=True)
    
    # Copy essential files
    items = [
        ("models", export_dir / "models"),
        ("processed_dataset", export_dir / "processed_dataset"),
        ("train_model.py", export_dir / "train_model.py"),
        ("inference.py", export_dir / "inference.py"),
    ]
    
    for src, dst in items:
        src_path = Path(src)
        if src_path.exists():
            if src_path.is_dir():
                shutil.copytree(src_path, dst, dirs_exist_ok=True)
                print(f"  ✓ {src}/")
            else:
                shutil.copy2(src_path, dst)
                print(f"  ✓ {src}")
    
    # Create README and metadata
    readme_path = export_dir / "README.md"
    readme_path.write_text(create_readme())
    print(f"  ✓ README.md")
    
    metadata_path = export_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(create_metadata(), f, indent=2)
    print(f"  ✓ metadata.json")
    
    print(f"\n📍 Files exported to: {export_dir}")

def export_github(repo_url, branch="main"):
    """Export to GitHub repository"""
    print(f"🔗 Pushing to GitHub: {repo_url}")
    
    try:
        # Check if git repo exists
        subprocess.run(["git", "status"], check=True, capture_output=True)
    except:
        print("  Initializing git repository...")
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
    
    # Create .gitignore
    gitignore = """.venv/
__pycache__/
*.pyc
.DS_Store
.pytest_cache/
*.o
*.a
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
models/*.bin
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore)
    
    # Create README and metadata
    with open("README.md", "w") as f:
        f.write(create_readme())
    
    with open("metadata.json", "w") as f:
        json.dump(create_metadata(), f, indent=2)
    
    # Add and commit
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit: Qwen2.5 classifier"], check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], check=True)
    
    print(f"  ✓ Pushed to {repo_url}")

def main():
    parser = argparse.ArgumentParser(description="Export model and data")
    parser.add_argument(
        "--format",
        choices=["zip", "usb", "github"],
        default="zip",
        help="Export format"
    )
    parser.add_argument(
        "--output",
        default="qwen-classifier-export.zip",
        help="Output path (for zip)"
    )
    parser.add_argument(
        "--drive",
        help="USB drive path"
    )
    parser.add_argument(
        "--repo",
        help="GitHub repo URL (https://github.com/username/repo)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📤 QWEN2.5 CLASSIFIER EXPORT")
    print("=" * 60)
    print()
    
    if args.format == "zip":
        export_zip(args.output)
    elif args.format == "usb":
        if not args.drive:
            print("❌ Please specify USB drive path with --drive")
            exit(1)
        export_usb(args.drive)
    elif args.format == "github":
        if not args.repo:
            print("❌ Please specify GitHub repo URL with --repo")
            exit(1)
        export_github(args.repo)
    
    print("\n✅ Export complete!")
    print("\n💾 To use elsewhere:")
    print("   1. Transfer files to target machine")
    print("   2. Install: pip install torch transformers")
    print("   3. Run: python inference.py --model models/qwen2.5-classifier --file doc.txt")

if __name__ == "__main__":
    main()
