# Qwen2.5 Government Data Classification Model

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
