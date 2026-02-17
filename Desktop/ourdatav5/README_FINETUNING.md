# Qwen2.5 Document Classification System

Fine-tune and deploy Qwen2.5-7B-Instruct for government document classification.

## Overview

This system provides:
1. **Data Preparation** - Convert your documents to train/test JSONL format
2. **Model Fine-tuning** - Use LoRA to efficiently fine-tune Qwen2.5 on your data
3. **Deployment** - Classify documents with the trained model using a strict classification framework

## Files

### Data Processing
- `proc_complete.py` - Process and merge all documents, split into 70% train / 30% test
- `processed_dataset/train.jsonl` - 65 training documents
- `processed_dataset/test.jsonl` - 28 test documents

### Model Training & Deployment
- `finetune_qwen.py` - Fine-tune model using LoRA
- `deploy_classifier.py` - Deploy and use the fine-tuned model
- `qwen_finetuned_classifier/` - Fine-tuned model output (created after training)

### Setup
- `requirements_finetuning.txt` - Python dependencies
- `run_finetuning.sh` - Automated fine-tuning script

## Requirements

- GPU with at least 16GB VRAM (24GB+ recommended for Qwen2.5-7B)
- CUDA 11.8+ 
- Python 3.8+

## Quick Start

### Step 1: Install Dependencies
```bash
pip install torch transformers datasets peft tqdm
```

### Step 2: Prepare Data (Already Done ✅)
Your data has been processed into:
- `processed_dataset/train.jsonl` (65 documents)
- `processed_dataset/test.jsonl` (28 documents)

### Step 3: Fine-tune the Model
```bash
python finetune_qwen.py
```

This will:
- Load Qwen2.5-7B-Instruct
- Apply LoRA for efficient training
- Train for 3 epochs with evaluation
- Save to `qwen_finetuned_classifier/`

**Estimated time:** 30-90 minutes depending on GPU

### Step 4: Deploy & Classify
```bash
python deploy_classifier.py
```

This will:
- Load the fine-tuned model
- Classify sample documents
- Save results to `classification_results.json`

## Classification Framework

The system uses a **strict deterministic classification** with these levels:

### 1. **Top Secret** (High Impact)
Requires explicit national-level indicators:
- National security, sovereignty, military operations
- Intelligence collection/handling
- Critical infrastructure at national scale
- Sovereign financial collapse risk
- Continuity of government impact

### 2. **Secret** (Medium Impact)
Requires explicit indicators:
- Major economic impact (large scale)
- Significant operational damage
- Large contractual exposure
- System-wide security vulnerabilities
- NOT national-scale impact

### 3. **Restricted** (Low Impact)
For sensitive organizational data:
- Personal identity/salary data
- Medical records
- Internal emails/policies
- Vendor contracts
- Sub-levels: A (sector-wide), B (multi-entity), C (single entity)

### 4. **Public** (No Impact)
- Public-facing materials
- Press releases, job postings
- Published reports
- No sensitivity indicators

## Key Features

✅ **Strict Evidence-Based** - Only classifies based on explicit content  
✅ **Anti-Hallucination** - Defaults to lower classification when unsure  
✅ **LoRA Efficient** - Trains quickly on consumer GPUs  
✅ **Deterministic** - Consistent, reproducible results  
✅ **Low Memory** - 16GB VRAM sufficient with gradient checkpointing  

## Example Usage

```python
from deploy_classifier import classify_document

# Classify a document
doc_text = "..."  # Your document content
result = classify_document(doc_text)

print(f"Classification: {result['classification_level']}")
print(f"Impact: {result['impact_level']}")
print(f"Justification: {result['justification']}")
```

## Advanced Configuration

### Adjust Fine-tuning Parameters

Edit `finetune_qwen.py`:
```python
training_args = TrainingArguments(
    num_train_epochs=3,              # Increase for more epochs
    per_device_train_batch_size=2,   # Increase if more VRAM
    learning_rate=5e-5,              # Adjust learning rate
    gradient_accumulation_steps=4,   # Increase for larger batches
)
```

### Modify Classification Prompt

Edit the `SYSTEM_PROMPT` in `deploy_classifier.py` to adjust classification rules.

## Troubleshooting

### Out of Memory Error
```python
# In finetune_qwen.py, reduce batch size:
per_device_train_batch_size=1  # Instead of 2
gradient_accumulation_steps=8  # Increase to compensate
```

### Model Loading Issues
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/
python finetune_qwen.py
```

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Performance Tips

1. **GPU Optimization**
   - Use `torch_dtype=torch.float16` for faster training
   - Enable `gradient_checkpointing=True` to reduce memory

2. **Data Optimization**
   - Start with smaller dataset for testing
   - Use balanced classes (already done ✅)

3. **Inference Speed**
   - Use merged LoRA weights (done in deploy.py)
   - Reduce `max_new_tokens` for faster generation

## Output Format

Classification results are provided in strict format:

```
Classification Level: Secret
Sub-Level (if Restricted): N/A
Impact Level: Medium
Short Justification:
* Explicit label "Classification: Secret" found
* References major contractual exposure
* Impact at organizational level (multi-entity)
* No national-scale indicator
```

## Next Steps

1. ✅ Data prepared (65 train, 28 test documents)
2. ⏳ Run `python finetune_qwen.py` to train
3. ⏳ Run `python deploy_classifier.py` to classify
4. 📊 Review `classification_results.json` for results
5. 🔄 Iterate: Adjust prompt, re-train if needed

## Support

For issues with:
- **Data processing**: Check `processed_dataset/` files
- **Fine-tuning**: Monitor logs in `./logs/`
- **Classification**: Review raw predictions in output JSON

Good luck! 🚀
