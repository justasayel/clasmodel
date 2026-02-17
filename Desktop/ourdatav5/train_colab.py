#!/usr/bin/env python3
"""
Optimized Qwen2.5 Fine-tuning for Google Colab
Run: python train_colab.py
"""

import os
import torch
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import json

print("=" * 60)
print("🚀 QWEN2.5 FINE-TUNING FOR COLAB")
print("=" * 60)

# ============ CONFIGURATION ============
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "processed_dataset"
OUTPUT_DIR = "models/qwen2.5-classifier"
EPOCHS = 2
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

# Colab-specific settings
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

print("\n📊 System Info:")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============ LOAD DATA ============
print(f"\n📂 Loading data...")
try:
    dataset = load_dataset("json", data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "test": os.path.join(DATA_DIR, "test.jsonl")
    })
    print(f"  ✓ Train: {len(dataset['train'])} samples")
    print(f"  ✓ Test: {len(dataset['test'])} samples")
except Exception as e:
    print(f"  ❌ Failed to load data: {e}")
    sys.exit(1)

# ============ LOAD MODEL & TOKENIZER ============
print(f"\n🤖 Loading model (this may take 2-3 minutes)...")

try:
    # Load tokenizer first
    print("  📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_auth_token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Tokenizer ready")

    # Load model with minimal memory overhead
    print("  📥 Loading model from HuggingFace (this is the slow part)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=os.environ.get("HF_TOKEN"),
        low_cpu_mem_usage=True
    )
    print("  ✓ Model loaded")

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # ============ APPLY LORA (Memory efficient fine-tuning) ============
    print("\n⚙️  Applying LoRA for efficient training...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    print("  ✓ LoRA applied")

except Exception as e:
    print(f"\n❌ Error loading model!")
    print(f"Error: {str(e)[:300]}")
    print("\nTroubleshooting:")
    print("  1. Check internet connection")
    print("  2. Restart runtime if download failed")
    print("  3. Clear cache: !rm -rf ~/.cache/huggingface")
    sys.exit(1)

# ============ PREPARE DATA ============
print(f"\n🔄 Preparing data...")

def preprocess_function(examples):
    texts = []
    for prompt, completion in zip(examples['prompt'], examples['completion']):
        texts.append(f"{prompt}{completion}")
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=1024,  # Reduced from 2048 for speed
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset_processed = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names
)
print("  ✓ Data ready")

# ============ TRAINING ============
print(f"\n⚙️  Setting up training...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    warmup_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=5,
    fp16=True,
    remove_unused_columns=False,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_processed['train'],
    eval_dataset=dataset_processed['test'],
    tokenizer=tokenizer,
)

print(f"\n🎓 Starting training...")
print(f"  Duration: ~20-30 minutes on T4 GPU")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")

try:
    trainer.train()
    print("\n✅ Training completed!")
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted")
except Exception as e:
    print(f"\n❌ Training error: {e}")
    sys.exit(1)

# ============ SAVE MODEL ============
print(f"\n💾 Saving model...")

# Save LoRA adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save config
with open(os.path.join(OUTPUT_DIR, "training_config.json"), 'w') as f:
    json.dump({
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "method": "LoRA",
    }, f, indent=2)

print(f"  ✓ Model saved to: {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\n📦 Next steps:")
print(f"   1. Download the model folder")
print(f"   2. Use for inference with: python inference.py --model {OUTPUT_DIR}")
print()
