#!/usr/bin/env python3
"""
Lightweight Qwen2.5 fine-tuning - Optimized for macOS/CPU
For GPU training, use Google Colab or cloud services
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import json
from pathlib import Path

print("=" * 70)
print("🚀 QWEN2.5 LIGHTWEIGHT FINE-TUNING (macOS/CPU Optimized)")
print("=" * 70)

# ============ CONFIGURATION ============
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "processed_dataset"
OUTPUT_DIR = "models/qwen2.5-classifier"
EPOCHS = 1  # Reduced for CPU
BATCH_SIZE = 1  # Minimal for CPU  
LEARNING_RATE = 2e-4
MAX_STEPS = 100  # Early stopping for testing

print("\n⚠️  WARNING: Training on CPU will be VERY SLOW")
print("   Estimated time: 8-12 hours for 1 epoch")
print("   Recommended: Use Google Colab (FREE GPU)")
print("\n💡 FASTER ALTERNATIVES:")
print("   1. Google Colab (Free GPU): https://colab.research.google.com")
print("   2. Lambda Labs (GPU): https://lambdalabs.com")
print("   3. AWS SageMaker (GPU): https://aws.amazon.com/sagemaker")

response = input("\n⏱️  Continue with CPU training? (y/n): ").lower()
if response != 'y':
    print("\n📖 To train faster:")
    print("  1. Open: https://colab.research.google.com")
    print("  2. Clone from GitHub: https://github.com/justasayel/clasmodel")
    print("  3. Run: !python train_model.py")
    exit(0)

# ============ CHECK HARDWARE ============
print("\n📊 System Check:")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
print(f"  Computing Device: CPU (Training will be slow!)")

# ============ LOAD DATA ============
print(f"\n📂 Loading data from {DATA_DIR}...")
train_path = os.path.join(DATA_DIR, "train.jsonl")
test_path = os.path.join(DATA_DIR, "test.jsonl")

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print(f"❌ Error: Data files not found!")
    exit(1)

dataset = load_dataset("json", data_files={
    "train": train_path,
    "test": test_path
})

print(f"  ✓ Train samples: {len(dataset['train'])}")
print(f"  ✓ Test samples: {len(dataset['test'])}")

# ============ LOAD MODEL & TOKENIZER ============
print(f"\n🤖 Loading {MODEL_NAME}...")
print("   ⏳ This will download ~14GB model (first time only)...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # CPU uses float32
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Optimize for CPU memory
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("  ✓ Model loaded successfully")
except Exception as e:
    print(f"  ❌ Error: {e}")
    print("\n💡 Troubleshooting:")
    print("  - Check internet connection")
    print("  - Ensure 50GB free disk space")
    print("  - Try: huggingface-cli login")
    exit(1)

# ============ PREPARE DATA ============
print(f"\n🔄 Preparing data...")

def preprocess_function(examples):
    """Prepare prompts for training"""
    texts = []
    for prompt, completion in zip(examples['prompt'], examples['completion']):
        full_text = f"{prompt}{completion}"
        texts.append(full_text)
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=1024,  # Reduced for CPU
        padding="max_length",
        return_tensors="pt"
    )
    
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

# Process only subset for CPU testing
print("  Sampling data for CPU processing...")
train_subset = dataset['train'].select(range(min(10, len(dataset['train']))))
test_subset = dataset['test'].select(range(min(5, len(dataset['test']))))

dataset_processed = {
    'train': train_subset.map(preprocess_function, batched=True, remove_columns=train_subset.column_names),
    'test': test_subset.map(preprocess_function, batched=True, remove_columns=test_subset.column_names)
}

print(f"  ✓ Data prepared (using {len(dataset_processed['train'])} training samples for speed)")

# ============ TRAINING SETUP ============
print(f"\n⚙️  Setting up training...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=5,
    weight_decay=0.01,
    learning_rate=LEARNING_RATE,
    save_strategy="steps",
    save_steps=10,
    eval_strategy="steps",
    eval_steps=10,
    logging_steps=1,
    max_steps=MAX_STEPS,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_processed['train'],
    eval_dataset=dataset_processed['test'],
    tokenizer=tokenizer,
)

# ============ TRAIN ============
print(f"\n🎓 Starting training...")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max steps: {MAX_STEPS}")
print(f"  Output: {OUTPUT_DIR}")
print("\n⏱️  Training in progress... (This will take several hours on CPU)")

try:
    trainer.train()
    print("\n✅ Training completed!")
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training error: {e}")
    print("💡 If out of memory, try reducing MAX_STEPS or BATCH_SIZE")
    exit(1)

# ============ SAVE MODEL ============
print(f"\n💾 Saving trained model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

config_path = os.path.join(OUTPUT_DIR, "training_config.json")
with open(config_path, 'w') as f:
    json.dump({
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_steps": MAX_STEPS,
        "device": "CPU",
    }, f, indent=2)

print(f"  ✓ Model saved to: {OUTPUT_DIR}")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📦 To use the model:")
print(f"   python inference.py --model {OUTPUT_DIR} --text 'your document'")
print("\n💡 For production training:")
print("   - Use GPU: Google Colab, Lambda Labs, or AWS")
print("   - Full dataset: Use all 65 training samples")
print("   - More epochs: Set EPOCHS=3 for better results")
print()
