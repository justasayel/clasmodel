#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-7B on your data classification dataset
Run: python train_model.py
"""

import os
import torch
import traceback
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
import json
from pathlib import Path

# ============ CONFIGURATION ============
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "processed_dataset"
OUTPUT_DIR = "models/qwen2.5-classifier"
EPOCHS = 3
BATCH_SIZE = 2  # Reduced for limited memory
LEARNING_RATE = 2e-4

print("=" * 60)
print("🚀 QWEN2.5 FINE-TUNING PIPELINE")
print("=" * 60)

# ============ CHECK HARDWARE ============
print("\n📊 System Check:")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠️  No GPU detected - training will be very slow on CPU")
    print("  💡 Tip: Consider using Google Colab or Lambda Labs for GPU access")

# ============ LOAD DATA ============
print(f"\n📂 Loading data from {DATA_DIR}...")
train_path = os.path.join(DATA_DIR, "train.jsonl")
test_path = os.path.join(DATA_DIR, "test.jsonl")

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print(f"❌ Error: Data files not found!")
    print(f"   Expected: {train_path}")
    print(f"   Expected: {test_path}")
    exit(1)

# Load datasets
dataset = load_dataset("json", data_files={
    "train": train_path,
    "test": test_path
})

print(f"  ✓ Train samples: {len(dataset['train'])}")
print(f"  ✓ Test samples: {len(dataset['test'])}")

# ============ LOAD MODEL & TOKENIZER ============
print(f"\n🤖 Loading {MODEL_NAME}...")

try:
    # Load tokenizer first
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Tokenizer loaded")
    
    # Then load model with fallback strategy
    if torch.cuda.is_available():
        print("  GPU detected - loading model with 8-bit quantization...")
        model = None
        
        # Try 1: With 8-bit quantization (most memory efficient)
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True
            )
            print("  ✓ Loaded with 8-bit quantization")
        except Exception as e1:
            print(f"  ⚠️  Quantization failed, trying without...")
            
            # Try 2: Without quantization
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("  ✓ Loaded without quantization")
            except Exception as e2:
                print(f"  ⚠️  Float16 failed, trying float32...")
                
                # Try 3: Float32 fallback
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("  ✓ Loaded with float32")
    else:
        print("  No GPU detected - loading on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    
    print(f"  ✓ Model loaded successfully!")
    
except Exception as e:
    print(f"  ❌ Error loading model!")
    print(f"\n  Error: {str(e)[:200]}")
    traceback.print_exc()
    print(f"\n  💡 Troubleshooting for Colab:")
    print(f"     1. Install: !pip install --upgrade transformers torch")
    print(f"     2. Restart runtime: Runtime → Restart runtime")
    print(f"     3. Clear cache: !rm -rf ~/.cache/huggingface/transformers")
    print(f"     4. Try again")
    exit(1)

# ============ PREPARE DATA ============
print(f"\n🔄 Preparing data...")

def preprocess_function(examples):
    """Prepare prompts and completions for training"""
    texts = []
    for prompt, completion in zip(examples['prompt'], examples['completion']):
        # Format: prompt + completion
        full_text = f"{prompt}{completion}"
        texts.append(full_text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Set labels same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# Process dataset
dataset_processed = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names
)

print("  ✓ Data prepared")

# ============ TRAINING SETUP ============
print(f"\n⚙️  Setting up training...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=False,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=LEARNING_RATE,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_processed['train'],
    eval_dataset=dataset_processed['test'],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

# ============ TRAIN ============
print(f"\n🎓 Starting training...")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Output: {OUTPUT_DIR}")

try:
    trainer.train()
    print("\n✅ Training completed!")
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training error: {e}")
    exit(1)

# ============ SAVE MODEL ============
print(f"\n💾 Saving trained model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training config
config_path = os.path.join(OUTPUT_DIR, "training_config.json")
with open(config_path, 'w') as f:
    json.dump({
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_samples": len(dataset['train']),
        "test_samples": len(dataset['test']),
    }, f, indent=2)

print(f"  ✓ Model saved to: {OUTPUT_DIR}")
print(f"  ✓ Config saved to: {config_path}")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\n📦 To deploy the model:")
print(f"   python inference.py --model {OUTPUT_DIR}")
print("\n")
