"""
Fine-tune Qwen2.5-7B-Instruct on document classification task
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import json

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "qwen_finetuned_classifier"
TRAIN_FILE = "processed_dataset/train.jsonl"
EVAL_FILE = "processed_dataset/test.jsonl"

print("=" * 60)
print("🚀 FINE-TUNING QWEN2.5 FOR DOCUMENT CLASSIFICATION")
print("=" * 60)

# 1. Load tokenizer and model
print("\n📥 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Setup LoRA for efficient fine-tuning
print("⚙️  Setting up LoRA configuration...")
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,  # LoRA scaling
    target_modules=["q_proj", "v_proj"],  # Target attention modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Load and process datasets
print(f"\n📚 Loading datasets...")

def format_dataset(examples):
    """Format JSONL data for training"""
    texts = []
    for prompt, completion in zip(examples['prompt'], examples['completion']):
        # Format as chat for Qwen
        text = f"{prompt}{completion}</s>"
        texts.append(text)
    return {'text': texts}

# Load datasets
train_dataset = load_dataset('json', data_files=TRAIN_FILE)['train']
eval_dataset = load_dataset('json', data_files=EVAL_FILE)['train']

print(f"  Train samples: {len(train_dataset)}")
print(f"  Eval samples: {len(eval_dataset)}")

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=2048,
    )

train_dataset = train_dataset.map(
    lambda x: {'text': [f"{p}{c}</s>" for p, c in zip(x['prompt'], x['completion'])]},
    batched=True
)
eval_dataset = eval_dataset.map(
    lambda x: {'text': [f"{p}{c}</s>" for p, c in zip(x['prompt'], x['completion'])]},
    batched=True
)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# 4. Training setup
print("\n⚙️  Setting up training configuration...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Adjust if you have more VRAM
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=20,
    save_steps=50,
    save_total_limit=2,
    learning_rate=5e-5,
    fp16=True,
    gradient_checkpointing=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 5. Train
print("\n🔥 Starting fine-tuning...")
trainer.train()

# 6. Save the fine-tuned model
print(f"\n💾 Saving fine-tuned model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save LoRA config
lora_config.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 60)
print("✅ Fine-tuning complete!")
print(f"📁 Model saved to: {OUTPUT_DIR}")
print("=" * 60)
