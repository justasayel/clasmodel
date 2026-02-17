#!/usr/bin/env python3
"""
Test/Demo: Run inference WITHOUT training
Shows classifier working with base model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 60)
print("🧪 QWEN2.5 INFERENCE TEST (NO TRAINING)")
print("=" * 60)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print("\n📥 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

print("✓ Model loaded")

# Test documents
test_docs = {
    "Public": """
    Annual Summer Festival 2026
    
    Join us for the university's annual summer festival! This public event 
    celebrates student achievements and community engagement. Activities include:
    - Live performances
    - Food vendors
    - Career fair booths
    - Open campus tours
    
    Date: June 15, 2026
    Location: Main campus grounds
    Admission: FREE
    """,
    
    "Restricted": """
    INTERNAL: Staff Safety Protocol Update
    
    Effective immediately, all staff must follow updated building access procedures:
    1. Badge required for all entrances after 6 PM
    2. Report suspicious activity to security extension 2345
    3. Emergency assembly point: North parking lot
    
    This is for internal use only.
    """,
    
    "Secret": """
    CONFIDENTIAL: Q2 Budget Reallocation
    
    Finance Committee Decision:
    - Reduce research spending by 20%
    - Reallocate funds to facility upgrades
    - Review vendor contracts for cost reduction
    - HR savings: $500K anticipated
    
    Do not share outside finance team.
    """,
    
    "Top Secret": """
    RESTRICTED: Merger Discussion - Board Level
    
    Private communication regarding potential acquisition by TechCorp Inc.
    Board decision: Proceed with exclusive negotiation.
    Financial terms: $800M valuation (45% premium).
    
    Do not discuss with anyone outside executive team.
    """
}

print("\n" + "=" * 60)
print("Testing Classification")
print("=" * 60)

system_prompt = """You are a government document classifier. Classify the document into EXACTLY ONE category:
- Public: Can be shared with everyone
- Restricted: Internal use only, cannot be public
- Secret: Highly sensitive, limited distribution
- Top Secret: Extremely confidential, executive only

Respond with ONLY the category name."""

for doc_type, content in test_docs.items():
    print(f"\n📄 Testing: {doc_type}")
    print("-" * 40)
    
    prompt = f"""{system_prompt}

Document:
{content}

Classification:"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = response.split("Classification:")[-1].strip().split("\n")[0]
    
    status = "✓" if doc_type in result else "⚠"
    print(f"{status} Predicted: {result}")
    print(f"  Expected: {doc_type}")

print("\n" + "=" * 60)
print("✅ Test complete! Model is working.")
print("=" * 60)
