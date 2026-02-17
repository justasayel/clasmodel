#!/usr/bin/env python3
"""
Test inference with pre-trained Qwen2.5 (no training needed)
Shows how the classifier will work once fine-tuned
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("=" * 70)
print("🧪 DEMO: Qwen2.5 Classifier - Test Inference")
print("=" * 70)

# Classification prompt
SYSTEM_PROMPT = """You are an AI Government Data Classification Engine.
Classify the document strictly based on explicit evidence found.

CLASSIFICATION LEVELS:
- Top Secret: National security threats, sovereignty, intelligence operations
- Secret: Medium impact, large-scale system vulnerabilities  
- Restricted: Personal data, internal communications, single entity
- Public: Public distribution, no sensitivity indicators

OUTPUT FORMAT:
Classification Level: <Top Secret | Secret | Restricted | Public>
Sub-Level (if Restricted only): <A | B | C | N/A>
Impact Level: <High | Medium | Low | None>
Short Justification: <brief explanation>"""

# Test documents
TEST_DOCS = {
    "Public": """Subject: Weekly Team Meeting
    
Hi Team,

Our weekly sync is scheduled for Thursday at 2 PM. Please come prepared 
to discuss project progress and upcoming deadlines.

Best regards,
Manager""",
    
    "Restricted": """Subject: Employee Salary Review
    
CONFIDENTIAL
Date: February 17, 2026

Employee: John Smith (ID: E12345)
Current Salary: $85,000
Proposed Raise: 5% to $89,250

Medical Records on File - Approved for Review

This information is internal only.""",
    
    "Secret": """Subject: System Architecture Review
    
CLASSIFICATION: Secret

The enterprise database infrastructure contains significant vulnerabilities 
in the authentication layer. Critical systems across 30+ servers are at risk.

Executive Summary:
- SQL injection vectors in login module
- Unencrypted API endpoints
- Weak password hashing algorithm

Large-scale breach could impact all operational systems.""",
    
    "Top Secret": """Subject: NATIONAL SECURITY - CLASSIFIED
    
TOP SECRET - EXECUTIVE SECURE CHANNELS ONLY

From: National Cyber Defense Command
To: Cabinet Level Officials

Critical threat to national infrastructure detected. 

Intelligence Assessment:
- Sovereign nation-state coordinated attack
- Critical infrastructure at national scale targeted
- Power grid, water systems, emergency services affected
- National continuity at risk

This assessment affects national survival and government continuity."""
}

print("\n📥 Loading Qwen2.5-7B-Instruct model...")
print("   ⏳ First download will take ~30 seconds...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    print("   ✓ Model loaded!")
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("\n   💡 Make sure you have:")
    print("      - Internet connection")
    print("      - 20GB free disk space")
    print("      - torch & transformers installed")
    exit(1)

# Test classification
print("\n" + "=" * 70)
print("🧪 TESTING CLASSIFIER")
print("=" * 70)

for doc_type, document in TEST_DOCS.items():
    print(f"\n📄 Testing: {doc_type}")
    print("-" * 70)
    print(f"Document: {document[:100]}...")
    
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this document:\n\n{document}"}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt")
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\n🤖 Classification Result:")
        print(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("✅ DEMO COMPLETE!")
print("=" * 70)
print("\n💡 Next Steps:")
print("   1. Once model is fine-tuned, it will classify better")
print("   2. To train now: Use GPU (Google Colab recommended)")
print("   3. Fine-tuned model will be saved to: models/qwen2.5-classifier/")
print()
