"""
Deploy fine-tuned Qwen2.5 model for document classification
Uses the trained model to classify documents with high accuracy
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FINETUNED_MODEL_PATH = "qwen_finetuned_classifier"

# Government Data Classification System Prompt
SYSTEM_PROMPT = """Government Data Classification Engine
(Strict Deterministic Version – Low Hallucination)

You are an AI Government Data Classification Engine.
Your task is to classify a document strictly based on explicit evidence found inside the provided content.
You must not speculate. You must not infer unstated intent. You must not assume missing context. You must not exaggerate impact.
If evidence is not explicitly present → DO NOT assume it exists.
If no clear sensitivity indicators are found → classify as Public.

STEP 1 – Evidence Extraction (Internal Reasoning Rule)
Before deciding, internally verify:
* Are there explicit national-level indicators?
* Are there explicit intelligence or military references?
* Are there explicit sovereign financial risk references?
* Are there explicit large-scale impact indicators?
* Are there explicit personal data references?
* Is the document clearly labeled confidential/secret?
* Is it clearly public-facing?
If none of the above are explicitly stated → default to lower classification.

STRICT ESCALATION RULE
Escalate classification level ONLY if:
1. Explicit national-level security indicators appear.
2. Explicit intelligence operations or classified handling terms appear.
3. Explicit sovereign financial collapse or national stability risks appear.
4. Explicit large-scale harm statements appear.
Mere use of words like:
* "strategy"
* "risk"
* "security"
* "confidential"
* "internal"
DOES NOT automatically mean Top Secret or Secret.
Context and impact scale must be clearly national or sovereign level.

CLASSIFICATION LEVELS

Top Secret – High Impact
Classify as Top Secret ONLY IF the content explicitly references:
* National Security
* Sovereignty
* National survival
* Continuity of government
* National command structures
* Classified intelligence feeds
* Intelligence collection methodologies
* Military operations
* Critical infrastructure at national scale
* Sovereign financial collapse
* National economic stability at risk
* Executive national crisis governance
* Top Secret handling protocols
* Cleared personnel only (national level)
AND the impact clearly affects:
* National security
* Sovereign stability
* Entire country infrastructure
* Intelligence systems
* Military command
* National-level financial systems
If national scale is NOT clearly stated → DO NOT classify as Top Secret.
Impact Level: High

Secret – Medium Impact
Classify as Secret if the document explicitly shows:
* Major economic impact (large scale)
* Significant operational damage
* Large contractual exposure
* Major cybersecurity architecture details
* System-wide security vulnerabilities
* High-level internal SOC simulation
* Explicit label: "Classification: Secret"
But impact does NOT clearly threaten national survival or sovereignty.
Impact Level: Medium

Restricted – Low Impact
Classify as Restricted if the content includes:
* Personal identity data
* Employee salary data
* Medical records
* Internal emails
* Vendor contracts
* Internal policies
* Internal investigations
* Department restructuring
* Business development discussions
* Legal disputes
* Financial tracking tables
* Examination materials
Sub-Levels:
Level A → Sector-wide impact
Level B → Multi-entity impact
Level C → Single entity or individual impact
If only one organization or person is affected → Level C.
Impact Level: Low

Public – No Impact
Classify as Public ONLY if:
* Content is clearly meant for public distribution
* Public reports
* Press releases
* Job postings
* Published financial summaries
* Public website content
* No sensitivity indicators detected
If no clear sensitive indicators appear → choose Public.
Impact Level: None

Keyword Assistance (Non-Automatic Trigger)
The following keywords assist classification BUT DO NOT automatically trigger escalation without context scale:
National Indicators:
* sovereignty
* national authority
* critical infrastructure
* national governance protocols
Intelligence Indicators:
* classified sources
* human intelligence
* cyber intelligence
* technical intelligence
* intelligence protocols
Sovereign Financial Indicators:
* sovereign financial sensitivity
* national economic stability
* cascading effects on economy
Handling Controls:
* Top Secret Channels Only
* secure environments
* classified mitigation strategies
These must clearly imply national-scale impact to escalate.

FINAL OUTPUT FORMAT (STRICT)
You must respond ONLY in this format:
Classification Level: <Top Secret / Secret / Restricted / Public>
Sub-Level (if Restricted): <A / B / C / N/A>
Impact Level: <High / Medium / Low / None>
Short Justification:
* List only explicit indicators found in the text
* Reference scale of impact (national / multi-entity / single entity)
* No speculation
* No assumptions

Anti-Hallucination Guardrails
* If unsure → choose the lower classification.
* If national scale is not explicitly mentioned → do not assume it.
* If impact scale is unclear → classify as Restricted Level C.
* Never fabricate unseen risks.
* Never upgrade classification based on tone alone.
* Always justify using exact textual indicators."""

print("=" * 70)
print("🚀 DOCUMENT CLASSIFICATION WITH FINE-TUNED QWEN2.5")
print("=" * 70)

# Load model and tokenizer
print("\n📥 Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(model, FINETUNED_MODEL_PATH)
model = model.merge_and_unload()  # Merge LoRA weights into base model

print("✅ Model loaded successfully")

def classify_document(document_text: str) -> dict:
    """
    Classify a document using the fine-tuned Qwen model
    
    Args:
        document_text: The document content to classify
        
    Returns:
        dict: Classification result with level, sub-level, impact, and justification
    """
    
    # Create chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Classify this document:\n\n{document_text}"}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and generate
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode response
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0]
    
    # Parse response
    result = parse_classification_response(response)
    return result

def parse_classification_response(response: str) -> dict:
    """
    Parse the model's classification response
    
    Args:
        response: The raw model output
        
    Returns:
        dict: Structured classification result
    """
    lines = response.strip().split('\n')
    result = {
        'classification_level': 'Public',
        'sub_level': 'N/A',
        'impact_level': 'None',
        'justification': '',
        'raw_response': response
    }
    
    for line in lines:
        line = line.strip()
        if line.startswith('Classification Level:'):
            result['classification_level'] = line.split(':', 1)[1].strip()
        elif line.startswith('Sub-Level'):
            result['sub_level'] = line.split(':', 1)[1].strip()
        elif line.startswith('Impact Level:'):
            result['impact_level'] = line.split(':', 1)[1].strip()
        elif line.startswith('Short Justification:'):
            result['justification'] = '\n'.join(lines[lines.index(line)+1:])
            break
    
    return result

def batch_classify_documents(documents: list, output_file: str = None):
    """
    Classify multiple documents and optionally save results
    
    Args:
        documents: List of document strings or dicts with 'text' key
        output_file: Optional file to save results as JSON
        
    Returns:
        list: Classification results for all documents
    """
    results = []
    
    for i, doc in enumerate(documents, 1):
        print(f"\n[{i}/{len(documents)}] Classifying document...")
        
        if isinstance(doc, dict):
            doc_text = doc.get('text', '')
            doc_name = doc.get('name', f'Document_{i}')
        else:
            doc_text = doc
            doc_name = f'Document_{i}'
        
        result = classify_document(doc_text)
        result['document_name'] = doc_name
        results.append(result)
        
        print(f"  📍 Classification: {result['classification_level']}")
        print(f"  📊 Impact: {result['impact_level']}")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")
    
    return results

# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("📋 EXAMPLE: Classifying sample documents")
    print("=" * 70)
    
    # Example documents
    sample_docs = [
        {
            "name": "Public Letter",
            "text": "Dear Employees,\n\nWe are pleased to announce that our company will be hosting a summer picnic on July 15th. All employees and their families are invited. Please register by June 30th.\n\nBest regards,\nHuman Resources"
        },
        {
            "name": "Restricted Doc",
            "text": "Employee Salary Review\nJohn Smith - $75,000\nJane Doe - $82,000\nInternal HR Use Only"
        },
        {
            "name": "Secret Document",
            "text": "Strategic Business Development\nConfidential merger discussion with potential acquisition target. National economic impact potential. Board approval required."
        }
    ]
    
    # Classify documents
    results = batch_classify_documents(
        sample_docs,
        output_file="classification_results.json"
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION SUMMARY")
    print("=" * 70)
    for result in results:
        print(f"\n📄 {result['document_name']}")
        print(f"   Classification: {result['classification_level']}")
        print(f"   Sub-Level: {result['sub_level']}")
        print(f"   Impact: {result['impact_level']}")
