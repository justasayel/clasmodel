#!/usr/bin/env python3
"""
Inference script - Classify documents using fine-tuned Qwen2.5 model
Usage:
  python inference.py --model models/qwen2.5-classifier --text "your document text"
  python inference.py --model models/qwen2.5-classifier --file document.txt
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json

# ============ CLASSIFICATION SYSTEM PROMPT ============
SYSTEM_PROMPT = """You are an AI Government Data Classification Engine.
Your task is to classify a document strictly based on explicit evidence found inside the provided content.
You must not speculate. You must not infer unstated intent. You must not assume missing context. You must not exaggerate impact.
If evidence is not explicitly present → DO NOT assume it exists.
If no clear sensitivity indicators are found → classify as Public.

CLASSIFICATION LEVELS:
1. Top Secret – National level security threats (sovereignty, national security, intelligence operations, military operations, critical infrastructure at national scale)
2. Secret – Medium impact affecting major systems/operations but not national survival
3. Restricted – Personal data, employee records, internal communications, single entity impact
4. Public – Public distribution, no sensitivity indicators

OUTPUT FORMAT (MUST BE EXACTLY THIS):
Classification Level: <Top Secret | Secret | Restricted | Public>
Sub-Level (if Restricted only): <A | B | C | N/A>
Impact Level: <High | Medium | Low | None>
Short Justification: <brief explanation with specific textual indicators>"""

def load_model(model_path):
    """Load fine-tuned model and tokenizer"""
    print(f"🤖 Loading model from {model_path}...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"  ✓ Model loaded on device: {next(model.parameters()).device}")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)

def classify_document(model, tokenizer, document_text):
    """Classify a document using the fine-tuned model"""
    
    # Prepare prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Classify this document:\n\n{document_text}"}
    ]
    
    # Format for Qwen chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate classification
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
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def parse_classification(response):
    """Parse classification response"""
    lines = response.strip().split('\n')
    
    result = {
        "classification_level": "Unknown",
        "sub_level": "N/A",
        "impact_level": "Unknown",
        "justification": "",
        "raw_response": response
    }
    
    for line in lines:
        if "Classification Level:" in line:
            result["classification_level"] = line.split(":", 1)[1].strip().split()[0]
        elif "Sub-Level" in line:
            result["sub_level"] = line.split(":", 1)[1].strip().split()[0]
        elif "Impact Level:" in line:
            result["impact_level"] = line.split(":", 1)[1].strip()
        elif "Short Justification:" in line:
            result["justification"] = line.split(":", 1)[1].strip()
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Classify documents using fine-tuned Qwen2.5 model"
    )
    parser.add_argument(
        "--model",
        default="models/qwen2.5-classifier",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--text",
        help="Document text to classify"
    )
    parser.add_argument(
        "--file",
        help="Path to document file to classify"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--batch",
        help="Batch classify files in directory"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    results = []
    
    # Single text classification
    if args.text:
        print(f"\n📄 Classifying text...")
        response = classify_document(model, tokenizer, args.text)
        result = parse_classification(response)
        results.append(result)
        
        print(f"\n✅ Classification Result:")
        print(f"  Level: {result['classification_level']}")
        print(f"  Sub-Level: {result['sub_level']}")
        print(f"  Impact: {result['impact_level']}")
        print(f"  Justification: {result['justification']}")
        
    # Single file classification
    elif args.file:
        print(f"\n📂 Reading file: {args.file}")
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print(f"  File size: {len(text)} chars")
            response = classify_document(model, tokenizer, text)
            result = parse_classification(response)
            results.append(result)
            
            print(f"\n✅ Classification Result:")
            print(f"  Level: {result['classification_level']}")
            print(f"  Sub-Level: {result['sub_level']}")
            print(f"  Impact: {result['impact_level']}")
            print(f"  Justification: {result['justification']}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            exit(1)
    
    # Batch classification
    elif args.batch:
        print(f"\n📁 Batch classifying files in: {args.batch}")
        batch_path = Path(args.batch)
        files = list(batch_path.glob("**/*.txt")) + list(batch_path.glob("**/*.csv"))
        
        print(f"  Found {len(files)} files to classify...")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()[:5000]  # Limit to first 5000 chars
                
                response = classify_document(model, tokenizer, text)
                result = parse_classification(response)
                result["file"] = str(file_path)
                results.append(result)
                
                print(f"  ✓ {file_path.name}: {result['classification_level']}")
            except Exception as e:
                print(f"  ✗ {file_path.name}: {e}")
    
    else:
        parser.print_help()
        return
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
