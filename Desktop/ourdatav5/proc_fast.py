import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# --- إعداد المجلدات ---
ROOT_DIR = "."                   # المجلد الرئيسي مع مجلدات التصنيف
OUTPUT_DIR = "processed_dataset" # المجلد لحفظ JSONL
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- دوال استخراج النص ---
def extract_text_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        text = ""
        for col in df.columns:
            # Convert to string and handle NaN values
            text += " ".join(df[col].astype(str).str.replace('nan', '')) + "\n"
        return text.strip()
    except Exception as e:
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    except Exception as e:
        return ""

# --- دمج البيانات ---
all_data = []

for class_name in os.listdir(ROOT_DIR):
    class_path = os.path.join(ROOT_DIR, class_name)
    # Skip hidden files, output folders, and system folders
    if os.path.isdir(class_path) and not class_name.startswith('.') and class_name not in ['processed_dataset', '.venv', '__pycache__']:
        print(f"Processing folder: {class_name}")
        for fname in os.listdir(class_path):
            if fname.startswith('.'):  # Skip hidden files
                continue
            file_path = os.path.join(class_path, fname)
            
            # Process only CSV and TXT files (fast files)
            if fname.endswith('.csv'):
                try:
                    text = extract_text_from_csv(file_path)
                    if text and len(text) > 10:  # Only add if text is substantial
                        all_data.append({
                            "file_name": fname,
                            "text": text,
                            "label": class_name
                        })
                        print(f"  ✓ {fname}")
                except Exception as e:
                    print(f"  ✗ {fname}: {str(e)[:50]}")
                    
            elif fname.endswith('.txt'):
                try:
                    text = extract_text_from_txt(file_path)
                    if text and len(text) > 10:  # Only add if text is substantial
                        all_data.append({
                            "file_name": fname,
                            "text": text,
                            "label": class_name
                        })
                        print(f"  ✓ {fname}")
                except Exception as e:
                    print(f"  ✗ {fname}: {str(e)[:50]}")

print(f"\nTotal files collected: {len(all_data)}")

if len(all_data) == 0:
    print("No data found! Please check your folder structure.")
    exit(1)

df = pd.DataFrame(all_data)

# --- توازن البيانات لكل تصنيف ---
label_counts = df['label'].value_counts()
print(f"Label distribution before balancing: \n{label_counts}")

# Don't balance if there are too few samples
if len(df) < 20:
    balanced_df = df.copy()
    print(f"Dataset small, skipping balancing")
else:
    min_count = label_counts.min()
    # Ensure min_count is at least 1
    min_count = max(1, min_count)
    
    # Balance dataset while preserving all columns
    balanced_data = []
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        if len(label_df) > 0:
            sample_size = min(len(label_df), min_count)
            sampled = label_df.sample(n=sample_size, random_state=42)
            balanced_data.append(sampled)
    
    balanced_df = pd.concat(balanced_data, ignore_index=True)

print(f"Balanced dataset size: {len(balanced_df)}")
print(f"Balanced dataset columns: {balanced_df.columns.tolist()}")

# --- تقسيم البيانات ---
if len(balanced_df) < 4:
    print(f"Warning: Dataset too small ({len(balanced_df)} samples), skipping stratification")
    train_df, test_df = train_test_split(balanced_df, test_size=0.30, random_state=42)
else:
    train_df, test_df = train_test_split(balanced_df, test_size=0.30, random_state=42, stratify=balanced_df['label'])

print(f"Train: {len(train_df)} (70%), Test: {len(test_df)} (30%)")

# --- تحويل إلى JSONL جاهز للفائن-تيون ---
def convert_to_jsonl(df, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            json_obj = {
                "prompt": f"Classify the following document according to the official Data Classification Policy:\n{row['text']}\n\nRespond in structured format: Classification Level, Sub-Level (if Restricted), Impact Level, Short Justification.",
                "completion": f" {row['label']}"
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

convert_to_jsonl(train_df, os.path.join(OUTPUT_DIR, "train.jsonl"))
convert_to_jsonl(test_df, os.path.join(OUTPUT_DIR, "test.jsonl"))

print(f"\n✅ JSONL files ready for training!")
print(f"  - Train: {os.path.join(OUTPUT_DIR, 'train.jsonl')}")
print(f"  - Test: {os.path.join(OUTPUT_DIR, 'test.jsonl')}")
