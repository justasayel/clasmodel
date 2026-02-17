import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# --- إعداد المجلدات ---
ROOT_DIR = "."                   
OUTPUT_DIR = "processed_dataset" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- دوال استخراج النص ---
def extract_text_from_csv(file_path):
    """Extract ALL content from CSV file"""
    try:
        df = pd.read_csv(file_path)
        # Convert entire dataframe to string, preserving structure
        text = df.to_string()
        return text.strip()
    except Exception as e:
        print(f"    Error reading CSV: {e}")
        return ""

def extract_text_from_txt(file_path):
    """Extract ALL content from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
        return content
    except Exception as e:
        print(f"    Error reading TXT: {e}")
        return ""

# --- دمج البيانات ---
all_data = []
file_count = {
    'csv': 0,
    'txt': 0,
    'skipped': 0
}

for class_name in os.listdir(ROOT_DIR):
    class_path = os.path.join(ROOT_DIR, class_name)
    # Skip hidden files, output folders, and system folders
    if os.path.isdir(class_path) and not class_name.startswith('.') and class_name not in ['processed_dataset', '.venv', '__pycache__']:
        print(f"\n📁 Processing folder: {class_name}")
        for fname in os.listdir(class_path):
            if fname.startswith('.'):  # Skip hidden files
                continue
            file_path = os.path.join(class_path, fname)
            
            # Process CSV files
            if fname.endswith('.csv'):
                try:
                    text = extract_text_from_csv(file_path)
                    if text and len(text) > 20:  # Only add if substantial content
                        all_data.append({
                            "file_name": fname,
                            "text": text,
                            "label": class_name
                        })
                        file_count['csv'] += 1
                        print(f"  ✓ {fname} ({len(text)} chars)")
                    else:
                        file_count['skipped'] += 1
                except Exception as e:
                    print(f"  ✗ {fname}: {str(e)[:60]}")
                    file_count['skipped'] += 1
                    
            # Process TXT files
            elif fname.endswith('.txt'):
                try:
                    text = extract_text_from_txt(file_path)
                    if text and len(text) > 20:  # Only add if substantial content
                        all_data.append({
                            "file_name": fname,
                            "text": text,
                            "label": class_name
                        })
                        file_count['txt'] += 1
                        print(f"  ✓ {fname} ({len(text)} chars)")
                    else:
                        file_count['skipped'] += 1
                except Exception as e:
                    print(f"  ✗ {fname}: {str(e)[:60]}")
                    file_count['skipped'] += 1

print(f"\n{'='*60}")
print(f"📊 COLLECTION SUMMARY:")
print(f"  CSV files: {file_count['csv']}")
print(f"  TXT files: {file_count['txt']}")
print(f"  Skipped: {file_count['skipped']}")
print(f"  Total documents: {len(all_data)}")
print(f"{'='*60}\n")

if len(all_data) == 0:
    print("❌ No data found! Please check your folder structure.")
    exit(1)

df = pd.DataFrame(all_data)

# --- توازن البيانات لكل تصنيف ---
label_counts = df['label'].value_counts()
print(f"📈 Label distribution before balancing:")
print(label_counts)
print()

# Balance dataset while preserving all columns
balanced_data = []
for label in sorted(df['label'].unique()):
    label_df = df[df['label'] == label]
    if len(label_df) > 0:
        balanced_data.append(label_df)

balanced_df = pd.concat(balanced_data, ignore_index=True)

print(f"✅ Balanced dataset size: {len(balanced_df)} documents")
print(f"   Label distribution:")
print(balanced_df['label'].value_counts())
print()

# --- تقسيم البيانات ---
if len(balanced_df) < 4:
    print(f"⚠️  Dataset small, skipping stratification")
    train_df, test_df = train_test_split(balanced_df, test_size=0.30, random_state=42)
else:
    train_df, test_df = train_test_split(balanced_df, test_size=0.30, random_state=42, stratify=balanced_df['label'])

train_count = len(train_df)
test_count = len(test_df)
train_pct = (train_count / (train_count + test_count)) * 100
test_pct = (test_count / (train_count + test_count)) * 100

print(f"📊 TRAIN/TEST SPLIT:")
print(f"  Train: {train_count} samples ({train_pct:.1f}%)")
print(f"  Test:  {test_count} samples ({test_pct:.1f}%)")
print(f"  Total: {train_count + test_count} samples")
print()

# --- تحويل إلى JSONL جاهز للفائن-تيون ---
def convert_to_jsonl(df, output_path):
    """Convert dataframe to JSONL format"""
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            json_obj = {
                "prompt": f"Classify the following document according to the official Data Classification Policy:\n\n{row['text']}\n\nRespond in structured format: Classification Level, Sub-Level (if Restricted), Impact Level, Short Justification.",
                "completion": f" {row['label']}"
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print("💾 Converting to JSONL format...")
convert_to_jsonl(train_df, os.path.join(OUTPUT_DIR, "train.jsonl"))
convert_to_jsonl(test_df, os.path.join(OUTPUT_DIR, "test.jsonl"))

print(f"\n{'='*60}")
print(f"✅ SUCCESS! Files ready for training:")
print(f"{'='*60}")
print(f"  📄 Train:  {os.path.join(OUTPUT_DIR, 'train.jsonl')} ({train_count} samples)")
print(f"  📄 Test:   {os.path.join(OUTPUT_DIR, 'test.jsonl')} ({test_count} samples)")
print(f"{'='*60}\n")
