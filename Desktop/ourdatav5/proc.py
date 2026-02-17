import os
import pandas as pd
import pdfplumber
from docx import Document
from sklearn.model_selection import train_test_split
import json

# --- إعداد المجلدات ---
ROOT_DIR = "."                   # المجلد الرئيسي مع مجلدات التصنيف
OUTPUT_DIR = "processed_dataset" # المجلد لحفظ CSV و JSONL
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- دوال استخراج النص ---
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

def extract_text_from_excel(file_path):
    df = pd.read_excel(file_path)
    text = ""
    for col in df.columns:
        # Convert to string and handle NaN values
        text += " ".join(df[col].astype(str).str.replace('nan', '')) + "\n"
    return text.strip()

def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    text = ""
    for col in df.columns:
        # Convert to string and handle NaN values
        text += " ".join(df[col].astype(str).str.replace('nan', '')) + "\n"
    return text.strip()

def extract_text(file_path):
    # Skip PDF files as they can be slow
    if file_path.endswith(".pdf"):
        return ""
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        return extract_text_from_excel(file_path)
    elif file_path.endswith(".csv"):
        return extract_text_from_csv(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    else:
        return ""

# --- دمج البيانات ---
all_data = []

for class_name in os.listdir(ROOT_DIR):
    class_path = os.path.join(ROOT_DIR, class_name)
    if os.path.isdir(class_path):
        print(f"Processing folder: {class_name}")
        for fname in os.listdir(class_path):
            if fname.startswith('.'):  # Skip hidden files
                continue
            file_path = os.path.join(class_path, fname)
            if fname.endswith(('.txt', '.csv', '.xlsx', '.xls', '.docx')):
                try:
                    text = extract_text(file_path)
                    if text.strip():  # Only add if text is not empty
                        all_data.append({
                            "file_name": fname,
                            "text": text,
                            "label": class_name
                        })
                        print(f"  ✓ {fname}")
                except Exception as e:
                    print(f"  ✗ Error reading {fname}: {e}")

print(f"\nTotal files collected: {len(all_data)}")

if len(all_data) == 0:
    print("No data found! Please check your folder structure.")
    exit(1)

df = pd.DataFrame(all_data)

# --- توازن البيانات لكل تصنيف ---
min_count = df['label'].value_counts().min()
balanced_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min_count, random_state=42))
print(f"Balanced dataset size: {len(balanced_df)} (each class has {min_count} samples)")

# --- تقسيم البيانات ---
train_df, test_df = train_test_split(balanced_df, test_size=0.30, random_state=42, stratify=balanced_df['label'])

# Save only to JSONL format to avoid CSV size issues
print(f"\nTrain: {len(train_df)} (70%), Test: {len(test_df)} (30%)")

# --- تحويل إلى JSONL جاهز للفائن-تيون Qwen2.5 ---
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

print("JSONL files ready for training!")
