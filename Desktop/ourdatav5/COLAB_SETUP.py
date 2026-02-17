"""
COPY-PASTE THIS ENTIRE SCRIPT INTO GOOGLE COLAB
Run as a single cell for reliable training
"""

# ============ CELL 1: SETUP & CLONE ============
import subprocess
import os
import sys

print("=" * 60)
print("🚀 QWEN2.5 COLAB SETUP")
print("=" * 60)

# Step 1: Ensure clean state
print("\n📁 Setting up directories...")
os.system("cd /content && rm -rf clasmodel 2>/dev/null")
os.system("cd /content && mkdir -p clasmodel")

# Step 2: Clone repository
print("📥 Cloning repository...")
result = subprocess.run(
    "git clone https://github.com/justasayel/clasmodel.git /content/clasmodel",
    shell=True,
    capture_output=True,
    text=True
)
if result.returncode != 0:
    print(f"❌ Clone failed: {result.stderr}")
    sys.exit(1)
print("✓ Repository cloned")

# Step 3: Verify files
print("\n✓ Checking files...")
os.chdir("/content/clasmodel")
files_needed = ["train_colab.py", "processed_dataset/train.jsonl", "processed_dataset/test.jsonl"]
for f in files_needed:
    if os.path.exists(f):
        print(f"  ✓ {f}")
    else:
        print(f"  ❌ MISSING: {f}")

# Step 4: Install dependencies
print("\n📦 Installing dependencies (this takes ~2 minutes)...")
subprocess.run("pip install -q --no-cache-dir torch transformers datasets peft", shell=True)
print("✓ Dependencies installed")

print("\n" + "=" * 60)
print("✅ SETUP COMPLETE - READY TO TRAIN")
print("=" * 60)
print("\nNow run this in the NEXT CELL:\n")
print("  !python /content/clasmodel/train_colab.py\n")

# ============ IF YOU WANT TO RUN TRAINING IN SAME CELL ============
# Uncomment below to train immediately:

# print("\n🎓 Starting training...")
# os.chdir("/content/clasmodel")
# subprocess.run("python train_colab.py", shell=True)
