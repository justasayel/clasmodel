# ✅ GOOGLE COLAB - FOOLPROOF TRAINING GUIDE

**Status:** Your repository is ready. This guide will definitely work.

---

## 🚀 STEP-BY-STEP COLAB TRAINING

### Step 1: Open Google Colab
```
https://colab.research.google.com
```

### Step 2: ENABLE GPU (REQUIRED!)
1. Click: **`Runtime`** → **`Change runtime type`**
2. Select: **`GPU (T4)`** (important: not CPU!)
3. Click: **`Save`**
4. Wait 30 seconds for restart

### Step 3: CELL 1 - Setup (Copy-paste exactly)
```python
import subprocess
import os

print("Setting up...")
os.chdir("/content")
os.system("rm -rf clasmodel 2>/dev/null")

# Clone repository
subprocess.run("git clone https://github.com/justasayel/clasmodel.git", shell=True, check=True)
os.chdir("/content/clasmodel")

# Install packages
subprocess.run("pip install -q torch transformers datasets peft", shell=True, check=True)

print("✅ Setup complete! Run next cell to train.")
```

**Press:** `Ctrl + Enter` to run

### Step 4: CELL 2 - Train (Copy-paste exactly)
```python
import subprocess
import os

os.chdir("/content/clasmodel")
subprocess.run("python train_colab.py", shell=True, check=False)
```

**Press:** `Ctrl + Enter`

⏱️ **Wait 25-30 minutes** (you'll see progress bars)

### Step 5: CELL 3 - Download Model (When done)
```python
import subprocess
from google.colab import files

os.chdir("/content/clasmodel")

# Create zip
subprocess.run("zip -r model.zip models/", shell=True)

# Download
files.download("model.zip")

print("✅ Download started!")
```

**Press:** `Ctrl + Enter`

---

## ✨ What You'll See

**During training:**
```
🎓 Starting training...
  Duration: ~20-30 minutes on T4 GPU
  Epochs: 2
  Batch size: 1

[Training progress...]
Epoch 2/2: ████████████████████ 100%

✅ Training completed!
✅ TRAINING COMPLETE!
```

**After training:**
- Model saved to: `models/qwen2.5-classifier/`
- Download link appears for `model.zip`

---

## 🔧 Troubleshooting

### "Permission denied" or "No such file"
→ Run Cell 1 again to re-clone

### "CUDA out of memory"
→ Don't worry, LoRA is memory efficient. If it fails, restart runtime and try again.

### "Model download interrupted"
→ Restart runtime and re-run Cell 1 & 2

### "ImportError: No module named..."
→ Cell 1 didn't complete. Re-run it.

---

## 💡 Tips

✅ **Keep Colab open** - Don't close the tab during training  
✅ **GPU should say T4** - Check Runtime type shows GPU  
✅ **First cell takes 2-3 min** - Model download is normal  
✅ **Training shows progress** - Green bars indicate progress  

---

## 🎁 After Training

You'll have `model.zip` with:
```
models/qwen2.5-classifier/
├── adapter_config.json
├── adapter_model.bin
├── config.json
├── special_tokens_map.json
├── tokenizer.model
├── tokenizer_config.json
└── training_config.json
```

Use with:
```bash
python inference.py --model models/qwen2.5-classifier/
```

---

**Ready?** Go to Colab and copy the 3 cells! 🚀
