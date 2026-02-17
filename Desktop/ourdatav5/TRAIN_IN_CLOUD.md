# 🚀 QUICK START: TRAIN IN CLOUD (30 MINUTES)

## ⚡ FASTEST PATH TO TRAINED MODEL

Your CPU would take 8+ hours. Use **FREE GPU on Colab** instead:

---

## 🔴 OPTION 1: GOOGLE COLAB (FREE - RECOMMENDED!)

### ONE-TIME SETUP (2 minutes)

1. **Open Colab:** https://colab.research.google.com

2. **Enable GPU:**
   - Click: `Runtime` → `Change runtime type`
   - Select: `GPU (T4)`
   - Click: `Save`

3. **Cell 1: Setup**
   ```python
   # Clean install
   !pip install -q --no-cache-dir torch transformers datasets peft
   !git clone https://github.com/justasayel/clasmodel.git
   %cd clasmodel
   ```
   **Run:** Ctrl+Enter

4. **Cell 2: Train (OPTIMIZED FOR COLAB)**
   ```python
   !python train_colab.py
   ```
   **Run:** Ctrl+Enter
   
   ⏱️ **Wait ~25-30 minutes**

5. **Cell 3: Download Model**
   ```python
   from google.colab import files
   !zip -r model.zip models/
   files.download('model.zip')
   ```
   **Run:** Ctrl+Enter
   
   ✅ Model downloads to your computer!

### Why `train_colab.py`?
- ✅ Uses LoRA for 75% less memory
- ✅ Optimized batch sizes for T4 GPU
- ✅ Better error recovery
- ✅ Faster training (25-30 min vs 40+ min)

---

## 🟡 OPTION 2: KAGGLE (FREE - ALSO GOOD)

### Step 1: Create Kaggle Account
```
https://www.kaggle.com
```

### Step 2: Create New Notebook
```
New → Notebook
```

### Step 3: Set GPU
```
Settings → Accelerator → GPU (T4)
```

### Step 3: Paste Code
```python
!git clone https://github.com/justasayel/clasmodel.git
%cd clasmodel
!pip install -q torch transformers datasets bitsandbytes
!python train_model.py
```

### Step 4: Run & Download
- Run cells
- Export trained model

---

## 🟢 OPTION 3: LAMBDA LABS ($0.50/hr GPU)

### Step 1: Sign Up
```
https://lambdalabs.com
```

### Step 2: Launch GPU Instance
- Select: GPU Cloud
- Choose: Tesla V100 (~$1.08/hr) or RTX 4090 (~$0.99/hr)
- Select Ubuntu 22.04
- SSH into instance

### Step 3: Train
```bash
git clone https://github.com/justasayel/clasmodel.git
cd clasmodel
bash setup.sh
python train_model.py

# Save results
scp -r models/ user@your-machine:/path/
```

---

## 🔵 OPTION 4: AWS SAGEMAKER (Pay as you go)

### Step 1: AWS Console
```
https://console.aws.amazon.com
```

### Step 2: SageMaker → Notebooks
- Create new notebook instance
- Instance type: `ml.p3.2xlarge` (GPU)

### Step 3: Open Terminal
```bash
git clone https://github.com/justasayel/clasmodel.git
cd clasmodel
bash setup.sh
python train_model.py
```

### Step 4: Download Model
Use S3 or SageMaker Studio File Manager

---

## 📱 OPTION 5: REPLIT (FREE TIER)

### Step 1: Create Account
```
https://replit.com
```

### Step 2: Import from GitHub
- Click: `+ Create` → `Import from GitHub`
- Paste: `https://github.com/justasayel/clasmodel.git`

### Step 3: Shell
```bash
bash setup.sh
python train_model.py
```

**Note**: Free tier is CPU only, but still faster than local development

---

## 🎯 RECOMMENDED: GOOGLE COLAB

**Why?**
- ✅ Completely FREE
- ✅ 12GB VRAM T4 GPU (good enough)
- ✅ Fast training (~30 min)
- ✅ No account needed
- ✅ Easy to use
- ✅ Can save to Google Drive

**Total Cost**: $0  
**Training Time**: ~30 minutes  
**Difficulty**: Easy

---

## 📊 COMPARISON

| Service | Cost | Speed | Effort | GPU |
|---------|------|-------|--------|-----|
| **Colab** | FREE | 30m | ⭐ | T4 |
| **Kaggle** | FREE | 30m | ⭐ | T4 |
| **Lambda** | $1.08/hr | 20m | ⭐⭐ | V100 |
| **Replit** | FREE | 2hr | ⭐ | None |
| **AWS** | $1-5/hr | 25m | ⭐⭐⭐ | P3 |
| **Local (Mac)** | FREE | 8-12hr | ⭐⭐ | None |

---

## 🔧 STEP-BY-STEP: GOOGLE COLAB

### 1. Open Link
```
https://colab.research.google.com
```

### 2. Create Notebook
- File → New notebook

### 3. Cell 1: Setup
```python
# Install dependencies
!pip install -q torch transformers datasets bitsandbytes peft

# Clone repo
!git clone https://github.com/justasayel/clasmodel.git
%cd clasmodel
```

**Run**: Ctrl+Enter

### 4. Cell 2: Enable GPU
```python
import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

**If GPU not showing:**
- Go to: `Runtime` → `Change runtime type`
- Select: `GPU` 
- Click: `Save`

### 5. Cell 3: Train
```python
!python train_model.py
```

**Run**: Ctrl+Enter

*Wait ~30 minutes for training to complete*

### 6. Cell 4: Download
```python
from google.colab import files

# Zip model
!zip -r model.zip models/

# Download
files.download('model.zip')
```

**The trained model will download to your computer!**

---

## 💾 WHAT HAPPENS AFTER TRAINING

1. Model saved to: `models/qwen2.5-classifier/`
2. Download from cloud service
3. Extract on your machine
4. Use locally with: `python inference.py --model models/qwen2.5-classifier/`

---

## ⚡ QUICK COMMANDS (Copy-Paste Ready)

### Google Colab Setup
```python
!pip install -q torch transformers datasets bitsandbytes peft
!git clone https://github.com/justasayel/clasmodel.git
%cd clasmodel
!python train_model.py
```

### Lambda Labs SSH
```bash
git clone https://github.com/justasayel/clasmodel.git
cd clasmodel && bash setup.sh && python train_model.py
```

### Download trained model from Colab
```python
!zip -r model.zip models/
from google.colab import files
files.download('model.zip')
```

---

## 🎯 YOUR OPTIMAL PATH

1. **Right Now (5 min)**: Go to Google Colab
2. **Next (30 min)**: Run training
3. **Then (2 min)**: Download model
4. **Finally**: Use on your machine

**Total Time**: 40 minutes  
**Cost**: FREE  
**Result**: Fully trained classifier ready to deploy

---

## 📞 TROUBLESHOOTING

### "GPU not available in Colab"
```
Runtime → Change runtime type → Select GPU → Save
```

### "Out of memory"
```
Reduce BATCH_SIZE in train_model.py (change 2 to 1)
```

### "Model download too slow"
```
Save to Google Drive instead:
  from google.colab import drive
  drive.mount('/content/drive')
  !cp -r models/ /content/drive/My\ Drive/
```

---

## 🚀 READY?

**Fastest path to trained model:**

1. Open: https://colab.research.google.com
2. Paste the setup code from above
3. Enable GPU
4. Run training
5. Download model

**Time: 40 minutes | Cost: $0**

✅ **START NOW IN COLAB!**

---

## 📖 REFERENCE: YOUR GITHUB

Clone link: `https://github.com/justasayel/clasmodel.git`

All scripts and data are there!

---

**Questions?** See `START_HERE.md` in your repository.
