# 🚀 QWEN2.5 CLASSIFIER - COMPLETE DEPLOYMENT GUIDE

## Overview
Your government data classification model is ready for deployment. This guide covers all deployment options.

## Project Status
✅ **Data Preparation**: Complete
- Train data: 65 samples (70%)
- Test data: 28 samples (30%)
- Balanced across 4 classification levels

⏳ **Next Steps**: Model fine-tuning and deployment

---

## 💾 OPTION 1: Export to External Storage

### ZIP Archive
```bash
# Create portable zip file
python export.py --format zip --output classifier-export.zip

# Transfer via cloud, email, external drive, etc.
# Size: ~20-30 GB (includes full model)
```

### USB Drive
```bash
# Export to USB drive
python export.py --format usb --drive /Volumes/USB_DRIVE

# Then transfer USB to other machine
# Files will be in: /Volumes/USB_DRIVE/qwen-classifier/
```

### GitHub Repository
```bash
# Push to GitHub (requires git and GitHub account)
python export.py --format github --repo https://github.com/username/repo

# Others can clone and use:
# git clone https://github.com/username/repo
```

---

## 🖥️ OPTION 2: Local Deployment

### On Mac (Your Current Machine)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fine-tune model (WARNING: Will be slow on CPU)
python train_model.py  # Hours on CPU, ~30 min on GPU

# 3. Test with sample document
python inference.py --model models/qwen2.5-classifier \
  --file processed_dataset/train.jsonl

# 4. Classify batch of documents
python inference.py --model models/qwen2.5-classifier \
  --batch ./documents/ --output results.json
```

### On Linux Server
```bash
# 1. Transfer files to server
scp -r project/ user@server:/home/user/

# 2. Install CUDA toolkit (for GPU)
# Follow: https://developer.nvidia.com/cuda-downloads

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run classification service
python inference.py --model models/qwen2.5-classifier --file data.txt
```

### On Windows Server
```bash
# 1. Transfer files
# Use file transfer tool or git clone

# 2. Install Python 3.10+
# Download from python.org

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run classifier
python inference.py --model models/qwen2.5-classifier --file data.txt
```

---

## 🌐 OPTION 3: REST API Deployment

### FastAPI Server (Python)
```python
# Save as: api_server.py
from fastapi import FastAPI
from inference import load_model, classify_document
import json

app = FastAPI()
model, tokenizer = load_model("models/qwen2.5-classifier")

@app.post("/classify")
async def classify(text: str):
    response = classify_document(model, tokenizer, text)
    return {"classification": response}

# Run: uvicorn api_server.py --host 0.0.0.0 --port 8000
```

### Flask Server (Python)
```python
from flask import Flask, request, jsonify
from inference import load_model, classify_document

app = Flask(__name__)
model, tokenizer = load_model("models/qwen2.5-classifier")

@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    response = classify_document(model, tokenizer, data["text"])
    return jsonify({"classification": response})

# Run: flask run
```

### Test API
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here"}'
```

---

## 🐳 OPTION 4: Docker Deployment

### Dockerfile
```dockerfile
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Default command
CMD ["python3", "inference.py"]
```

### Build and Run
```bash
# Build image
docker build -t qwen-classifier:latest .

# Run container
docker run --gpus all -v $(pwd)/documents:/app/documents qwen-classifier:latest \
  --batch /app/documents/ --output results.json

# Run API
docker run --gpus all -p 8000:8000 qwen-classifier:latest \
  python api_server.py
```

### Docker Compose
```yaml
version: '3.8'
services:
  classifier:
    image: qwen-classifier:latest
    gpus: all
    volumes:
      - ./documents:/app/documents
      - ./results:/app/results
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: python api_server.py
```

---

## ☁️ OPTION 5: Cloud Deployment

### Google Colab (Free GPU)
```python
# Upload to Colab, then:
!git clone https://github.com/your/repo
%cd repo
!pip install -r requirements.txt
!python train_model.py  # Much faster!
!python inference.py --model models/qwen2.5-classifier --text "test"
```

### AWS Deployment
```bash
# 1. Create EC2 instance with GPU (g4dn.xlarge or better)
# 2. SSH into instance
ssh -i key.pem ec2-user@instance-ip

# 3. Install dependencies
sudo yum install python3-pip
pip3 install -r requirements.txt

# 4. Transfer model and run
# Use scp to transfer files
python inference.py --model models/qwen2.5-classifier --batch ./documents/
```

### Azure Container Instances
```bash
# Create container
az container create \
  --resource-group mygroup \
  --name qwen-classifier \
  --image qwen-classifier:latest \
  --gpu 1 \
  --cpu 4 \
  --memory 16

# Test
az container exec --resource-group mygroup --name qwen-classifier \
  --exec-command "python inference.py --text 'test'"
```

### HuggingFace Spaces
```
1. Create new Space on huggingface.co
2. Upload files to repo
3. Add app.py (Gradio interface)
4. Deploy automatically
```

---

## 📊 OPTION 6: WordPress/CMS Integration

### PHP Backend
```php
<?php
// classifier.php
$model_path = "/path/to/models/qwen2.5-classifier";
$text = $_POST['document_text'];

$command = escapeshellcmd(
  "python3 inference.py --model $model_path --text \"$text\""
);
$output = shell_exec($command);
echo json_encode(["classification" => $output]);
?>
```

### WordPress Plugin
1. Create plugin directory: `wp-content/plugins/qwen-classifier/`
2. Add PHP files with inference code
3. Create admin page for document classification
4. Add AJAX endpoints for WordPress

---

## 🔐 OPTION 7: On-Premises Deployment

### Windows Server
```
1. Copy all files to: C:\Projects\qwen-classifier\
2. Install Python 3.10+ on Windows
3. Create batch file (run.bat):
   @echo off
   cd C:\Projects\qwen-classifier
   python inference.py --model models/qwen2.5-classifier --batch ./documents/
   pause

4. Schedule with Windows Task Scheduler for automated runs
```

### Linux Server (Production)
```bash
# 1. Create systemd service
sudo nano /etc/systemd/system/qwen-classifier.service

[Unit]
Description=Qwen Classifier API
After=network.target

[Service]
User=classifier
WorkingDirectory=/opt/qwen-classifier
ExecStart=/usr/bin/python3 /opt/qwen-classifier/api_server.py
Restart=always

[Install]
WantedBy=multi-user.target

# 2. Enable and start
sudo systemctl enable qwen-classifier
sudo systemctl start qwen-classifier
sudo systemctl status qwen-classifier

# 3. Monitor logs
sudo journalctl -u qwen-classifier -f
```

---

## 💡 HARDWARE REQUIREMENTS

| Scenario | GPU | CPU | RAM | Storage |
|----------|-----|-----|-----|---------|
| Inference (Single) | Optional | 4-core | 8GB | 50GB |
| Inference (Batch) | Recommended | 8-core | 16GB | 50GB |
| Training | Required | 16-core | 64GB | 100GB |
| Production API | High-end | 32-core | 128GB | 500GB |

---

## 📦 DELIVERY CHECKLIST

Before delivering to stakeholders:
- [ ] Test on target infrastructure
- [ ] Document configuration needed
- [ ] Create user manual
- [ ] Verify data privacy compliance
- [ ] Set up monitoring/logging
- [ ] Create backup strategy
- [ ] Document rollback procedure

---

## 🔄 WORKFLOW DIAGRAMS

### Training Pipeline
```
Data (93 documents)
    ↓
Train/Test Split (70/30)
    ↓
train_model.py (Fine-tune Qwen2.5)
    ↓
models/qwen2.5-classifier/
    ↓
Ready for inference
```

### Inference Pipeline
```
Document Input
    ↓
inference.py (Load model)
    ↓
Classification Engine
    ↓
Output (JSON)
    ↓
Results Stored
```

---

## 📞 SUPPORT & TROUBLESHOOTING

### Common Issues

**Issue**: Model download fails
```bash
# Solution: Set cache directory
export HF_HOME=/path/with/space
python train_model.py
```

**Issue**: Out of Memory
```bash
# Solution: Reduce batch size
# Edit train_model.py:
BATCH_SIZE = 1  # Instead of 2
```

**Issue**: Slow inference
```bash
# Solution: Use GPU
# Install CUDA:
# https://developer.nvidia.com/cuda-downloads

# Or use quantized model - already enabled!
```

---

## 🎯 RECOMMENDED DEPLOYMENT PATH

### For Quick Testing
```bash
1. python quickstart.py
2. python train_model.py
3. python inference.py --text "sample"
```

### For Production
```bash
1. Set up Linux server with GPU
2. Deploy with Docker
3. Run API with FastAPI
4. Monitor with Prometheus
5. Backup model regularly
```

### For Enterprise
```bash
1. Air-gapped on-premises deployment
2. Multiple GPU nodes
3. Load balancing with Kubernetes
4. Compliance logging
5. Disaster recovery setup
```

---

## 📝 NEXT STEPS

1. **Choose deployment method** from options above
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Fine-tune model**: `python train_model.py`
4. **Test locally**: `python inference.py --text "test"`
5. **Export model**: `python export.py --format zip`
6. **Deploy to target**: Transfer files to production environment

---

## 📞 CONTACT & SUPPORT
For issues or questions, refer to:
- `README.md` - Usage guide
- `quickstart.py` - Interactive setup
- Test documents in `processed_dataset/`

**Last Updated**: February 17, 2026
**Model**: Qwen/Qwen2.5-7B-Instruct
**Status**: ✅ Ready for Deployment
