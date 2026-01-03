# Usage Guide - Chest X-Ray Classifier

Complete step-by-step instructions for training and deploying the chest X-ray classifier.

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/ishsta/Chest-X-Ray-Classifier.git
cd Chest-X-Ray-Classifier

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

The data has already been prepared with:
✅ 10,000 images in `data/images/`
✅ Filtered CSV with binary labels
✅ Train/val/test splits (70/15/15)

To verify:
```bash
python verify_pipeline.py
```

## Training

### Option A: Federated Learning (Recommended)

Train using 3 distributed clients with FedAvg:

```bash
# Using Vision Transformer
python main.py --federated --model vit --num_clients 3 --rounds 5

# Using LSTM classifier
python main.py --federated --model lstm --num_clients 3 --rounds 5
```

**Training Time Estimates:**
- CPU: ~2-4 hours (5 rounds)
- GPU (CUDA): ~20-40 minutes (5 rounds)

**Expected Results:**
- Validation accuracy: >75%
- F1-score: >0.70

### Option B: Quick Test Run

For quick testing (1 round, 2 clients):

```bash
python main.py --federated --model vit --num_clients 2 --rounds 1
```

### Training Outputs

After training, you'll find:
```
checkpoints/
├── federated_round_1.pth
├── federated_round_2.pth
├── ...
└── federated_round_5.pth

results/
├── confusion_matrix.png
├── classification_report.txt
├── predictions.png
├── misclassified.png
├── federated_training_history.png
└── test_metrics.txt

models/saved_models/
└── federated_final_model.pth
```

## Evaluation

Evaluation automatically runs after training. To evaluate a specific checkpoint:

```python
# Coming soon: standalone evaluation mode
python main.py --eval_only --checkpoint checkpoints/federated_round_5.pth
```

## Interactive Demo

### Launch Streamlit App

```bash
streamlit run app.py
```

This opens a web interface at `http://localhost:8501` with:
- 📤 Image upload
- 🔍 Real-time prediction
- 📊 Confidence visualization
- 💡 Result interpretation

### Using the Demo

1. **Select Model**: Choose ViT or LSTM from sidebar
2. **Load Checkpoint**: Specify path to trained model (optional)
3. **Upload Image**: Drag & drop or browse for chest X-ray
4. **Analyze**: Click "Analyze Image" to get prediction

**Demo Features:**
- Interactive confidence bars
- Medical disclaimer
- Model architecture details
- Performance metrics

## Configuration

Edit `config.yaml` to customize:

```yaml
training:
  batch_size: 32          # Adjust based on GPU memory
  learning_rate: 0.0001   # Learning rate
  num_epochs: 10          # For centralized training
  
federated:
  num_clients: 3          # Number of FL clients
  num_rounds: 5           # Communication rounds
  local_epochs: 2         # Epochs per client per round
  allocation: balanced    # 'balanced', 'random', or 'iid'

model:
  type: vit              # 'vit' or 'lstm'
  image_size: 224        # Input image size
```

## Command-Line Reference

### Training Arguments

```bash
python main.py [OPTIONS]

Options:
  --federated              Use federated learning
  --model {vit,lstm}      Model architecture
  --num_clients INT       Number of FL clients (default: 3)
  --rounds INT            Communication rounds (default: 5)
  --config PATH           Path to config file (default: config.yaml)
  --eval_only             Evaluation mode only
  --checkpoint PATH       Model checkpoint path
```

### Examples

```bash
# Federated ViT with 4 clients, 10 rounds
python main.py --federated --model vit --num_clients 4 --rounds 10

# Federated LSTM with custom config
python main.py --federated --model lstm --config my_config.yaml

# Quick test (1 round)
python main.py --federated --model vit --rounds 1
```

## Deployment

### For Portfolio

1. **Train the Model**:
   ```bash
   python main.py --federated --model vit --rounds 5
   ```

2. **Test the Demo**:
   ```bash
   streamlit run app.py
   ```

3. **Share**:
   - GitHub repository with README
   - Live demo (Streamlit Cloud, HuggingFace Spaces)
   - Screenshots of results
   - Training metrics visualization

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy `app.py`
5. Upload checkpoint to Streamlit secrets

## Troubleshooting

### Out of Memory (OOM)

**Solution**: Reduce batch size in `config.yaml`
```yaml
training:
  batch_size: 16  # or 8 for limited GPU memory
```

### Slow Training on CPU

**Solutions**:
- Use smaller model: `create_vit_small` or `create_lstm_small`
- Reduce number of workers: `num_workers: 0`
- Use fewer rounds: `--rounds 2`

### Images Not Found

**Check**:
```bash
python verify_pipeline.py
```

Expected output:
```
✅ 10,000 images present
✅ Filtered CSV with proper labels
✅ Split indices saved correctly
```

### Module Import Errors

**Solution**: Ensure all dependencies installed
```bash
pip install -r requirements.txt --upgrade
```

## Performance Optimization

### GPU Acceleration

Automatically uses CUDA if available:
```python
# Check if GPU is being used
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Mixed Precision Training

For faster training on modern GPUs (not yet implemented):
```python
# Future enhancement
python main.py --federated --model vit --amp
```

## Next Steps

After successful training:

1. **Analyze Results**: Check `results/` for metrics and visualizations
2. **Fine-tune**: Adjust hyperparameters in `config.yaml`
3. **Deploy Demo**: Launch Streamlit app for portfolio
4. **Experiment**: Try different model architectures
5. **Extend**: Add more clients, implement other FL algorithms

## Additional Resources

- **NIH Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **FedAvg Paper**: McMahan et al., 2017
- **ViT Paper**: Dosovitskiy et al., 2020
- **Project Repository**: [GitHub](https://github.com/ishsta/Chest-X-Ray-Classifier)

## Support

For issues or questions:
1. Check this guide first
2. Review `README.md`
3. Open GitHub issue with:
   - Error message
   - Steps to reproduce
   - System specs (GPU, RAM, OS)
