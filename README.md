# Chest X-Ray Classifier with Federated Learning

AI-powered binary classification system for chest X-rays (Normal vs Abnormal) using Vision Transformer and LSTM architectures with Federated Learning capabilities.

## 🎯 Project Overview

This project implements a medical imaging classifier using the NIH ChestX-ray8 dataset, featuring:
- **Dual Model Architectures**: Vision Transformer (ViT) and Bidirectional LSTM with attention
- **Federated Learning**: Privacy-preserving distributed training using FedAvg algorithm
- **Interactive Demo**: Streamlit web app for portfolio showcase

## 📊 Dataset

**NIH ChestX-ray8 Subset**:
- **Source**: [Kaggle NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **Subset**: images_002 (~10,000 images)
- **Labels**: Binary classification - Normal (0) vs Abnormal (1)
- **Format**: 1024×1024 PNG grayscale images

## 🏗️ Architecture

### Vision Transformer (ViT)
- Patch-based image processing (16×16 patches)
- Multi-head self-attention mechanism
- 6-layer transformer encoder
- Positional embeddings for spatial awareness

### LSTM Classifier
- Image-to-sequence conversion via patch extraction
- 2-layer bidirectional LSTM
- Optional attention mechanism
- Dropout regularization for robustness

### Federated Learning
- **Algorithm**: Federated Averaging (FedAvg)
- **Clients**: 3 distributed clients
- **Communication Rounds**: 5 rounds
- **Local Training**: 2 epochs per round

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ishsta/Chest-X-Ray-Classifier.git
cd Chest-X-Ray-Classifier

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

#### Automated (CSV only):
```bash
python download_data.py
```

#### Manual (Images - Required):
1. Visit: [NIH Box Download](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)
2. Download `images_002.tar.gz` (~4GB)
3. Extract to `data/images_002/`

See [DATA_DOWNLOAD.md](DATA_DOWNLOAD.md) for detailed instructions.

### 3. Data Preparation

```bash
# Filter annotations and create train/val/test splits
python data_preparation.py
```

### 4. Training (Coming Soon)

```bash
# Train with ViT model using federated learning
python main.py --model vit --federated --num_clients 3 --rounds 5

# Train with LSTM model
python main.py --model lstm --federated --num_clients 3 --rounds 5
```

### 5. Launch Demo (Coming Soon)

```bash
streamlit run app.py
```

## 📁 Project Structure

```
Chest-X-Ray-Classifier/
├── data/                      # Dataset directory (not tracked)
│   ├── Data_Entry_2017.csv   # Annotations
│   ├── filtered_data.csv     # Filtered subset
│   └── images_002/           # Image files
├── models/                    # Model architectures
│   ├── vit.py               # Vision Transformer
│   ├── lstm_classifier.py   # LSTM models
│   └── __init__.py
├── federated/                 # Federated learning (coming soon)
├── data_preparation.py        # Data loading & splitting
├── dataset.py                # PyTorch Dataset class
├── train.py                  # Training utilities (coming soon)
├── evaluate.py               # Evaluation & visualization (coming soon)
├── main.py                   # Main execution script (coming soon)
├── app.py                    # Streamlit demo (coming soon)
├── config.yaml               # Configuration file
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Model hyperparameters (learning rate, batch size, etc.)
- Federated learning settings (clients, rounds)
- Data augmentation parameters
- Training device (GPU/CPU)

## 🔬 Current Progress

### ✅ Completed
- [x] Project setup and dependencies
- [x] Data download infrastructure
- [x] Data preparation module (filtering, splitting)
- [x] PyTorch Dataset implementation with augmentations
- [x] Vision Transformer architecture
- [x] LSTM classifier with attention mechanism
- [x] Model verification tests

### 🚧 In Progress
- [ ] Training utilities (train_one_epoch, evaluate_model)
- [ ] Federated learning implementation (FedAvg)
- [ ] Evaluation and visualization tools
- [ ] Main execution pipeline
- [ ] Streamlit interactive demo

## 📈 Expected Performance

Based on similar implementations with NIH ChestX-ray8:
- **Target Accuracy**: >75%
- **Target F1-Score**: >0.70
- **Training Time** (ViT, single client): ~45 min on NVIDIA RTX 4090
- **Federated Training**: ~2-3 hours for 5 rounds

## 🎓 Use Cases

- **Medical AI Research**: Federated learning for privacy-preserving medical imaging
- **Computer Vision**: Transformer and LSTM applications in image classification
- **Portfolio Project**: Demonstrates end-to-end ML system design and deployment

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **Dataset**: NIH Clinical Center
- **Reference Paper**: "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases" (Wang et al.)

## 📧 Contact

For questions or collaboration: [Your GitHub Profile](https://github.com/ishsta)

---

**Note**: This is a portfolio/research project. Not intended for clinical use.