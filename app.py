"""
Streamlit Interactive Demo for Chest X-Ray Classification

Features:
- Image upload
- Real-time prediction
- Model performance dashboard
- Prediction confidence visualization
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Import models
from models import create_vit_small, create_lstm_small
from dataset import get_val_transform


# Page config
st.set_page_config(
    page_title="Chest X-Ray Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_type='vit', checkpoint_path=None):
    """Load trained model."""
    if model_type == 'vit':
        model = create_vit_small(num_classes=2)
    else:
        model = create_lstm_small(num_classes=2)
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        st.success(f"✅ Loaded model from {checkpoint_path}")
    else:
        st.warning("⚠️ No checkpoint loaded - using untrained model")
    
    model.eval()
    return model


@st.cache_data
def load_config():
    """Load configuration."""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {
            'deployment': {
                'demo': {
                    'title': 'Chest X-Ray Abnormality Classifier',
                    'description': 'AI-powered binary classification: Normal vs Abnormal'
                }
            }
        }


def predict_image(model, image, device='cpu'):
    """
    Make prediction on uploaded image.
    
    Args:
        model: Trained model
        image: PIL Image
        device: Device to run on
        
    Returns:
        tuple: (predicted_class, confidence, probabilities)
    """
    # Preprocess image
    transform = get_val_transform(image_size=224)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    return predicted_class, confidence, probabilities.cpu().numpy()


def plot_confidence(probabilities, class_names=['Normal', 'Abnormal']):
    """Create confidence bar chart."""
    fig, ax = plt.subplots(figsize=(8, 3))
    
    colors = ['#2ecc71' if probabilities[0] > 0.5 else '#e74c3c',
              '#e74c3c' if probabilities[1] > 0.5 else '#2ecc71']
    
    bars = ax.barh(class_names, probabilities, color=colors, alpha=0.7)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(prob + 0.02, i, f'{prob*100:.1f}%',
               va='center', fontweight='bold', fontsize=12)
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    config = load_config()
    demo_config = config.get('deployment', {}).get('demo', {})
    
    # Header
    st.title(f"🫁 {demo_config.get('title', 'Chest X-Ray Classifier')}")
    st.markdown(f"*{demo_config.get('description', 'AI-powered medical imaging analysis')}*")
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Model Configuration")
    
    model_type = st.sidebar.selectbox(
        "Select Model Architecture",
        options=['vit', 'lstm'],
        format_func=lambda x: 'Vision Transformer (ViT)' if x == 'vit' else 'LSTM Classifier'
    )
    
    # Default checkpoint path
    default_checkpoint = f"checkpoints/federated_round_5.pth"
    
    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint Path (optional)",
        value=default_checkpoint if Path(default_checkpoint).exists() else ""
    )
    
    # Load model
    try:
        model = load_model(model_type, checkpoint_path if checkpoint_path else None)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.success(f"✅ Running on: {device.upper()}")
    
    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    st.sidebar.info(f"📊 Model Parameters: {num_params:,}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This model classifies chest X-rays into:
    - **Normal**: No abnormalities detected
    - **Abnormal**: Potential findings present
    
    Built with PyTorch and trained using Federated Learning on the NIH ChestX-ray8 dataset.
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-Ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image for classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded X-Ray', use_container_width=True)
            
            # Predict button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    predicted_class, confidence, probabilities = predict_image(
                        model, image, device
                    )
                    
                    # Store results in session state
                    st.session_state['prediction'] = {
                        'class': predicted_class,
                        'confidence': confidence,
                        'probabilities': probabilities
                    }
    
    with col2:
        st.header("📊 Prediction Results")
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            class_names = ['Normal', 'Abnormal']
            predicted_label = class_names[pred['class']]
            
            # Result display
            if pred['class'] == 0:  # Normal
                st.success(f"### ✅ {predicted_label}")
            else:  # Abnormal
                st.warning(f"### ⚠️ {predicted_label}")
            
            st.metric(
                "Confidence",
                f"{pred['confidence']*100:.1f}%",
                help="Model's confidence in the prediction"
            )
            
            # Confidence chart
            fig = plot_confidence(pred['probabilities'], class_names)
            st.pyplot(fig)
            
            # Interpretation
            st.markdown("---")
            st.markdown("### 💡 Interpretation")
            
            if pred['class'] == 0:
                st.markdown("""
                **Normal Classification**
                - No obvious abnormalities detected
                - Routine follow-up recommended
                - Always consult a medical professional
                """)
            else:
                st.markdown("""
                **Abnormal Classification**
                - Potential findings detected
                - Further medical evaluation recommended
                - This model provides screening assistance only
                """)
            
            # Disclaimer
            st.warning("""
            **⚕️ Medical Disclaimer**: This is a research/portfolio project and should NOT be used 
            for clinical diagnosis. Always consult qualified healthcare professionals for medical advice.
            """)
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results")
    
    # Additional information
    st.markdown("---")
    
    with st.expander("📚 Model Information"):
        st.markdown(f"""
        **Architecture**: {model_type.upper()}
        
        **Training Method**: Federated Learning (FedAvg)
        - 3 distributed clients
        - 5 communication rounds
        - Privacy-preserving training
        
        **Dataset**: NIH ChestX-ray8 Subset
        - 10,000 chest X-ray images
        - Binary classification task
        - Stratified train/val/test splits
        
        **Performance Metrics** (expected):
        - Accuracy: >75%
        - F1-Score: >0.70
        - Precision & Recall balanced
        """)
    
    with st.expander("🔬 Technical Details"):
        if model_type == 'vit':
            st.markdown("""
            **Vision Transformer Details**:
            - Patch size: 16×16
            - Number of patches: 196 (from 224×224 image)
            - Embedding dimension: 384
            - Transformer layers: 6
            - Attention heads: 6
            - Parameters: ~22M
            """)
        else:
            st.markdown("""
            **LSTM Classifier Details**:
            - Patch-based sequence processing
            - Bidirectional LSTM: 2 layers
            - Hidden dimension: 128
            - Attention mechanism: Yes
            - Parameters: ~2M
            """)


if __name__ == "__main__":
    main()
