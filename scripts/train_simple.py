#!/usr/bin/env python3
"""
Simple training script for counterfeit detection using scikit-learn and basic CNN.
Works without TensorFlow/PyTorch for environments with network issues.
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_images_from_directory(data_dir, img_size=(64, 64)):
    """
    Load images from directory structure and return features and labels.
    """
    images = []
    labels = []
    class_names = []
    
    # Get class directories
    for class_dir in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            class_names.append(class_dir)
            
            # Load images from this class
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        # Load and resize image
                        img = Image.open(img_path).convert('RGB').resize(img_size)
                        img_array = np.array(img) / 255.0  # Normalize to [0,1]
                        
                        # Flatten image for scikit-learn
                        img_flat = img_array.flatten()
                        
                        images.append(img_flat)
                        labels.append(class_dir)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
    
    return np.array(images), np.array(labels), class_names

def extract_features(images):
    """
    Extract simple features from images for better classification.
    """
    features = []
    
    for img_flat in images:
        # Reshape back to image shape
        img_size = int(np.sqrt(len(img_flat) // 3))
        img = img_flat.reshape(img_size, img_size, 3)
        
        # Extract color histogram features
        r_hist = np.histogram(img[:,:,0], bins=16, range=(0,1))[0]
        g_hist = np.histogram(img[:,:,1], bins=16, range=(0,1))[0]
        b_hist = np.histogram(img[:,:,2], bins=16, range=(0,1))[0]
        
        # Extract texture features (simple gradients)
        gray = np.mean(img, axis=2)
        grad_x = np.abs(np.diff(gray, axis=0))
        grad_y = np.abs(np.diff(gray, axis=1))
        texture_feat = [np.mean(grad_x), np.mean(grad_y), np.std(grad_x), np.std(grad_y)]
        
        # Combine features
        feature_vector = np.concatenate([r_hist, g_hist, b_hist, texture_feat])
        features.append(feature_vector)
    
    return np.array(features)

def train_simple_classifier(X_train, y_train, X_val, y_val, class_names):
    """
    Train a simple neural network classifier.
    """
    print("🧠 Training classifier...")
    
    # Create MLP classifier
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        learning_rate_init=0.0001,
        max_iter=100,
        random_state=42,
        verbose=True
    )
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Evaluate on validation set
    train_score = clf.score(X_train, y_train)
    val_score = clf.score(X_val, y_val)
    
    print(f"\nTraining accuracy: {train_score:.4f}")
    print(f"Validation accuracy: {val_score:.4f}")
    
    # Detailed evaluation
    y_pred = clf.predict(X_val)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return clf

def save_model(model, label_encoder, class_names, feature_extractor_params):
    """
    Save the trained model and preprocessing components.
    """
    os.makedirs('models', exist_ok=True)
    
    # Save model
    with open('models/trusttag_v1.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save label encoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save class names
    with open('models/class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    
    # Save feature extraction parameters
    with open('models/feature_params.pkl', 'wb') as f:
        pickle.dump(feature_extractor_params, f)
    
    print("💾 Model saved to models/trusttag_v1.pkl")

def create_simple_cnn_features(images, img_size=(64, 64)):
    """
    Create simple CNN-like features using convolution operations.
    """
    features = []
    
    for img_flat in images:
        # Reshape back to image shape
        img = img_flat.reshape(img_size[0], img_size[1], 3)
        
        # Simple edge detection for each channel
        edge_features = []
        for channel in range(3):
            channel_img = img[:, :, channel]
            
            # Horizontal edges
            h_edges = np.abs(np.diff(channel_img, axis=0))
            # Vertical edges  
            v_edges = np.abs(np.diff(channel_img, axis=1))
            
            # Edge statistics
            edge_features.extend([
                np.mean(h_edges), np.std(h_edges),
                np.mean(v_edges), np.std(v_edges)
            ])
        
        # Color statistics
        color_stats = []
        for channel in range(3):
            channel_img = img[:, :, channel]
            color_stats.extend([
                np.mean(channel_img), np.std(channel_img),
                np.min(channel_img), np.max(channel_img)
            ])
        
        # Combine features
        feature_vector = np.concatenate([edge_features, color_stats])
        features.append(feature_vector)
    
    return np.array(features)

def main():
    """
    Main training function.
    """
    print("🚀 Starting TrustTag counterfeit detection training (Simple Version)...")
    
    # Configuration
    DATA_DIR = 'data/processed'
    IMG_SIZE = (64, 64)  # Smaller size for faster processing
    
    # Load data
    print("\n📁 Loading data...")
    images, labels, class_names = load_images_from_directory(DATA_DIR, IMG_SIZE)
    
    print(f"Loaded {len(images)} images from {len(class_names)} classes")
    print(f"Classes: {class_names}")
    print(f"Image shape: {IMG_SIZE} (flattened to {images.shape[1]} features)")
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Extract features
    print("\n🔧 Extracting features...")
    features = extract_features(images)
    print(f"Feature shape: {features.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels_encoded
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train classifier
    model = train_simple_classifier(X_train, y_train, X_val, y_val, class_names)
    
    # Save model
    feature_params = {
        'img_size': IMG_SIZE,
        'feature_type': 'histogram_texture'
    }
    save_model(model, label_encoder, class_names, feature_params)
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved to: models/trusttag_v1.pkl")
    print("Supporting files:")
    print("  - models/label_encoder.pkl")
    print("  - models/class_names.pkl") 
    print("  - models/feature_params.pkl")
    print("  - models/confusion_matrix.png")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    main()
