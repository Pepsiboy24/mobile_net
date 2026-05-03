import os
import random
import pickle
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_model_and_encoder():
    """Load the trained model and label encoder"""
    # Load the model
    with open('models/trusttag_v1.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load the label encoder
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load feature parameters
    with open('models/feature_params.pkl', 'rb') as f:
        feature_params = pickle.load(f)
    
    return model, label_encoder, feature_params

def preprocess_image(image_path, feature_params):
    """Preprocess image for prediction using the same feature extraction as training"""
    # Load and resize image to the size used in training
    img_size = feature_params['img_size']
    img = Image.open(image_path).convert('RGB').resize(img_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Flatten for feature extraction
    img_flat = img_array.flatten()
    
    # Extract features using the same method as training
    if feature_params['feature_type'] == 'histogram_texture':
        features = extract_histogram_texture_features([img_flat], img_size)[0]
    else:
        # Fallback to simple flattening if feature type is unknown
        features = img_flat
    
    return features, img

def extract_histogram_texture_features(images, img_size):
    """
    Extract the same features used during training (color histograms + texture)
    """
    features = []
    
    for img_flat in images:
        # Reshape back to image shape
        img = img_flat.reshape(img_size[0], img_size[1], 3)
        
        # Extract color histogram features (16 bins per channel)
        r_hist = np.histogram(img[:,:,0], bins=16, range=(0,1))[0]
        g_hist = np.histogram(img[:,:,1], bins=16, range=(0,1))[0]
        b_hist = np.histogram(img[:,:,2], bins=16, range=(0,1))[0]
        
        # Extract texture features (simple gradients)
        gray = np.mean(img, axis=2)
        grad_x = np.abs(np.diff(gray, axis=0))
        grad_y = np.abs(np.diff(gray, axis=1))
        texture_feat = [np.mean(grad_x), np.mean(grad_y), np.std(grad_x), np.std(grad_y)]
        
        # Combine features (48 histogram features + 4 texture features = 52 total)
        feature_vector = np.concatenate([r_hist, g_hist, b_hist, texture_feat])
        features.append(feature_vector)
    
    return np.array(features)

def get_random_images():
    """Get one random image from each category"""
    # Get paths to processed folders
    samsung_folder = 'data/processed/samsung'
    counterfeit_folder = 'data/processed/counterfeit'
    
    # Get all image files from each folder
    samsung_images = [f for f in os.listdir(samsung_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    counterfeit_images = [f for f in os.listdir(counterfeit_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Select random images
    samsung_image = random.choice(samsung_images)
    counterfeit_image = random.choice(counterfeit_images)
    
    return (os.path.join(samsung_folder, samsung_image), 
            os.path.join(counterfeit_folder, counterfeit_image))

def predict_and_display(model, label_encoder, feature_params, image_path, true_label):
    """Make prediction and display results"""
    # Preprocess image with feature extraction
    features, img = preprocess_image(image_path, feature_params)
    
    # Make prediction
    prediction = model.predict([features])[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba([features])[0]
        confidence = max(probabilities)
    else:
        confidence = 1.0
    
    # Display results
    print(f"\n{'='*50}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Correct: {'✓' if predicted_label == true_label else '✗'}")
    print(f"{'='*50}")
    
    # Display image
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPredicted: {predicted_label} ({confidence:.2%})")
    plt.axis('off')
    plt.show()
    
    return predicted_label, confidence

def main():
    """Main test function"""
    print("🔍 TrustTag System - Model Testing")
    print("="*50)
    
    # Load model and encoder
    try:
        model, label_encoder, feature_params = load_model_and_encoder()
        print("✅ Model and label encoder loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Classes: {list(label_encoder.classes_)}")
        print(f"Feature parameters: {feature_params}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get random test images
    try:
        samsung_path, counterfeit_path = get_random_images()
        print(f"\n✅ Test images selected:")
        print(f"Samsung: {os.path.basename(samsung_path)}")
        print(f"Counterfeit: {os.path.basename(counterfeit_path)}")
    except Exception as e:
        print(f"❌ Error selecting images: {e}")
        return
    
    # Test Samsung image
    print(f"\n🧪 Testing Samsung image...")
    samsung_pred, samsung_conf = predict_and_display(
        model, label_encoder, feature_params, samsung_path, "samsung"
    )
    
    # Test Counterfeit image
    print(f"\n🧪 Testing Counterfeit image...")
    counterfeit_pred, counterfeit_conf = predict_and_display(
        model, label_encoder, feature_params, counterfeit_path, "counterfeit"
    )
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"Samsung: {'✓ Correct' if samsung_pred == 'samsung' else '✗ Wrong'} ({samsung_conf:.2%})")
    print(f"Counterfeit: {'✓ Correct' if counterfeit_pred == 'counterfeit' else '✗ Wrong'} ({counterfeit_conf:.2%})")
    
    accuracy = (1 if samsung_pred == 'samsung' else 0) + (1 if counterfeit_pred == 'counterfeit' else 0)
    print(f"Overall Accuracy: {accuracy}/2 ({accuracy/2:.1%})")
    
    if accuracy == 2:
        print("\n🎉 Perfect! The TrustTag system is working correctly!")
    elif accuracy == 1:
        print("\n⚠️  Partial success. Model needs improvement.")
    else:
        print("\n❌ Model failed on both tests. Retraining needed.")

if __name__ == "__main__":
    main()
