#!/usr/bin/env python3
"""
Training script for counterfeit detection using MobileNetV3Small.
Loads 1,200 images, splits into 80% train/20% validation, trains for 10-20 epochs.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def load_data(data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Load and split data into training and validation sets.
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Load validation data
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    return train_generator, validation_generator

def create_model(num_classes=3, input_shape=(224, 224, 3)):
    """
    Create MobileNetV3Small model with custom classification head.
    """
    # Load pre-trained MobileNetV3Small
    base_model = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling=None  # We'll add our own pooling
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Create custom head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def train_model(model, train_generator, validation_generator, epochs=15, learning_rate=0.0001):
    """
    Train the model with specified parameters.
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Compile model
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/trusttag_v1.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """
    Plot training history.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_model(model, validation_generator):
    """
    Evaluate the model on validation set.
    """
    # Get predictions
    predictions = model.predict(validation_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())
    
    # Calculate accuracy per class
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))
    
    # Plot confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """
    Main training function.
    """
    print("🚀 Starting TrustTag counterfeit detection training...")
    
    # Configuration
    DATA_DIR = 'data/processed'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.0001
    
    # Load data
    print("\n📁 Loading data...")
    train_generator, validation_generator = load_data(
        DATA_DIR, 
        img_size=IMG_SIZE, 
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")
    
    # Create model
    print("\n🏗️  Creating model...")
    model, base_model = create_model(num_classes=3, input_shape=IMG_SIZE + (3,))
    
    print(f"Model architecture:")
    model.summary()
    
    # Train model
    print(f"\n🎯 Training model for {EPOCHS} epochs...")
    history = train_model(
        model, 
        train_generator, 
        validation_generator,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # Plot training history
    print("\n📊 Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    evaluate_model(model, validation_generator)
    
    # Save final model
    print("\n💾 Saving final model...")
    model.save('models/trusttag_v1_final.h5')
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved to: models/trusttag_v1.h5 (best checkpoint)")
    print(f"Final model saved to: models/trusttag_v1_final.h5")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    main()
