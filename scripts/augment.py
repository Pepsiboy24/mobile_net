#!/usr/bin/env python3
"""
Data augmentation script to reach target of 500 images per category.
Uses horizontal flips, random rotations (±15°), and brightness/contrast adjustments.
"""

import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
import argparse

def augment_image(image_path, output_dir, augment_count):
    """
    Generate augmented versions of an image.
    
    Args:
        image_path: Path to the original image
        output_dir: Directory to save augmented images
        augment_count: Number of augmented images to generate
    """
    image = Image.open(image_path)
    base_name = Path(image_path).stem
    extension = Path(image_path).suffix
    
    augmented_images = []
    
    for i in range(augment_count):
        # Start with original image
        aug_image = image.copy()
        
        # Apply random combination of augmentations
        augmentation_type = random.choice(['flip', 'rotate', 'brightness', 'contrast', 'combined'])
        
        if augmentation_type == 'flip' or augmentation_type == 'combined':
            # Horizontal flip
            if random.random() > 0.5:
                aug_image = aug_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        if augmentation_type == 'rotate' or augmentation_type == 'combined':
            # Random rotation ±15 degrees
            angle = random.uniform(-15, 15)
            aug_image = aug_image.rotate(angle, expand=True, fillcolor='white')
        
        if augmentation_type == 'brightness' or augmentation_type == 'combined':
            # Brightness adjustment (0.8 to 1.2)
            brightness_factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Brightness(aug_image)
            aug_image = enhancer.enhance(brightness_factor)
        
        if augmentation_type == 'contrast' or augmentation_type == 'combined':
            # Contrast adjustment (0.8 to 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(aug_image)
            aug_image = enhancer.enhance(contrast_factor)
        
        # Save augmented image
        output_name = f"{base_name}_aug_{i+1}{extension}"
        output_path = os.path.join(output_dir, output_name)
        aug_image.save(output_path)
        augmented_images.append(output_path)
    
    return augmented_images

def count_images(directory):
    """Count images in a directory."""
    if not os.path.exists(directory):
        return 0
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    count = 0
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description='Augment images to reach target count')
    parser.add_argument('--target', type=int, default=500, help='Target number of images per category')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory path')
    args = parser.parse_args()
    
    # Categories to augment
    categories = ['samsung', 'oraimo']
    
    for category in categories:
        category_dir = os.path.join(args.data_dir, category)
        
        if not os.path.exists(category_dir):
            print(f"Directory {category_dir} does not exist. Skipping.")
            continue
        
        current_count = count_images(category_dir)
        needed = args.target - current_count
        
        print(f"\n{category.capitalize()}:")
        print(f"  Current images: {current_count}")
        print(f"  Target: {args.target}")
        print(f"  Need to generate: {needed}")
        
        if needed <= 0:
            print(f"  Already at target. Skipping.")
            continue
        
        # Get all original images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        original_images = []
        for file in os.listdir(category_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                original_images.append(os.path.join(category_dir, file))
        
        if not original_images:
            print(f"  No images found in {category_dir}. Skipping.")
            continue
        
        # Calculate how many augmentations per original image
        augment_per_image = needed // len(original_images)
        remainder = needed % len(original_images)
        
        print(f"  Generating {augment_per_image} augmentations per image (+1 for {remainder} images)")
        
        # Generate augmentations
        generated_count = 0
        for i, image_path in enumerate(original_images):
            # Add one extra augmentation for the first 'remainder' images
            current_augment_count = augment_per_image + (1 if i < remainder else 0)
            
            if current_augment_count > 0:
                augment_image(image_path, category_dir, current_augment_count)
                generated_count += current_augment_count
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(original_images)} images...")
        
        final_count = count_images(category_dir)
        print(f"  Generated {generated_count} new images")
        print(f"  Final count: {final_count}")

if __name__ == "__main__":
    main()
