#!/usr/bin/env python3
"""
Generate counterfeit images by applying digital distortions to genuine images.
Takes 100 images from Samsung and 100 from Oraimo, applies logo blur, color shift, and noise.
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from pathlib import Path
import argparse

def apply_logo_blur(image, logo_region=None):
    """
    Apply blur to simulate fake logo printing.
    If logo_region is None, blur the top-right area where logos typically appear.
    """
    width, height = image.size
    
    # Default logo region: top-right corner (typical logo placement)
    if logo_region is None:
        logo_x = int(width * 0.6)  # Start from 60% from left
        logo_y = 0  # Start from top
        logo_w = int(width * 0.4)  # 40% width
        logo_h = int(height * 0.3)  # 30% height
    else:
        logo_x, logo_y, logo_w, logo_h = logo_region
    
    # Ensure region is within image bounds
    logo_x = max(0, min(logo_x, width - 1))
    logo_y = max(0, min(logo_y, height - 1))
    logo_w = min(logo_w, width - logo_x)
    logo_h = min(logo_h, height - logo_y)
    
    # Create a mask for the logo region
    mask = Image.new('L', (width, height), 0)
    mask_draw = Image.new('L', (logo_w, logo_h), 255)
    mask.paste(mask_draw, (logo_x, logo_y))
    
    # Apply blur to the entire image
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Composite: use original image everywhere except blurred logo region
    result = Image.composite(blurred_image, image, mask)
    
    return result

def apply_color_shift(image):
    """
    Apply color shift to simulate fake packaging colors.
    Makes colors slightly washed out or incorrect hue.
    """
    # Convert to HSV using PIL's color matrix manipulation
    img_array = np.array(image)
    
    # Convert RGB to HSV manually
    img_array_float = img_array.astype(np.float32) / 255.0
    r, g, b = img_array_float[:, :, 0], img_array_float[:, :, 1], img_array_float[:, :, 2]
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    diff = max_val - min_val
    
    # Hue calculation
    h = np.zeros_like(max_val)
    mask = max_val == r
    h[mask] = 60 * ((g[mask] - b[mask]) / diff[mask] % 6)
    mask = max_val == g
    h[mask] = 60 * ((b[mask] - r[mask]) / diff[mask] + 2)
    mask = max_val == b
    h[mask] = 60 * ((r[mask] - g[mask]) / diff[mask] + 4)
    
    # Saturation
    s = np.zeros_like(max_val)
    mask = max_val != 0
    s[mask] = diff[mask] / max_val[mask]
    
    # Value
    v = max_val
    
    # Apply color shifts
    # Randomly shift hue (color)
    hue_shift = random.randint(-20, 20)
    h = (h + hue_shift) % 360
    
    # Reduce saturation to make colors washed out
    saturation_factor = random.uniform(0.7, 0.9)
    s = s * saturation_factor
    
    # Slightly adjust brightness
    brightness_factor = random.uniform(0.9, 1.1)
    v = v * brightness_factor
    
    # Convert back to RGB
    c = v * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c
    
    rgb = np.zeros_like(img_array_float)
    
    # Red channel
    mask = (h >= 0) & (h < 60)
    rgb[mask, 0] = c[mask] + m[mask]
    mask = (h >= 60) & (h < 120)
    rgb[mask, 0] = x[mask] + m[mask]
    mask = (h >= 120) & (h < 180)
    rgb[mask, 0] = m[mask]
    mask = (h >= 180) & (h < 240)
    rgb[mask, 0] = m[mask]
    mask = (h >= 240) & (h < 300)
    rgb[mask, 0] = x[mask] + m[mask]
    mask = (h >= 300) & (h < 360)
    rgb[mask, 0] = c[mask] + m[mask]
    
    # Green channel
    mask = (h >= 0) & (h < 60)
    rgb[mask, 1] = x[mask] + m[mask]
    mask = (h >= 60) & (h < 120)
    rgb[mask, 1] = c[mask] + m[mask]
    mask = (h >= 120) & (h < 180)
    rgb[mask, 1] = c[mask] + m[mask]
    mask = (h >= 180) & (h < 240)
    rgb[mask, 1] = x[mask] + m[mask]
    mask = (h >= 240) & (h < 300)
    rgb[mask, 1] = m[mask]
    mask = (h >= 300) & (h < 360)
    rgb[mask, 1] = m[mask]
    
    # Blue channel
    mask = (h >= 0) & (h < 60)
    rgb[mask, 2] = m[mask]
    mask = (h >= 60) & (h < 120)
    rgb[mask, 2] = m[mask]
    mask = (h >= 120) & (h < 180)
    rgb[mask, 2] = x[mask] + m[mask]
    mask = (h >= 180) & (h < 240)
    rgb[mask, 2] = c[mask] + m[mask]
    mask = (h >= 240) & (h < 300)
    rgb[mask, 2] = c[mask] + m[mask]
    mask = (h >= 300) & (h < 360)
    rgb[mask, 2] = x[mask] + m[mask]
    
    # Clip and convert back to uint8
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(rgb)

def apply_noise(image):
    """
    Add digital noise to simulate poor printing quality.
    """
    img_array = np.array(image)
    
    # Generate random noise
    noise = np.random.normal(0, 15, img_array.shape)
    
    # Add noise to image
    noisy_array = img_array + noise
    
    # Clip values to valid range
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_array)

def create_counterfeit(image_path, output_dir, fake_id):
    """
    Create a counterfeit version of an image with multiple distortions.
    """
    image = Image.open(image_path)
    
    # Apply random combination of distortions
    # Always apply at least 2 distortions to make it clearly fake
    distortions = ['logo_blur', 'color_shift', 'noise']
    selected_distortions = random.sample(distortions, random.randint(2, 3))
    
    fake_image = image.copy()
    
    for distortion in selected_distortions:
        if distortion == 'logo_blur':
            fake_image = apply_logo_blur(fake_image)
        elif distortion == 'color_shift':
            fake_image = apply_color_shift(fake_image)
        elif distortion == 'noise':
            fake_image = apply_noise(fake_image)
    
    # Save counterfeit image
    base_name = Path(image_path).stem
    extension = Path(image_path).suffix
    output_name = f"fake_{fake_id:03d}_{base_name}{extension}"
    output_path = os.path.join(output_dir, output_name)
    fake_image.save(output_path)
    
    return output_path

def get_random_images(directory, count):
    """
    Get random sample of images from directory.
    """
    if not os.path.exists(directory):
        return []
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            images.append(os.path.join(directory, file))
    
    # Randomly sample images
    if len(images) <= count:
        return images
    else:
        return random.sample(images, count)

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
    parser = argparse.ArgumentParser(description='Generate counterfeit images')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory path')
    parser.add_argument('--samples_per_category', type=int, default=100, help='Samples per category')
    args = parser.parse_args()
    
    # Ensure counterfeit directory exists
    counterfeit_dir = os.path.join(args.data_dir, 'counterfeit')
    os.makedirs(counterfeit_dir, exist_ok=True)
    
    # Categories to process
    categories = ['samsung', 'oraimo']
    
    print("Generating counterfeit images...")
    print(f"Samples per category: {args.samples_per_category}")
    
    fake_id = 1
    total_generated = 0
    
    for category in categories:
        category_dir = os.path.join(args.data_dir, category)
        
        if not os.path.exists(category_dir):
            print(f"Directory {category_dir} does not exist. Skipping.")
            continue
        
        # Get random sample of images
        sample_images = get_random_images(category_dir, args.samples_per_category)
        
        print(f"\n{category.capitalize()}:")
        print(f"  Selected {len(sample_images)} images for fake generation")
        
        # Generate counterfeit versions
        for i, image_path in enumerate(sample_images):
            try:
                fake_path = create_counterfeit(image_path, counterfeit_dir, fake_id)
                total_generated += 1
                fake_id += 1
                
                if (i + 1) % 20 == 0:
                    print(f"  Generated {i + 1}/{len(sample_images)} fakes...")
                    
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                continue
    
    print(f"\nGenerated {total_generated} counterfeit images")
    
    # Final count verification
    print("\nFinal dataset counts:")
    total_images = 0
    
    for category in ['samsung', 'oraimo', 'counterfeit']:
        category_dir = os.path.join(args.data_dir, category)
        count = count_images(category_dir)
        total_images += count
        print(f"  {category.capitalize()}: {count} images")
    
    print(f"\nTotal images: {total_images}")
    
    if total_images == 1200:
        print("✅ Target achieved: 1,200 total images!")
    else:
        print(f"⚠️  Expected 1,200 images, got {total_images}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    main()
