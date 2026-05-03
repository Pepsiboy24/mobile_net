#!/usr/bin/env python3
"""
Image preprocessing script for MobileNetV3 training data.
Validates, filters, standardizes, and renames images.
"""

import os
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TARGET_SIZE = (224, 224)
MIN_FILE_SIZE = 10 * 1024  # 10KB in bytes
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}

def is_valid_image(file_path):
    """Check if file is a valid image and meets size requirements."""
    try:
        # Check file size
        if os.path.getsize(file_path) < MIN_FILE_SIZE:
            logger.warning(f"File too small: {file_path} ({os.path.getsize(file_path)} bytes)")
            return False
        
        # Try to open with PIL
        with Image.open(file_path) as img:
            # Verify image can be loaded
            img.verify()
        
        # Reopen to check if image can be processed
        with Image.open(file_path) as img:
            # Convert to RGB to ensure it's processable
            rgb_img = img.convert('RGB')
        
        return True
    except (UnidentifiedImageError, IOError, OSError, Exception) as e:
        logger.warning(f"Invalid image {file_path}: {str(e)}")
        return False

def process_image(input_path, output_path, category_prefix):
    """Process a single image: resize, convert to RGB, and save."""
    try:
        with Image.open(input_path) as img:
            # Convert to RGB
            rgb_img = img.convert('RGB')
            
            # Resize to target size using LANCZOS for high quality
            resized_img = rgb_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Save as JPEG with good quality
            resized_img.save(output_path, 'JPEG', quality=95, optimize=True)
            
        return True
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        return False

def preprocess_category(category_name, raw_dir, processed_dir):
    """Preprocess all images in a category folder."""
    logger.info(f"Processing category: {category_name}")
    
    raw_category_dir = raw_dir / category_name
    processed_category_dir = processed_dir / category_name
    
    # Create processed directory if it doesn't exist
    processed_category_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_category_dir.exists():
        logger.error(f"Raw directory not found: {raw_category_dir}")
        return 0
    
    # Get all files in raw directory
    all_files = list(raw_category_dir.iterdir())
    logger.info(f"Found {len(all_files)} files in {category_name}")
    
    valid_images = []
    deleted_files = []
    
    # Validate and filter images
    for file_path in all_files:
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            
            # Skip unsupported formats
            if file_ext not in SUPPORTED_FORMATS:
                logger.info(f"Skipping unsupported format: {file_path}")
                deleted_files.append(file_path.name)
                continue
            
            # Validate image
            if is_valid_image(file_path):
                valid_images.append(file_path)
            else:
                deleted_files.append(file_path.name)
    
    logger.info(f"Valid images: {len(valid_images)}, Deleted/Invalid: {len(deleted_files)}")
    
    # Process and rename images
    processed_count = 0
    category_prefix = category_name.lower()
    
    for i, img_path in enumerate(valid_images, 1):
        # Generate new filename
        new_filename = f"{category_prefix}_{i:03d}.jpg"
        output_path = processed_category_dir / new_filename
        
        # Process image
        if process_image(img_path, output_path, category_prefix):
            processed_count += 1
            logger.debug(f"Processed: {img_path.name} -> {new_filename}")
        else:
            logger.error(f"Failed to process: {img_path.name}")
    
    logger.info(f"Successfully processed {processed_count} images for {category_name}")
    return processed_count

def main():
    """Main preprocessing function."""
    # Setup paths
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    
    logger.info("Starting image preprocessing...")
    logger.info(f"Raw directory: {raw_dir}")
    logger.info(f"Processed directory: {processed_dir}")
    
    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Get categories (subdirectories in raw folder)
    categories = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    logger.info(f"Found categories: {categories}")
    
    # Process each category
    results = {}
    total_processed = 0
    
    for category in categories:
        count = preprocess_category(category, raw_dir, processed_dir)
        results[category] = count
        total_processed += count
    
    # Print final results
    logger.info("\n" + "="*50)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*50)
    for category, count in results.items():
        logger.info(f"{category.capitalize()}: {count} clean images")
    logger.info(f"Total: {total_processed} clean images")
    logger.info("="*50)
    
    return results

if __name__ == "__main__":
    results = main()
