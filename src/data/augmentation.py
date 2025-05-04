import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

def verify_directories(data_dir, output_dir):
    """
    Verify that directories exist and have proper structure.
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Input directory not found: {data_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify there are class directories
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not classes:
        raise ValueError(f"No class directories found in {data_dir}")
    
    return classes

def setup_augmentation():
    """
    Create an ImageDataGenerator with augmentation parameters.
    """
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def apply_augmentation(data_dir, output_dir, samples_per_class=500, img_size=(224, 224)):
    """
    Apply data augmentation to training images.
    
    Args:
        data_dir: Directory containing class folders with images
        output_dir: Output directory for augmented images
        samples_per_class: Target number of samples per class after augmentation
        img_size: Target image size (height, width)
    """
    # Verify directories and get classes
    classes = verify_directories(data_dir, output_dir)
    print(f"Found {len(classes)} classes in {data_dir}")
    
    # Setup augmentation generator
    datagen = setup_augmentation()
    
    for class_name in tqdm(classes, desc="Augmenting classes"):
        try:
            class_dir = os.path.join(data_dir, class_name)
            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Count existing images
            existing_images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            num_existing = len(existing_images)
            
            print(f"\nProcessing {class_name}: {num_existing} existing images")
            
            # Copy original images to output directory
            for img_file in existing_images:
                src_path = os.path.join(class_dir, img_file)
                dst_path = os.path.join(output_class_dir, img_file)
                if not os.path.exists(dst_path):
                    img = cv2.imread(src_path)
                    if img is not None:
                        img = cv2.resize(img, img_size)  # Resize original images too
                        cv2.imwrite(dst_path, img)
            
            # If we need more samples, generate augmented images
            if num_existing < samples_per_class:
                num_to_generate = samples_per_class - num_existing
                print(f"Generating {num_to_generate} augmented images for {class_name}")
                
                # Load a batch of images for augmentation
                image_batch = []
                for img_file in existing_images:
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        image_batch.append(img)
                
                if image_batch:
                    image_batch = np.array(image_batch)
                    
                    # Generate augmented images
                    aug_iter = datagen.flow(
                        image_batch, 
                        batch_size=len(image_batch),
                        save_to_dir=output_class_dir,
                        save_prefix='aug',
                        save_format='jpg'
                    )
                    
                    # Generate the required number of augmented images
                    batch_count = 0
                    while batch_count < num_to_generate // len(image_batch) + 1:
                        aug_iter.next()
                        batch_count += 1
                        
                        # Check if we've generated enough images
                        current_count = len([f for f in os.listdir(output_class_dir) 
                                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        if current_count >= samples_per_class:
                            break
                            
        except Exception as e:
            print(f"Error processing class {class_name}: {str(e)}")
            continue

def main():
    """Main function to run the augmentation process"""
    # Update paths to match your directory structure
    train_data_dir = "../../data/processed/train"
    augmented_dir = "../../data/processed/train_augmented"
    
    print(f"Starting data augmentation...")
    print(f"Source directory: {os.path.abspath(train_data_dir)}")
    print(f"Output directory: {os.path.abspath(augmented_dir)}")
    
    try:
        apply_augmentation(train_data_dir, augmented_dir)
        print("\nData augmentation completed successfully!")
        
        # Print statistics
        classes = os.listdir(augmented_dir)
        for class_name in classes:
            class_dir = os.path.join(augmented_dir, class_name)
            if os.path.isdir(class_dir):
                num_images = len([f for f in os.listdir(class_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"{class_name}: {num_images} images")
                
    except Exception as e:
        print(f"Error during augmentation: {str(e)}")

if __name__ == "__main__":
    main()