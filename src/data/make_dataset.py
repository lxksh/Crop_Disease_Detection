import os
from tqdm import tqdm
import random
import shutil

def verify_raw_data(raw_dir):
    """
    Verify the raw data directory structure and content.
    Args:
        raw_dir: Path to the raw data directory containing color images
    Returns:
        bool: True if verification passes, False otherwise
    """
    if not os.path.exists(raw_dir):
        print(f"Error: Raw data directory not found at {raw_dir}")
        return False
        
    classes = os.listdir(raw_dir)
    print(f"Found {len(classes)} classes")
    
    total_images = 0
    for cls in classes:
        class_dir = os.path.join(raw_dir, cls)
        if os.path.isdir(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            print(f"{cls}: {len(images)} images")
            total_images += len(images)
    
    print(f"\nTotal images found: {total_images}")
    return total_images > 0

def create_dataset(raw_data_dir, processed_data_dir, img_size=(224, 224), test_split=0.15, val_split=0.15):
    """Process raw images and split into train/validation/test sets."""
    # Verify paths
    raw_data_dir = os.path.join(raw_data_dir, "plant_village_dataset", "color")
    if not os.path.exists(raw_data_dir):
        raise ValueError(f"Raw data directory not found: {raw_data_dir}")

    # Create output directories
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(processed_data_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    # Get all classes
    classes = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]
    
    for class_name in tqdm(classes, desc="Processing classes"):
        # Create class directories in each split
        class_dir = os.path.join(raw_data_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"Warning: No images found in {class_name}")
            continue
            
        # Split images into train/val/test
        random.shuffle(images)
        test_size = int(len(images) * test_split)
        val_size = int(len(images) * val_split)
        
        test_imgs = images[:test_size]
        val_imgs = images[test_size:test_size + val_size]
        train_imgs = images[test_size + val_size:]
        
        # Copy images to respective directories
        for split, img_list in [('train', train_imgs), ('validation', val_imgs), ('test', test_imgs)]:
            dst_dir = os.path.join(processed_data_dir, split, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            
            for img in img_list:
                src = os.path.join(class_dir, img)
                dst = os.path.join(dst_dir, img)
                shutil.copy2(src, dst)

if __name__ == "__main__":
    raw_data_dir = "../../data/raw"
    processed_data_dir = "../../data/processed"
    
    try:
        create_dataset(raw_data_dir, processed_data_dir)
    except Exception as e:
        print(f"Error: {str(e)}")