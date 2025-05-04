import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
from pathlib import Path

def generate_class_mapping(test_dir, output_path):
    """Generate class mapping file from test directory structure"""
    try:
        # Convert to absolute paths
        test_dir = os.path.abspath(test_dir)
        output_path = os.path.abspath(output_path)
        
        # Verify test directory exists
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
            
        # Get class names from test directory
        class_names = sorted([d for d in os.listdir(test_dir) 
                            if os.path.isdir(os.path.join(test_dir, d))])
        
        if not class_names:
            raise ValueError(f"No class directories found in {test_dir}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create class mapping
        with open(output_path, 'w') as f:
            for idx, class_name in enumerate(class_names):
                f.write(f"{idx},{class_name}\n")
        
        print(f"Created class mapping with {len(class_names)} classes")
        print(f"Saved to: {output_path}")
        return class_names
        
    except PermissionError:
        print(f"Error: Permission denied. Try running as administrator or choose a different output location")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).resolve().parents[2]
    
    # Define paths relative to project root
    test_dir = project_root / "data" / "processed" / "test"
    output_path = project_root / "data" / "processed" / "class_mapping.txt"
    
    print(f"Test directory: {test_dir}")
    print(f"Output path: {output_path}")
    
    # Generate mapping
    classes = generate_class_mapping(str(test_dir), str(output_path))
    
    print("\nClass mapping:")
    for idx, class_name in enumerate(classes):
        print(f"{idx}: {class_name}")