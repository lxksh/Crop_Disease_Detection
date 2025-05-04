import numpy as np
import gc
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def evaluate_model(model_path, test_dir, img_size=(224, 224), batch_size=32):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to the saved model file
        test_dir: Directory containing test images organized in class folders
        img_size: Input image size expected by the model
        batch_size: Batch size for evaluation
    """
    
    gc.collect()
    tf.keras.backend.clear_session()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")

    # Verify test directory structure
    class_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    if not class_dirs:
        raise ValueError(f"No class directories found in {test_dir}")
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    # Use smaller batch size for prediction
    pred_batch_size = min(batch_size, 32)
    # Load the model
    model = load_model(model_path)
    
    # Set up data generator for test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    # Get class indices and create a mapping
    class_indices = test_generator.class_indices
    class_names = list(class_indices.keys())
    
    # Print class information for debugging
    print(f"Number of classes in test generator: {len(class_indices)}")
    print(f"Class names: {class_names}")
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Predict on test data
    try:
        y_pred_prob = model.predict(
            test_generator,
            batch_size=pred_batch_size,
            verbose=1
        )
        print(f"Predictions shape: {y_pred_prob.shape}")
        print(f"Number of classes in predictions: {y_pred_prob.shape[1]}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
        
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_generator.classes
    
    # Ensure class_names matches the number of classes in predictions
    if len(class_names) != y_pred_prob.shape[1]:
        print(f"Warning: Number of classes mismatch. Adjusting class names...")
        class_names = class_names[:y_pred_prob.shape[1]]
    
    # Create classification report with verified class names
    try:
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
    except Exception as e:
        print(f"Error creating classification report: {str(e)}")
        print("Falling back to basic classification report without class names")
        report = classification_report(
            y_true, 
            y_pred,
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()

    # ...rest of the existing code...
    
    # Save report to CSV
    report_path = os.path.join(os.path.dirname(model_path), 'classification_report.csv')
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix plot
    cm_path = os.path.join(os.path.dirname(model_path), 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    return test_acc, report_df, cm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="../../models/saved_models/best_model.h5")
    parser.add_argument("--test_dir", default="../../data/processed/test")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    try:
        test_acc, report_df, cm = evaluate_model(
            args.model_path, 
            args.test_dir,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size
        )
        print(f"\nTest Accuracy: {test_acc:.4f}")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")