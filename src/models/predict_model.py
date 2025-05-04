import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import cv2
import json

class PlantDiseasePredictor:
    def __init__(self, model_path, class_mapping_path, img_size=(224, 224)):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model
            class_mapping_path: Path to class mapping file
            img_size: Input image size (default: 96x96 to match model)
        """
        self.img_size = img_size
        
        if not tf.__version__.startswith("2.19"):
            raise ImportError(f"TensorFlow 2.19.x required, but found {tf.__version__}")
        
        # Load the class mapping first
        self.class_names = self._load_class_mapping(class_mapping_path)
        
        # Load model with custom objects and error handling
        try:
            # Try loading with custom object scope
            with tf.keras.utils.custom_object_scope({'GlorotUniform': tf.keras.initializers.glorot_uniform()}):
                self.model = tf.keras.models.load_model(model_path, compile=False)
                
            # Recompile the model with basic settings
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Verify model loading
            if not isinstance(self.model, tf.keras.Model):
                raise ValueError("Failed to load model properly")
                
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}. Please ensure TensorFlow 2.19.0 is installed.")
        
    def _load_class_mapping(self, mapping_path):
        """Load class mapping from file."""
        class_dict = {}
        with open(mapping_path, 'r') as f:
            for line in f:
                idx, name = line.strip().split(',', 1)
                class_dict[int(idx)] = name
        
        # Create an ordered list
        class_names = [class_dict[i] for i in range(len(class_dict))]
        return class_names
    
    def preprocess_image(self, img_path):
        """
        Preprocess an image for prediction.
        
        Args:
            img_path: Path to the image file
        
        Returns:
            Preprocessed image ready for model input
        """
        # Convert path to absolute path
        img_path = os.path.abspath(img_path)
        
        # Verify image exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Read image with error handling
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}. Verify the file is a valid image.")
        
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img / 255.0  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            return img
        except Exception as e:
            raise ValueError(f"Error processing image {img_path}: {str(e)}")
    
    def predict(self, img_path, top_k=3):
        """
        Make a prediction for an image.
        
        Args:
            img_path: Path to the image file
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess the image
        img = self.preprocess_image(img_path)
        
        # Make prediction
        preds = self.model.predict(img)[0]
        
        # Get top k predictions
        top_indices = np.argsort(preds)[-top_k:][::-1]
        top_preds = [(self.class_names[i], float(preds[i])) for i in top_indices]
        
        # Organize results
        result = {
            "filename": os.path.basename(img_path),
            "top_predictions": [{"class": c, "probability": p} for c, p in top_preds],
            "predicted_class": self.class_names[np.argmax(preds)],
            "confidence": float(np.max(preds))
        }
        
        return result
    
    def get_treatment_recommendation(self, disease_class):
        """
        Get treatment recommendations for a disease class.
        
        Args:
            disease_class: The predicted disease class
        
        Returns:
            Dictionary with treatment recommendations
        """
        # Simple treatment recommendation database (expand this)
        recommendations = {
            "Apple___Apple_scab": {
                "chemical": "Apply fungicides like captan or myclobutanil",
                "organic": "Remove infected leaves and apply neem oil spray",
                "preventive": "Ensure good air circulation, avoid overhead watering"
            },
            "Tomato___Early_blight": {
                "chemical": "Apply chlorothalonil or copper-based fungicides",
                "organic": "Apply copper spray or sulfur dust",
                "preventive": "Mulch around plants, ensure adequate spacing"
            },
            # Add more diseases and recommendations
        }
        
        # Return recommendations if available
        if disease_class in recommendations:
            return recommendations[disease_class]
        else:
            return {
                "note": "Specific recommendations not available for this disease",
                "general": "Consult a local agricultural extension for treatment advice"
            }

if __name__ == "__main__":
    from pathlib import Path
    
    # Get project root directory
    project_root = Path(__file__).resolve().parents[2]
    
    # Setup paths using Path for better cross-platform compatibility
    model_path = str(project_root / "models" / "saved_models" / "best_model.h5")
    class_mapping_path = str(project_root / "data" / "processed" / "class_mapping.txt")
    
    # Verify paths exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"Class mapping not found at: {class_mapping_path}")
    
    try:
        # Initialize predictor with correct image size
        predictor = PlantDiseasePredictor(
            model_path=model_path, 
            class_mapping_path=class_mapping_path,
            img_size=(96, 96)  # Match model's expected input size
        )
        
        # Find a test image automatically
        test_dir = project_root / "data" / "processed" / "test" / "Tomato___Early_blight"
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found at: {test_dir}")
        
        # Get first image from test directory
        test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.JPG"))
        if not test_images:
            raise FileNotFoundError(f"No JPG images found in {test_dir}")
        
        test_img = str(test_images[0])
        print(f"Using test image: {test_img}")
        
        # Make prediction
        result = predictor.predict(test_img)
        
        # Get treatment recommendations
        treatment = predictor.get_treatment_recommendation(result["predicted_class"])
        
        # Print results
        print("\nPrediction Results:")
        print(json.dumps(result, indent=2))
        print("\nTreatment Recommendations:")
        print(json.dumps(treatment, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")