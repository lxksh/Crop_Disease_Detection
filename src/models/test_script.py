import os
from pathlib import Path
from predict_model import PlantDiseasePredictor

# Get absolute paths
project_root = Path(__file__).resolve().parents[2]
model_path = str(project_root / "models" / "saved_models" / "best_model.h5")
class_mapping_path = str(project_root / "data" / "processed" / "class_mapping.txt")

# Initialize predictor
predictor = PlantDiseasePredictor(
    model_path=model_path,
    class_mapping_path=class_mapping_path,
    img_size=(96, 96)
)

# Find available test images
test_dir = project_root / "data" / "processed" / "test"
test_images = []

print("Looking for test images...")
for root, _, files in os.walk(test_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            test_images.append(img_path)
            print(f"Found image: {img_path}")

if not test_images:
    raise FileNotFoundError(f"No images found in {test_dir}")

# Test predictions on found images
for img_path in test_images[:2]:  # Test first 2 images
    if os.path.exists(img_path):
        try:
            result = predictor.predict(img_path)
            print(f"\nPredictions for {result['filename']}:")
            print("Top predictions:")
            for pred in result['top_predictions']:
                print(f"- {pred['class']}: {pred['probability']:.2%}")
            
            # Get treatment
            treatment = predictor.get_treatment_recommendation(result["predicted_class"])
            print("\nRecommended treatment:")
            for key, value in treatment.items():
                print(f"{key}: {value}")
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    else:
        print(f"Image not found: {img_path}")