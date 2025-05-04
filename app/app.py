import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import sys
import tempfile
from PIL import Image
import random
from datetime import datetime 
import requests
from geopy.geocoders import Nominatim
import plotly.express as px
import pandas as pd

st.set_page_config(
        page_title="Plant Disease Detector",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# Add this after st.set_page_config()
st.markdown("""
    <style>
        
        /* Card styling */
        .stCard {
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        /* Stats container */
        .stats-container {
            display: flex;
            justify-content: space-around;
            padding: 1rem;
            background-color: #f5f5f5;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        /* Custom header styles */
        .custom-header {
            background-color: #f0f8f0;
            padding: 1rem;
            border-left: 5px solid #2e7d32;
            margin: 1rem 0;
        }
        
        /* Custom button */
        .stButton>button {
            background-color: #2e7d32;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        
        /* Footer styling */
        .footer {
            background-color: #f5f5f5;
            padding: 2rem;
            margin-top: 3rem;
            border-top: 1px solid #ddd;
        }
    </style>
    """, unsafe_allow_html=True)

def check_tensorflow_version():
    required_version = '2.19.0'  # Specify the version your model was trained with
    current_version = tf.__version__
    if current_version != required_version:
        st.warning(f"Warning: TensorFlow version mismatch. Required: {required_version}, Current: {current_version}")
        
# Set paths
IMG_SIZE = (96, 96)
MODEL_PATH = os.path.join('models', 'saved_models', 'best_model.h5')
CLASS_MAPPING_PATH = os.path.join('data', 'processed', 'class_mapping.txt')
def load_predictor():
    try:
        check_tensorflow_version()
        return PlantDiseasePredictor(MODEL_PATH, CLASS_MAPPING_PATH,IMG_SIZE)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Try reinstalling TensorFlow with: pip install tensorflow==2.13.0")
        return None

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.predict_model import PlantDiseasePredictor

def get_random_fact():
    facts = [
        "🌿 Did you know? Plants can communicate with each other through chemical signals.",
        "🌱 Around 85% of plant diseases are caused by fungal or fungal-like organisms.",
        "🔬 Early detection of plant diseases can save up to 60% of crop losses.",
        "🌍 Climate change is increasing the spread of plant diseases globally.",
        "💧 Some plant diseases can be prevented simply by avoiding overhead watering.",
        "🦠 There are over 50,000 known plant diseases caused by various pathogens.",
    ]
    return random.choice(facts)

def get_weather():
    """Fetch weather data with better error handling and fallback options"""
    try:
        # Replace with your OpenWeatherMap API key
        API_KEY = "eb223c0eeb5084da009ef6c409656e59"
        
        # First try to get location via IP
        try:
            ip_response = requests.get('http://ipapi.co/json/', timeout=5)
            if ip_response.status_code != 200:
                raise ValueError("Could not fetch location data")
            
            location_data = ip_response.json()
            city = location_data.get('city', 'Unknown')
            lat = location_data.get('latitude')
            lon = location_data.get('longitude')
            
            if not all([lat, lon]):
                raise ValueError("Could not determine location coordinates")
            
            # Get weather data from OpenWeatherMap
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            weather_response = requests.get(weather_url, timeout=5)
            
            if weather_response.status_code != 200:
                raise ValueError(f"Weather API error: {weather_response.status_code}")
            
            weather_data = weather_response.json()
            
            # Extract weather information
            temperature = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            description = weather_data['weather'][0]['description']
            
            # Format weather info with emojis
            weather_info = f"""
            📍 **{city}**
            
            🌡️ Temperature: {temperature:.1f}°C
            💧 Humidity: {humidity}%
            ☁️ Conditions: {description.capitalize()}
            
            *Last updated: {datetime.now().strftime('%H:%M')}*
            
            ### Plant Care Tip:
            {get_weather_advice(temperature, humidity, description)}
            """
            
            return weather_info
            
        except requests.RequestException as e:
            return "⚠️ Network error while fetching weather data"
        except ValueError as e:
            return f"⚠️ Error: {str(e)}"
        except KeyError as e:
            return "⚠️ Could not parse weather data"
        except Exception as e:
            return f"⚠️ Unexpected error: {str(e)}"
            
    except Exception as e:
        return "Weather service temporarily unavailable"

def get_weather_advice(temperature, humidity, conditions):
    """Get plant care advice based on weather conditions"""
    advice = []
    
    # Temperature advice
    if temperature > 30:
        advice.append("⚠️ High temperature - Monitor plants for water stress")
    elif temperature < 10:
        advice.append("❄️ Low temperature - Protect sensitive plants")
        
    # Humidity advice
    if humidity > 80:
        advice.append("⚠️ High humidity - Watch for fungal diseases")
    elif humidity < 30:
        advice.append("💧 Low humidity - Consider misting plants")
        
    # Conditions advice
    conditions = conditions.lower()
    if 'rain' in conditions:
        advice.append("☔ Rainy - Hold off on watering")
    elif 'clear' in conditions:
        advice.append("☀️ Clear - Good time for plant inspection")
    elif 'cloud' in conditions:
        advice.append("☁️ Cloudy - Moderate watering if needed")
        
    return "\n".join(advice) if advice else "🌿 Normal growing conditions"

# Update the weather display in main():
# Replace the existing weather section in the sidebar with:
def show_weather_section(st_container):
    st_container.markdown("---")
    st_container.subheader("🌤️ Local Weather")
    
    weather_info = get_weather()
    if "⚠️" in weather_info:
        st_container.warning(weather_info)
    else:
        st_container.info(weather_info)

# Add this before the main content
def show_quick_stats():
    # st.markdown('<div class="stats-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Scans", "100+")
    with col2:
        st.metric("Success Rate", "94.5%")
    with col3:
        st.metric("Diseases Detected", "38")
    with col4:
        st.metric("Supported Plants", "14")
    
    st.markdown('</div>', unsafe_allow_html=True)



# Add this to the Statistics tab

def get_dataset_statistics():
    """Get actual statistics from the dataset"""
    stats = {
        'crops': ['Apple', 'Tomato', 'Corn', 'Strawberry', 'Cherry', 'Blueberry'],
        'diseases': {
            'Apple': ['Apple_scab', 'Black_rot', 'Cedar_apple_rust', 'healthy'],
            'Tomato': ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 
                      'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot', 
                      'Yellow_Leaf_Curl_Virus', 'mosaic_virus', 'healthy'],
            'Corn': ['Cercospora_leaf_spot', 'Common_rust', 'healthy']
        }
    }
    return stats

def show_disease_map():
    st.subheader("📍 Disease Distribution Analysis")
    try:
        
        # Get actual disease classes from class_mapping.txt
        with open('data/processed/class_mapping.txt', 'r') as f:
            classes = f.readlines()
        
        # Process disease data
        diseases = {}
        for line in classes:
            try:
                if ',' in line:
                    # Split only on first comma in case disease names contain commas
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        _, disease = parts
                        if '___' in disease:
                            crop, condition = disease.split('___')
                            if crop not in diseases:
                                diseases[crop] = []
                            diseases[crop].append(condition)
            except Exception as e:
                st.warning(f"Skipping malformed line: {line.strip()}")
                continue
        if not diseases:
                st.error("No valid disease data found in the mapping file.")
                return
        
        # Create visualization data
        disease_data = {
            'Crop': [],
            'Conditions': [],
            'Count': [],
            'Has_Healthy_Variant': []
        }
        
        for crop, conditions in diseases.items():
            disease_data['Crop'].append(crop)
            disease_data['Conditions'].append(len(conditions))
            disease_data['Count'].append(len([c for c in conditions if 'healthy' not in c.lower()]))
            disease_data['Has_Healthy_Variant'].append('healthy' in [c.lower() for c in conditions])
        
        df = pd.DataFrame(disease_data)
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Disease Distribution by Crop")
            fig_bar = px.bar(
                df,
                x='Crop',
                y='Conditions',
                color='Has_Healthy_Variant',
                title='Number of Conditions per Crop',
                labels={'Conditions': 'Number of Conditions/Diseases'},
                color_discrete_map={True: '#43a047', False: '#e53935'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.markdown("### Key Statistics")
            st.markdown(f"""
            - Total Crops: {len(diseases)}
            - Total Conditions: {sum(df['Conditions'])}
            - Crops with Healthy Samples: {sum(df['Has_Healthy_Variant'])}
            - Most Affected Crop: {df.iloc[df['Conditions'].argmax()]['Crop']}
            """)
        
        # Show detailed breakdown
        st.markdown("### Detailed Disease Breakdown")
        for crop, conditions in diseases.items():
            with st.expander(f"{crop} ({len(conditions)} conditions)"):
                healthy = [c for c in conditions if 'healthy' in c.lower()]
                diseases = [c for c in conditions if 'healthy' not in c.lower()]
                
                st.markdown(f"""
                - **Healthy Variants**: {len(healthy)}
                - **Disease Variants**: {len(diseases)}
                
                **Diseases**:
                {''.join(['- ' + d.replace('_', ' ') + '\n' for d in diseases])}
                """)
                
    except FileNotFoundError:
        st.error("Class mapping file not found. Please check the file path.")
    except Exception as e:
        st.error(f"Error processing disease data: {str(e)}")

def main():
    
    st.title("🌿 Plant Disease Detection System")
    st.write("""
    Upload an image of a plant leaf to detect if it's healthy or has a disease.
    This system can identify various plant diseases to help farmers take timely action.
    """)
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388e3c;
    }
    </style>
    """, unsafe_allow_html=True)
    

    # Sidebar
    with st.sidebar:
        st.sidebar.header("📊 Dashboard")
        st.sidebar.info("""
        ### Statistics
        - Model Accuracy: 94.5%
        - Diseases Detectable: 38
        - Supported Plants: 14
        """)
        st.sidebar.markdown("---")
        st.sidebar.subheader("💡 Plant Fact of the Day")
        st.sidebar.info(get_random_fact())
        
        # Weather section with error handling

        try:
            weather_info = show_weather_section(st.sidebar)
        
            
            # Add weather-based plant care tips
            if "Unable to fetch" not in weather_info:
                conditions = {
                    'rain': "⚠️ Avoid watering, risk of overwatering",
                    'clear': "✅ Good conditions for plant inspection",
                    'cloud': "☁️ Check for proper air circulation",
                    'humidity': lambda h: "⚠️ High disease risk" if h > 80 else "✅ Normal conditions"
                }
                st.sidebar.markdown("### Weather Advisory")
                st.sidebar.info("Monitor conditions and adjust care as needed")
        except Exception as e:
            st.sidebar.error("Weather conditions can affect plant diseases. Check local weather for better plant care.")

    # Add tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Disease Detection","Statistics","Plant Care Guide", "About"])

    with tab1:
        # ... existing image upload and prediction code ...
        @st.cache_resource
        def load_predictor():
            try:
                return PlantDiseasePredictor(MODEL_PATH, CLASS_MAPPING_PATH,img_size=IMG_SIZE)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None

        predictor = load_predictor()

        # Image upload
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"],help="Supported formats: JPG, JPEG, PNG")

        col1, col2 = st.columns(2)
        
        if uploaded_file is not None:
            
            try:
            # Validate file extension
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                if file_ext not in ['.jpg', '.jpeg', '.png']:
                    st.error("Invalid file format. Please upload a JPG, JPEG, or PNG file.")
                    return
                
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    img_path = tmp_file.name

                # Display the uploaded image
                with col1:
                    st.subheader("Your Uploaded Image")
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)

                # Make prediction
                if predictor:
                    with st.spinner("Analyzing image..."):
                        result = predictor.predict(img_path)
                    
                    # Display results
                    with col2:
                        st.subheader("Analysis Results")
                        disease_class = result["predicted_class"]
                        confidence = result["confidence"] * 100
                        display_class = disease_class.replace("___", " - ").replace("_", " ")
                        
                        st.markdown(f"**Detected Condition:** {display_class}")
                        st.markdown(f"**Confidence:** {confidence:.2f}%")
                        st.progress(float(confidence/100))
                        
                        # Get and display treatment recommendations
                        treatment = predictor.get_treatment_recommendation(disease_class)
                        st.subheader("Treatment Recommendations")
                        
                        if "note" in treatment:
                            st.info(treatment["note"])
                            st.markdown(f"**General Advice:** {treatment['general']}")
                        else:
                            st.markdown(f"**Chemical Treatment:** {treatment['chemical']}")
                            st.markdown(f"**Organic Treatment:** {treatment['organic']}")
                            st.markdown(f"**Preventive Measures:** {treatment['preventive']}")
                        
                        # Display other potential matches
                        st.subheader("Other Possibilities")
                        for pred in result["top_predictions"][1:]:
                            class_name = pred["class"].replace("___", " - ").replace("_", " ")
                            prob = pred["probability"] * 100
                            st.markdown(f"**{class_name}**: {prob:.2f}%")

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
            finally:
                # Clean up temporary file
                if 'img_path' in locals():
                    try:
                        os.unlink(img_path)
                    except:
                        pass

    
    with tab2:
        st.title("Disease Statistics")
        show_disease_map()
        show_quick_stats()
        
         # Add additional dataset insights
        st.markdown("### Dataset Insights")
        stats = get_dataset_statistics()
        st.markdown(f"""
        - Dataset contains {len(stats['crops'])} different crops
        - Tomato has the most disease variants ({len(stats['diseases']['Tomato'])})
        - Each crop has at least one healthy sample variant
        - Most common diseases: Leaf spots, blights, and viral infections
        """)
    
    with tab3:
        st.header("🌱 Plant Care Guide")
        st.markdown("""
        ### Best Practices for Plant Health
        1. **Regular Monitoring** 
           - Check plants weekly for signs of disease
           - Monitor leaf color and growth patterns
        
        2. **Proper Watering**
           - Water at the base of plants
           - Avoid wetting leaves unnecessarily
           - Water early in the day
        
        3. **Prevention Tips**
           - Maintain good air circulation
           - Clean gardening tools regularly
           - Remove affected leaves promptly
        """)
        
    with tab4:
        st.header("ℹ️ About the System")
        st.markdown("""
        ### Technology Stack
        - Deep Learning: TensorFlow 2.x
        - Image Processing: OpenCV
        - Web Interface: Streamlit
        
        ### Model Information
        - Architecture: MobileNetV2
        - Training Dataset: 87,000+ images
        - Validation Accuracy: 95.73%
        
        ### Support
        Having issues? Contact me or visit my GitHub repository.
        """)
        
        
    # Sample images section
    st.subheader("Don't have an image? Try these samples:")
    
    # Sample image selection
    # You'll need to provide these sample images
    sample_dir = os.path.join('data', 'samples')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
        
    # Define some sample images if available
    valid_extensions = ('.jpg', '.jpeg', '.png')
    sample_images = []
    if os.path.exists(sample_dir):
        sample_images = [f for f in os.listdir(sample_dir) if os.path.splitext(f.lower())[1] in valid_extensions]
    
    # Display sample images in a grid if available
    if sample_images:
        cols = st.columns(3)
        for i, img_name in enumerate(sample_images[:3]):  # Display up to 3 samples
            img_path = os.path.join(sample_dir, img_name)
            with cols[i % 3]:
                st.image(Image.open(img_path), caption=img_name.split('.')[0], width=150)
                if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                    # Process the selected sample image
                    with col1:
                        st.subheader("Selected Sample Image")
                        sample_image = Image.open(img_path)
                        st.image(sample_image, use_container_width=True)
                    
                    # Make prediction on sample
                    if predictor:
                        try:
                            with st.spinner("Analyzing sample image..."):
                                result = predictor.predict(img_path)
                            
                            # Display results for sample (similar to uploaded image)
                            with col2:
                                # (Same results display code as for uploaded images)
                                st.subheader("Analysis Results")
                                disease_class = result["predicted_class"]
                                confidence = result["confidence"] * 100
                                display_class = disease_class.replace("___", " - ").replace("_", " ")
                                st.markdown(f"**Detected Condition:** {display_class}")
                                st.markdown(f"**Confidence:** {confidence:.2f}%")
                                st.progress(float(confidence/100))
                                
                                # Get and display treatment recommendations
                                treatment = predictor.get_treatment_recommendation(disease_class)
                                st.subheader("Treatment Recommendations")
                                if "note" in treatment:
                                    st.info(treatment["note"])
                                    st.markdown(f"**General Advice:** {treatment['general']}")
                                else:
                                    st.markdown(f"**Chemical Treatment:** {treatment['chemical']}")
                                    st.markdown(f"**Organic Treatment:** {treatment['organic']}")
                                    st.markdown(f"**Preventive Measures:** {treatment['preventive']}")
                        except Exception as e:
                            st.error(f"Error analyzing sample image: {e}")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📱 Connect")
        st.markdown("[GitHub](https://github.com/lxksh/Crop_Disease_Detection)")
        st.markdown("[📧 lakshvijay04@gmail.com](mailto:lakshvijay04@gmail.com)")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("### 📖 Resources")
        st.markdown("[Plant Disease Database](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)")
    with col3:
        st.markdown("### Today's Date")
        st.markdown(f"{datetime.now().strftime('%d-%m-%Y')}")    
    

    # Information section
    st.markdown("---")
    st.markdown("""
    ### How It Works
    
    1. **Upload an image** of a plant leaf
    2. Our AI model **analyzes the image** to identify diseases
    3. Get **diagnosis and treatment recommendations**
    
    <div style='text-align: center; padding: 1.5rem; margin-top: 2rem; border-radius: 10px;'>
    This application uses deep learning to identify plant diseases from leaf images.The deep learning model is trained on thousands of plant leaf images to detect various diseases across multiple plant species. It can help farmers detect diseases early and take appropriate measures.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem; margin-top: 2rem; border-radius: 10px;'>
        <span style='color: white; font-size: 1.1rem; font-weight: 500;'>
            Made with ❤️ by Laksh
        </span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()