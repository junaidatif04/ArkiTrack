from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Gemini API
API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=API_KEY)

# Initialize variables for models
global_mobilenet = None
global_inception = None
global_vgg = None
stage_specific_models = {}
MODELS_LOADED = False

def load_models():
    """Load all required models."""
    global global_mobilenet, global_inception, global_vgg, stage_specific_models, MODELS_LOADED
    try:
        # Load global ensemble models
        global_mobilenet = tf.keras.models.load_model("models/mobilenet.keras", compile=False)
        global_inception = tf.keras.models.load_model("models/inception.keras", compile=False)
        global_vgg = tf.keras.models.load_model("models/vgg16.keras", compile=False)

        # Load stage-specific models
        stage_specific_models = {
            "foundation": tf.keras.models.load_model("models/Foundation_mobile.keras", compile=False),
            "superstructure": tf.keras.models.load_model("models/Superstructure_vgg16.keras", compile=False),
            "facade": tf.keras.models.load_model("models/Facade_inception.keras", compile=False),
            "Interior": tf.keras.models.load_model("models/Interior_mobile.keras", compile=False),
            "finishing works": tf.keras.models.load_model("models/finishing_mobile.keras", compile=False)
        }
        
        MODELS_LOADED = True
        print("All models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        MODELS_LOADED = False

# Define stage mappings
stages = {
    "facade": [
        "Exterior_Cladding_and_Finishes",
        "Window_and_Door_Installation",
        "exterior_wall_construction",
    ],
    "finishing works": [
        "Painting", 
        "fixture installation", 
        "Millwork and carpentry"
    ],
    "foundation": [
        "Excavation", 
        "Reinforcement Placement",
        "concrete curing", 
        "concrete_pouring"
    ],
    "Interior": [
        "Ceiling Installation",
        "Flooring Installation",
        "Staircase Finishing",
    ],

    "superstructure": [
        "Roof_Decking",
        "Stair Case",
        "Structural_Frame_Erection_(framing)",
        "Structural_Wall_Construction",
    ]
}

def ensemble_predict(image, is_facade=False):
    """Predict the stage using the ensemble of models with weighted contributions."""
    if not MODELS_LOADED:
        raise ValueError("Models not loaded")

    # Preprocess images for each model with correct dimensions
    # InceptionV3 requires 299x299
    inception_preprocessed = tf.image.resize(image, (299, 299))
    inception_preprocessed = tf.cast(inception_preprocessed, tf.float32) / 255.0
    inception_preprocessed = tf.expand_dims(inception_preprocessed, axis=0)

    # MobileNet and VGG use 224x224
    mobilenet_preprocessed = tf.image.resize(image, (224, 224))
    mobilenet_preprocessed = tf.cast(mobilenet_preprocessed, tf.float32) / 255.0
    mobilenet_preprocessed = tf.expand_dims(mobilenet_preprocessed, axis=0)

    vgg_preprocessed = tf.image.resize(image, (224, 224))
    vgg_preprocessed = tf.cast(vgg_preprocessed, tf.float32) / 255.0
    vgg_preprocessed = tf.expand_dims(vgg_preprocessed, axis=0)

    # Get predictions from each model
    mobilenet_output = global_mobilenet.predict(mobilenet_preprocessed, verbose=0)[0]
    inception_output = global_inception.predict(inception_preprocessed, verbose=0)[0]
    vgg_output = global_vgg.predict(vgg_preprocessed, verbose=0)[0]

    # Combine predictions using weighted average
    ensemble_output = (
        0.3 * mobilenet_output +  # 30% weight to MobileNet
        0.4 * inception_output +  # 40% weight to Inception
        0.3 * vgg_output          # 30% weight to VGG
    )
    
    # Get the predicted stage and confidence
    predicted_stage_index = np.argmax(ensemble_output)
    confidence_score = float(ensemble_output[predicted_stage_index] * 100)

    stage_list = list(stages.keys())
    predicted_stage = stage_list[predicted_stage_index]

    return predicted_stage, confidence_score

def classify_stage(image, selected_stage):
    """Perform stage-specific classification."""
    if not MODELS_LOADED or selected_stage not in stage_specific_models:
        raise ValueError("Models not loaded or invalid stage")

    # Preprocess image for stage-specific model
    if selected_stage == "facade":
        preprocessed = tf.image.resize(image, (299, 299))
    else:
        preprocessed = tf.image.resize(image, (224, 224))

    preprocessed = tf.cast(preprocessed, tf.float32) / 255.0
    preprocessed = tf.expand_dims(preprocessed, axis=0)

    # Get prediction from stage-specific model
    model = stage_specific_models[selected_stage]
    predictions = model.predict(preprocessed, verbose=0)[0]
    
    # Get the predicted sub-stage and confidence
    predicted_index = np.argmax(predictions)
    confidence = float(predictions[predicted_index] * 100)
    
    sub_stages = stages[selected_stage]
    predicted_sub_stage = sub_stages[predicted_index]

    return predicted_sub_stage, confidence

def describe_image_with_gemini(image_path, max_retries=3, retry_delay=1):
    """Generate image description using Gemini API with retry logic."""
    if not API_KEY:
        return "Image description unavailable: API key not configured"

    import time
    
    for attempt in range(max_retries):
        try:
            # Convert image to bytes
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Configure Gemini
            genai.configure(api_key=API_KEY)
            
            # Create model instance with the new recommended model
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Generate content with more conservative settings
            response = model.generate_content(
                contents=[
                    "Describe this construction site image breifly, focusing on the visible stage of construction",
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ],
                generation_config={
                    "temperature": 0.3,  # More conservative temperature
                    "top_p": 0.7,        # More focused sampling
                    "top_k": 20,         # More conservative top_k
                    "max_output_tokens": 200  # Limit response length
                }
            )
            
            # Check if response has text
            if hasattr(response, 'text'):
                return response.text
            else:
                return "Image description unavailable: No response from API"
                
        except Exception as e:
            error_message = str(e).lower()
            if 'rate limit' in error_message:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                return "Image description temporarily unavailable: API rate limit reached. Please try again later."
            else:
                return f"Error generating description: {str(e)}"
    
    return "Image description unavailable after multiple attempts"

# Try to load models
load_models()

# Define image size
IMAGE_SIZE = (224, 224)

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Render home page."""
    return render_template('home.html')

@app.route('/validate', methods=['POST'])
def validate_image():
    """Handle image validation."""
    try:
        if not MODELS_LOADED:
            return jsonify({
                'error': 'Models are not loaded properly.',
                'status': 'error'
            }), 500

        if 'image' not in request.files:
            return jsonify({
                'error': 'No image uploaded.',
                'status': 'error',
                'validation': 'Unable to process: No image uploaded',
                'description': '',
                'description_status': 'error'
            }), 400

        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type.',
                'status': 'error',
                'validation': 'Unable to process: Invalid file type',
                'description': '',
                'description_status': 'error'
            }), 400

        # Save the uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Load and preprocess the image
            image = Image.open(filepath).convert('RGB')
            image_array = np.array(image)

            # Parse form data
            selected_stage = request.form.get('stage')
            proceed = request.form.get('proceed', 'false')
            describe = request.form.get('describe', 'false')

            # Convert string values to boolean
            proceed = proceed.lower() in ['true', 'on', '1', 'yes']
            describe = describe.lower() in ['true', 'on', '1', 'yes']

            # Resize the image based on the selected stage
            is_facade = selected_stage == "facade"
            predicted_stage, global_confidence = ensemble_predict(image_array, is_facade)

            if predicted_stage != selected_stage and not proceed:
                return jsonify({
                    'validation': f"Mismatch detected! Global classifier predicts stage '{predicted_stage}' "
                              f"with confidence {global_confidence:.2f}%, but selected stage is '{selected_stage}'.",
                    'description': '',
                    'description_status': 'not_requested',
                    'status': 'mismatch'
                })

            # Stage-specific classification
            predicted_class, stage_confidence = classify_stage(image_array, selected_stage)
            result = {
                'validation': f"The image matches the selected stage '{selected_stage}'. Predicted: '{predicted_class}' "
                          f"with confidence {stage_confidence:.2f}%.",
                'status': 'success',
                'description': '',
                'description_status': 'not_requested'
            }

            if describe:
                description = describe_image_with_gemini(filepath)
                result.update({
                    'description': description,
                    'description_status': 'success' if 'error' not in description.lower() else 'error'
                })
            
            return jsonify(result)

        except Exception as e:
            return jsonify({
                'error': f'Error processing image: {str(e)}',
                'status': 'error',
                'validation': 'Unable to process image',
                'description': '',
                'description_status': 'error'
            }), 500

    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error',
            'validation': 'Server error occurred',
            'description': '',
            'description_status': 'error'
        }), 500


def test_model_loading():
    load_models()
    if global_mobilenet is None or global_inception is None or global_vgg is None:
        print("One or more models failed to load.")
    else:
        print("All models loaded successfully.")

# Call this function during app startup
test_model_loading()

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
