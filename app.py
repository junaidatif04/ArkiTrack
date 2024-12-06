from flask import Flask, render_template, request, jsonify, send_file
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from firebase_admin import credentials, firestore
import firebase_admin
import time
import pytz
from datetime import datetime
from werkzeug.utils import secure_filename
import tempfile
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=API_KEY)

FIREBASE_CRED_PATH ="config/fnatic-2cba7-firebase-adminsdk-ccao3-c65eb4de07.json"
cred = credentials.Certificate(FIREBASE_CRED_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

global_mobilenet = None
global_inception = None
global_vgg = None
stage_specific_models = {}
MODELS_LOADED = False

def load_models():
    global global_mobilenet, global_inception, global_vgg, stage_specific_models, MODELS_LOADED
    try:
        global_mobilenet = tf.keras.models.load_model("models/mobilenet.keras", compile=False)
        global_inception = tf.keras.models.load_model("models/inception.keras", compile=False)
        global_vgg = tf.keras.models.load_model("models/vgg16.keras", compile=False)

        stage_specific_models = {
            "foundation": tf.keras.models.load_model("models/Foundation_mobile.keras", compile=False),
            "superstructure": tf.keras.models.load_model("models/Superstructure_mobile.keras", compile=False),
            "facade": tf.keras.models.load_model("models/Facade_inception.keras", compile=False),
            "Interior": tf.keras.models.load_model("models/Interior_mobile.keras", compile=False),
            "finishing works": tf.keras.models.load_model("models/finishing_mobile.keras", compile=False)
        }

        MODELS_LOADED = True
        print("All models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        MODELS_LOADED = False

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
        "Staircase Finishing"
    ],
    "superstructure": [
        "Roof_Decking",
        "Stair Case",
        "Structural_Frame_Erection_(framing)",
        "Structural_Wall_Construction",
    ],
}

progress_indices = {
    ("foundation", "Excavation"): 0,
    ("foundation", "Reinforcement Placement"): 1,
    ("foundation", "concrete_pouring"): 2,
    ("foundation", "concrete curing"): 3,

    ("superstructure", "Structural_Frame_Erection_(framing)"): 4,
    ("superstructure", "Stair Case"): 5,
    ("superstructure", "Structural_Wall_Construction"): 6,
    ("superstructure", "Roof_Decking"): 7,

    ("facade", "exterior_wall_construction"): 8,
    ("facade", "Window_and_Door_Installation"): 9,
    ("facade", "Exterior_Cladding_and_Finishes"): 10,

    ("Interior", "Ceiling Installation"): 11,
    ("Interior", "Flooring Installation"): 12,
    ("Interior", "Staircase Finishing"): 13,

    ("finishing works", "Painting"): 14,
    ("finishing works", "fixture installation"): 15,
    ("finishing works", "Millwork and carpentry"): 16
}

def get_sub_stage_progress_index(stage, sub_stage):
    return progress_indices.get((stage, sub_stage), -1)

def ensemble_predict(image, is_facade=False):
    if not MODELS_LOADED:
        raise ValueError("Models not loaded")

    inception_preprocessed = tf.image.resize(image, (299, 299))
    inception_preprocessed = tf.cast(inception_preprocessed, tf.float32) / 255.0
    inception_preprocessed = tf.expand_dims(inception_preprocessed, axis=0)

    mobilenet_preprocessed = tf.image.resize(image, (224, 224))
    mobilenet_preprocessed = tf.cast(mobilenet_preprocessed, tf.float32) / 255.0
    mobilenet_preprocessed = tf.expand_dims(mobilenet_preprocessed, axis=0)

    vgg_preprocessed = tf.image.resize(image, (224, 224))
    vgg_preprocessed = tf.cast(vgg_preprocessed, tf.float32) / 255.0
    vgg_preprocessed = tf.expand_dims(vgg_preprocessed, axis=0)

    mobilenet_output = global_mobilenet.predict(mobilenet_preprocessed, verbose=0)[0]
    inception_output = global_inception.predict(inception_preprocessed, verbose=0)[0]
    vgg_output = global_vgg.predict(vgg_preprocessed, verbose=0)[0]

    ensemble_output = (
        0.3 * mobilenet_output +
        0.4 * inception_output +
        0.3 * vgg_output
    )

    predicted_stage_index = np.argmax(ensemble_output)
    confidence_score = float(ensemble_output[predicted_stage_index] * 100)

    stage_list = list(stages.keys())
    predicted_stage = stage_list[predicted_stage_index]

    return predicted_stage, confidence_score

def classify_stage(image, selected_stage):
    if not MODELS_LOADED or selected_stage not in stage_specific_models:
        raise ValueError("Models not loaded or invalid stage")

    if selected_stage == "facade":
        preprocessed = tf.image.resize(image, (299, 299))
    else:
        preprocessed = tf.image.resize(image, (224, 224))

    preprocessed = tf.cast(preprocessed, tf.float32) / 255.0
    preprocessed = tf.expand_dims(preprocessed, axis=0)

    model = stage_specific_models[selected_stage]
    predictions = model.predict(preprocessed, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    confidence = float(predictions[predicted_index] * 100)

    sub_stages = stages[selected_stage]
    predicted_sub_stage = sub_stages[predicted_index]

    return predicted_sub_stage, confidence

def describe_image_with_gemini(image_path, max_retries=3, retry_delay=1):
    if not API_KEY:
        return "Image description unavailable: API key not configured"
    import time
    for attempt in range(max_retries):
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                contents=[
                    "Describe this construction site image briefly, focusing on the visible stage of construction",
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ],
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.7,
                    "top_k": 20,
                    "max_output_tokens": 200
                }
            )
            if hasattr(response, 'text'):
                return response.text
            else:
                return "Image description unavailable: No response from API"
        except Exception as e:
            error_message = str(e).lower()
            if 'rate limit' in error_message:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return "Image description temporarily unavailable: API rate limit reached. Please try again later."
            else:
                return f"Error generating description: {str(e)}"
    return "Image description unavailable after multiple attempts"

load_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/validate_image', methods=['POST'])
def validate_image():
    try:
        if not MODELS_LOADED:
            return jsonify({
                'error': 'Models are not loaded properly'
            }), 500

        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type'
            }), 400
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = Image.open(filepath).convert('RGB')
            image_array = np.array(image)

            selected_stage = request.form.get('stage')
            proceed = request.form.get('proceed', 'false').lower() in ['true', 'on', '1', 'yes']
            describe = request.form.get('describe', 'false').lower() in ['true', 'on', '1', 'yes']

            is_facade = (selected_stage == "facade")
            predicted_stage, global_confidence = ensemble_predict(image_array, is_facade)
            
            if predicted_stage != selected_stage and not proceed:
                return jsonify({
                    'success': False,
                    'message': f"Mismatch detected! Global classifier predicts stage '{predicted_stage}' "
                               f"with confidence {global_confidence:.2f}%, but selected stage is '{selected_stage}'."
                }), 200

            predicted_class, stage_confidence = classify_stage(image_array, selected_stage)
            description = None
            if describe:
                description = describe_image_with_gemini(filepath)

            doc_ref = db.collection('validations').document()
            validation_data = {
                'timestamp': firestore.SERVER_TIMESTAMP,
                'image_path': os.path.join(app.config['UPLOAD_FOLDER'], filename),
                'primary_stage': selected_stage,
                'specific_classification': predicted_class,
                'confidence_scores': {
                    'Stage Confidence': f"{stage_confidence:.2f}",
                    'Global Confidence': f"{global_confidence:.2f}"
                },
                'description_requested': describe,
                'ai_description': description if describe else "Description not generated",
                'status': 'success'
            }
            doc_ref.set(validation_data)

            ist = pytz.timezone('Asia/Kolkata')
            timestamp = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            return jsonify({
                'success': True,
                'message': f"The image matches the selected stage '{selected_stage}'. Predicted: '{predicted_class}' with confidence {stage_confidence:.2f}%.",
                'validation_id': doc_ref.id,
                'description': description,
                'timestamp': timestamp
            }), 200

        except Exception as e:
            return jsonify({
                'error': f'Error processing image: {str(e)}'
            }), 500

    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

stage_weights = {
    "foundation": 20,
    "superstructure": 30,
    "facade": 20,
    "Interior": 15,
    "finishing works": 15
}

sub_stage_weights = {
    "foundation": {
        "Excavation": 25,
        "Reinforcement Placement": 25,
        "concrete curing": 25,
        "concrete_pouring": 25
    },
    "superstructure": {
        "Roof_Decking": 15,
        "Stair Case": 20,
        "Structural_Frame_Erection_(framing)": 40,
        "Structural_Wall_Construction": 25
    },
    "facade": {
        "Exterior_Cladding_and_Finishes": 25,
        "Window_and_Door_Installation": 35,
        "exterior_wall_construction": 40
    },
    "Interior": {
        "Ceiling Installation": 35,
        "Flooring Installation": 35,
        "Staircase Finishing": 30
    },
    "finishing works": {
        "Painting": 35,
        "fixture installation": 35,
        "Millwork and carpentry": 30
    }
}

stage_indices = {
    'foundation': 1,
    'superstructure': 2,
    'facade': 3,
    'Interior': 4,
    'finishing works': 5
}

def calculate_progress(stage, sub_stage):
    """
    Cumulative Logic:
    We sum all sub-stages from the beginning of the stage up to and including the predicted sub-stage.
    This ensures that if we move from an earlier sub-stage to a later one, we consider all intermediate 
    sub-stages completed as well.

    For example:
    - If previously at "Excavation" (foundation) = 25%
      and now at "concrete curing", we sum: Excavation(25%) + Reinforcement Placement(25%) + concrete curing(25%) = 75% complete for foundation.
    - If we moved to a new stage (e.g., from foundation to superstructure), we consider foundation 100% complete.
    """

    if stage not in stages or sub_stage not in stages[stage]:
        return 0, 0, []

    stage_sub_stages = stages[stage]
    current_index = stage_sub_stages.index(sub_stage)

    # Cumulative sum of all sub-stages up to current_index
    stage_progress = 0
    for i in range(current_index + 1):
        sstg = stage_sub_stages[i]
        stage_progress += sub_stage_weights[stage][sstg]

    # stage_progress is a percentage of that stage
    current_stage_index = stage_indices[stage]
    overall_progress = 0
    completed_stages = []

    # Add fully completed previous stages
    for s, idx in stage_indices.items():
        if idx < current_stage_index:
            overall_progress += stage_weights[s]
            completed_stages.append(s)

    # Add the current stage's partial completion
    stage_contribution = (stage_weights[stage] * (stage_progress / 100.0))
    overall_progress += stage_contribution

    # If this sub-stage completes the stage at 100%, add it to completed_stages
    if stage_progress == 100:
        completed_stages.append(stage)

    completed_stages.sort(key=lambda x: stage_indices[x])
    return stage_progress, overall_progress, completed_stages

def get_progress_message(prev_stage, prev_sub_stage, curr_stage, curr_sub_stage):
    prev_stage_progress, prev_overall_progress, prev_completed = calculate_progress(prev_stage, prev_sub_stage)
    curr_stage_progress, curr_overall_progress, curr_completed = calculate_progress(curr_stage, curr_sub_stage)

    stage_progress_diff = curr_stage_progress - prev_stage_progress if curr_stage == prev_stage else curr_stage_progress
    overall_progress_diff = curr_overall_progress - prev_overall_progress

    prev_stage_idx = stage_indices[prev_stage]
    curr_stage_idx = stage_indices[curr_stage]

    if curr_stage_idx < prev_stage_idx:
        status = 'invalid'
        message = [
            f"Invalid progress: Cannot move from {prev_stage} (Stage {prev_stage_idx}) "
            f"to {curr_stage} (Stage {curr_stage_idx})",
            "Construction stages must proceed in order."
        ]
        return status, "\n".join(message)

    if curr_overall_progress > prev_overall_progress:
        status = 'advanced'
        message = [
            f"Progress has advanced from Stage {prev_stage_idx}: {prev_stage} ({prev_sub_stage}) "
            f"to Stage {curr_stage_idx}: {curr_stage} ({curr_sub_stage})"
        ]
        newly_completed = set(curr_completed) - set(prev_completed)
        if newly_completed:
            completed_info = [f"Stage {stage_indices[s]}: {s}" for s in sorted(newly_completed, key=lambda x: stage_indices[x])]
            message.append(f"Completed stages: {', '.join(completed_info)} (100%)")

        message.append(
            f"Current Stage {curr_stage_idx} ({curr_stage}) is {curr_stage_progress:.1f}% complete "
            f"(contributing {((stage_weights[curr_stage] * curr_stage_progress) / 100.0):.1f}% to overall progress)"
        )

        message.append("Project Progress Breakdown:")
        # List fully or partially completed stages
        for completed_stage in curr_completed:
            if completed_stage == curr_stage and curr_stage_progress < 100:
                # Partially completed current stage
                partial_contribution = (stage_weights[curr_stage] * (curr_stage_progress / 100.0))
                message.append(
                    f"- Stage {curr_stage_idx}: {curr_stage}: {curr_stage_progress:.1f}% "
                    f"(contributing {partial_contribution:.1f}% out of possible {stage_weights[curr_stage]}%)"
                )
            else:
                # Fully completed stages
                message.append(
                    f"- Stage {stage_indices[completed_stage]}: {completed_stage}: "
                    f"100% (contributing {stage_weights[completed_stage]}%)"
                )

        message.append(f"Overall project is {curr_overall_progress:.1f}% complete (+{overall_progress_diff:.1f}%)")

    elif curr_overall_progress == prev_overall_progress:
        status = 'same'
        message = [
            f"No progress detected. Staying at Stage {curr_stage_idx}: {curr_stage} ({curr_sub_stage})",
            f"Current stage is {curr_stage_progress:.1f}% complete",
            f"Overall project is {curr_overall_progress:.1f}% complete"
        ]
    else:
        status = 'regressed'
        message = [
            f"Progress has regressed from Stage {prev_stage_idx}: {prev_stage} ({prev_sub_stage}) "
            f"to Stage {curr_stage_idx}: {curr_stage} ({curr_sub_stage})",
            f"Current stage ({curr_stage}) is {curr_stage_progress:.1f}% complete",
            f"Overall project is {curr_overall_progress:.1f}% complete ({overall_progress_diff:.1f}% change)"
        ]

    return status, "\n".join(message)

@app.route('/compare', methods=['POST'])
def compare_progress():
    previous_doc_id = request.form.get('previous_doc_id')
    current_doc_id = request.form.get('current_doc_id')

    if not previous_doc_id or not current_doc_id:
        return jsonify({'error': 'Missing required parameters: previous_doc_id, current_doc_id'}), 400

    try:
        # Fetch previous record
        prev_doc_ref = db.collection('validations').document(previous_doc_id)
        prev_doc = prev_doc_ref.get()
        if not prev_doc.exists:
            return jsonify({'error': 'Previous document not found'}), 404
        prev_data = prev_doc.to_dict()

        # Fetch current record
        curr_doc_ref = db.collection('validations').document(current_doc_id)
        curr_doc = curr_doc_ref.get()
        if not curr_doc.exists:
            return jsonify({'error': 'Current document not found'}), 404
        curr_data = curr_doc.to_dict()

        # Get stage information using new field names
        prev_stage = prev_data.get('primary_stage')
        prev_sub_stage = prev_data.get('specific_classification')
        prev_image_path = prev_data.get('image_path', '')

        curr_stage = curr_data.get('primary_stage')
        curr_sub_stage = curr_data.get('specific_classification')
        curr_image_path = curr_data.get('image_path', '')

        # Convert image paths to URLs
        prev_image_url = '/' + prev_image_path if prev_image_path else ''
        curr_image_url = '/' + curr_image_path if curr_image_path else ''

        progress_status, progress_message = get_progress_message(
            prev_stage, prev_sub_stage, curr_stage, curr_sub_stage
        )

        curr_stage_progress, curr_overall_progress, curr_completed = calculate_progress(curr_stage, curr_sub_stage)

        return jsonify({
            'progress': progress_status,
            'message': progress_message,
            'previous': {
                'stage': prev_stage,
                'sub_stage': prev_sub_stage,
                'image_url': prev_image_url
            },
            'current': {
                'stage': curr_stage,
                'sub_stage': curr_sub_stage,
                'image_url': curr_image_url,
                'stage_progress': curr_stage_progress,
                'overall_progress': curr_overall_progress,
                'completed_stages': curr_completed
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_validations', methods=['GET'])
def get_validations():
    try:
        docs = db.collection('validations').order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        validations = []
        for doc in docs:
            data = doc.to_dict()
            validations.append({
                'id': doc.id,
                'timestamp': data.get('timestamp', ''),
                'stage': data.get('primary_stage', ''),
                'sub_stage': data.get('specific_classification', '')
            })
        
        return jsonify({'validations': validations}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve validations: {str(e)}'}), 500

@app.route('/generate_report/<validation_id>', methods=['GET'])
def generate_report(validation_id):
    try:
        # Get validation data from Firestore
        doc_ref = db.collection('validations').document(validation_id)
        validation = doc_ref.get()
        
        if not validation.exists:
            return jsonify({'error': 'Validation not found'}), 404
            
        validation_data = validation.to_dict()
        
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            # Create the PDF document
            doc = SimpleDocTemplate(tmp.name, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            story.append(Paragraph('Construction Stage Analysis Report', title_style))
            story.append(Spacer(1, 12))
            
            # Date
            story.append(Paragraph(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Add image if it exists
            image_path = validation_data.get('image_path', '')
            if os.path.exists(image_path):
                img = RLImage(image_path, width=400, height=300)
                story.append(img)
                story.append(Spacer(1, 12))
            
            # Analysis Results
            story.append(Paragraph('Stage Analysis Results', styles['Heading2']))
            story.append(Paragraph(f'Primary Stage: {validation_data.get("primary_stage", "N/A")}', styles['Normal']))
            story.append(Paragraph(f'Stage-Specific Classification: {validation_data.get("specific_classification", "N/A")}', styles['Normal']))
            story.append(Spacer(1, 12))
            
            # AI Description
            story.append(Paragraph('AI Description', styles['Heading2']))
            description = validation_data.get('ai_description', 'Description not generated')
            if not validation_data.get('description_requested', False):
                description = 'Description not generated'
            story.append(Paragraph(description, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Confidence Scores
            story.append(Paragraph('Confidence Scores', styles['Heading2']))
            confidence_scores = validation_data.get('confidence_scores', {})
            for model, score in confidence_scores.items():
                story.append(Paragraph(f'{model}: {score}%', styles['Normal']))
            
            # Build the PDF
            doc.build(story)
            
            # Return the PDF file
            return send_file(
                tmp.name,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'construction_report_{validation_id}.pdf'
            )
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
