from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from firebase_admin import credentials, firestore
import firebase_admin
from auth import create_user, verify_user, login_required, role_required, set_firestore_client
import time
import pytz
from datetime import datetime
from werkzeug.utils import secure_filename
import tempfile
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import json
import onnxruntime as ort
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor
import base64
from io import BytesIO

load_dotenv()

app = Flask(__name__)
app.secret_key = '6e9c8e4e0d8f4a36a3f5e87e9f1d2c4785e94d7328b8cfd39cf2c03b00a3d2f7'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=API_KEY)

FIREBASE_CRED_PATH = os.path.join(os.path.dirname(__file__), 'config', 'fnatic-2cba7-firebase-adminsdk-ccao3-c65eb4de07.json')

try:
    if not os.path.exists(FIREBASE_CRED_PATH):
        raise FileNotFoundError(f"Firebase credentials file not found at: {FIREBASE_CRED_PATH}")
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    set_firestore_client(db)
except Exception as e:
    print(f"Error initializing Firebase: {str(e)}")
    raise

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

# Final Stages for Prediction (as requested):
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

# Stage weights remain the same
stage_weights = {
    "foundation": 20,
    "superstructure": 30,
    "facade": 20,
    "Interior": 15,
    "finishing works": 15
}

# Sub-stage weights (keys remain consistent, no order dependency here)
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

# Final Progress Indices with swapped 2 and 3 for foundation
progress_indices = {
    ("foundation", "Excavation"): 0,
    ("foundation", "Reinforcement Placement"): 1,
    ("foundation", "concrete_pouring"): 2,   # swapped with curing
    ("foundation", "concrete curing"): 3,     # swapped with pouring

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
    inception_preprocessed = tf.reshape(inception_preprocessed, (1, 299, 299, 3))

    mobilenet_preprocessed = tf.image.resize(image, (224, 224))
    mobilenet_preprocessed = tf.cast(mobilenet_preprocessed, tf.float32) / 255.0
    mobilenet_preprocessed = tf.reshape(mobilenet_preprocessed, (1, 224, 224, 3))

    vgg_preprocessed = tf.image.resize(image, (224, 224))
    vgg_preprocessed = tf.cast(vgg_preprocessed, tf.float32) / 255.0
    vgg_preprocessed = tf.reshape(vgg_preprocessed, (1, 224, 224, 3))

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

def calculate_progress(stage, sub_stage):
    if stage not in stages or sub_stage not in stages[stage]:
        return 0, 0, []

    stage_sub_stages = stages[stage]
    current_index = stage_sub_stages.index(sub_stage)

    stage_progress = 0
    for i in range(current_index + 1):
        sstg = stage_sub_stages[i]
        stage_progress += sub_stage_weights[stage][sstg]

    # Determine stage index (1 to 5)
    # foundation=1, superstructure=2, facade=3, Interior=4, finishing works=5
    stage_order = ["foundation", "superstructure", "facade", "Interior", "finishing works"]
    current_stage_index = stage_order.index(stage) + 1

    overall_progress = 0
    completed_stages = []

    for s in stage_order:
        if stage_order.index(s) + 1 < current_stage_index:
            overall_progress += stage_weights[s]
            completed_stages.append(s)

    stage_contribution = (stage_weights[stage] * (stage_progress / 100.0))
    overall_progress += stage_contribution

    if stage_progress == 100:
        completed_stages.append(stage)

    completed_stages.sort(key=lambda x: stage_order.index(x) + 1)

    return stage_progress, overall_progress, completed_stages

def get_progress_message(prev_stage, prev_sub_stage, curr_stage, curr_sub_stage):
    prev_stage_progress, prev_overall_progress, prev_completed = calculate_progress(prev_stage, prev_sub_stage)
    curr_stage_progress, curr_overall_progress, curr_completed = calculate_progress(curr_stage, curr_sub_stage)

    stage_order = ["foundation", "superstructure", "facade", "Interior", "finishing works"]
    prev_stage_idx = stage_order.index(prev_stage) + 1
    curr_stage_idx = stage_order.index(curr_stage) + 1

    stage_progress_diff = curr_stage_progress - prev_stage_progress if curr_stage == prev_stage else curr_stage_progress
    overall_progress_diff = curr_overall_progress - prev_overall_progress

    if curr_stage_idx < prev_stage_idx:
        status = 'invalid'
        message = [
            f"Invalid progress: Cannot move from {prev_stage} (Stage {prev_stage_idx}) to {curr_stage} (Stage {curr_stage_idx})",
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
            completed_info = [f"Stage {stage_order.index(s)+1}: {s}" for s in sorted(newly_completed, key=lambda x: stage_order.index(x))]
            message.append(f"Completed stages: {', '.join(completed_info)} (100%)")

        message.append(
            f"Current Stage {curr_stage_idx} ({curr_stage}) is {curr_stage_progress:.1f}% complete "
            f"(contributing {((stage_weights[curr_stage] * curr_stage_progress) / 100.0):.1f}% to overall progress)"
        )

        message.append("Project Progress Breakdown:")
        for completed_stage in curr_completed:
            if completed_stage == curr_stage and curr_stage_progress < 100:
                partial_contribution = (stage_weights[curr_stage] * (curr_stage_progress / 100.0))
                message.append(
                    f"- Stage {curr_stage_idx}: {curr_stage}: {curr_stage_progress:.1f}% "
                    f"(contributing {partial_contribution:.1f}% out of possible {stage_weights[curr_stage]}%)"
                )
            else:
                message.append(
                    f"- Stage {stage_order.index(completed_stage)+1}: {completed_stage}: "
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
@login_required
def compare_progress():
    try:
        data = request.form
        user_id = session.get('user_id')
        project_id = data.get('project_id')
        previous_doc_id = data.get('previous_doc_id')
        current_doc_id = data.get('current_doc_id')

        if not all([user_id, project_id, previous_doc_id, current_doc_id]):
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400

        project_ref = db.collection('users').document(user_id).collection('projects').document(project_id)
        
        previous_validation = project_ref.collection('validations').document(previous_doc_id).get()
        current_validation = project_ref.collection('validations').document(current_doc_id).get()

        if not previous_validation.exists or not current_validation.exists:
            return jsonify({
                'success': False,
                'error': 'One or both validations not found'
            }), 404

        prev_data = previous_validation.to_dict()
        curr_data = current_validation.to_dict()

        prev_stage = prev_data.get('primary_stage')
        prev_sub_stage = prev_data.get('specific_classification')
        curr_stage = curr_data.get('primary_stage')
        curr_sub_stage = curr_data.get('specific_classification')

        status, progress_message = get_progress_message(prev_stage, prev_sub_stage, curr_stage, curr_sub_stage)
        prev_stage_progress, prev_overall_progress, prev_completed = calculate_progress(prev_stage, prev_sub_stage)
        curr_stage_progress, curr_overall_progress, curr_completed = calculate_progress(curr_stage, curr_sub_stage)

        ist = pytz.timezone('Asia/Kolkata')
        prev_timestamp = prev_data.get('timestamp')
        curr_timestamp = curr_data.get('timestamp')
        
        if prev_timestamp:
            prev_timestamp = prev_timestamp.astimezone(ist).strftime('%Y-%m-%d %H:%M:%S %Z')
        if curr_timestamp:
            curr_timestamp = curr_timestamp.astimezone(ist).strftime('%Y-%m-%d %H:%M:%S %Z')

        # Fix image paths by ensuring they are relative to static directory
        prev_image_path = prev_data.get('image_path', '')
        curr_image_path = curr_data.get('image_path', '')
        
        # Convert absolute paths to relative paths for static serving
        if prev_image_path:
            prev_image_path = os.path.basename(prev_image_path)
            prev_image_path = f'uploads/{prev_image_path}'
        if curr_image_path:
            curr_image_path = os.path.basename(curr_image_path)
            curr_image_path = f'uploads/{curr_image_path}'

        return jsonify({
            'success': True,
            'previous': {
                'stage': prev_stage,
                'sub_stage': prev_sub_stage,
                'stage_progress': prev_stage_progress,
                'overall_progress': prev_overall_progress,
                'completed_stages': prev_completed,
                'timestamp': prev_timestamp,
                'image_path': prev_image_path
            },
            'current': {
                'stage': curr_stage,
                'sub_stage': curr_sub_stage,
                'stage_progress': curr_stage_progress,
                'overall_progress': curr_overall_progress,
                'completed_stages': curr_completed,
                'timestamp': curr_timestamp,
                'image_path': curr_image_path
            },
            'progress_status': status,
            'progress_message': progress_message
        })

    except Exception as e:
        app.logger.error(f"Error in compare_progress: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

load_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/home')
@login_required
def home():
    project_id = request.args.get('project_id')
    if not project_id:
        return redirect(url_for('worker_dashboard' if session.get('role') == 'worker' else 'expert_dashboard'))
    
    try:
        user_id = session.get('user_id')
        project_ref = db.collection('users').document(user_id).collection('projects').document(project_id)
        project = project_ref.get()
        
        if not project.exists:
            flash('Project not found', 'error')
            return redirect(url_for('worker_dashboard'))
            
        project_data = project.to_dict()
        project_data['id'] = project.id
        
        validations_ref = project_ref.collection('validations')\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .limit(5)
        
        validations = []
        latest_validation = None
        for validation in validations_ref.stream():
            validation_data = validation.to_dict()
            validation_data['id'] = validation.id
            validations.append(validation_data)
            if not latest_validation:
                latest_validation = validation_data

        if latest_validation:
            current_stage = latest_validation.get('primary_stage', 'Not started')
            current_sub_stage = latest_validation.get('specific_classification', 'Not started')
            project_data['current_stage'] = current_stage
            project_data['current_sub_stage'] = current_sub_stage
            
            stage_progress, overall_progress, completed_stages = calculate_progress(current_stage, current_sub_stage)
            project_data['stage_progress'] = stage_progress
            project_data['progress_percentage'] = overall_progress
            project_data['completed_stages'] = completed_stages
        else:
            project_data['progress_percentage'] = 0
            project_data['current_stage'] = 'Not started'
            project_data['current_sub_stage'] = 'Not started'
            project_data['stage_progress'] = 0
            project_data['completed_stages'] = []
            
        return render_template('home.html', 
                             project=project_data,
                             validations=validations,
                             user_role=session.get('role'))
    except Exception as e:
        app.logger.error(f"Error in home route: {str(e)}")
        flash('Error loading project', 'error')
        return redirect(url_for('worker_dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        print("\n" + "=" * 50)
        print("🔍 FORM DATA INSPECTION:")
        for key, value in request.form.items():
            print(f"{key}: {value}")
        
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        print("\n" + "=" * 50)
        print("🚀 LOGIN ATTEMPT:")
        print(f"Email: {email}")
        print(f"Password Length: {len(password)}")
        
        try:
            user = verify_user(email, password)
            user_doc = db.collection('users').document(user['user_id']).get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
                
                session['user_id'] = user['user_id']
                session['email'] = email
                session['role'] = user_data.get('role', 'unknown')
                
                print("\n📋 SESSION DETAILS:")
                print(f"User ID: {session['user_id']}")
                print(f"Email: {session['email']}")
                print(f"Role: {session['role']}")
                
                print("\n🚪 ROLE-BASED ROUTING:")
                if session['role'] == 'expert':
                    print("Redirecting to EXPERT Dashboard")
                    return redirect(url_for('expert_dashboard'))
                elif session['role'] == 'worker':
                    print("Redirecting to WORKER Dashboard")
                    return redirect(url_for('worker_dashboard'))
                else:
                    print("❌ INVALID USER ROLE")
                    flash('Invalid user role', 'error')
                    return redirect(url_for('login'))
            else:
                print("\n❌ NO USER DOCUMENT FOUND IN FIRESTORE")
                flash('User not found in database', 'error')
                return redirect(url_for('login'))
        
        except Exception as auth_error:
            print("\n❌ AUTHENTICATION ERROR:")
            print(f"Error: {auth_error}")
            flash(str(auth_error), 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user_role = session.get('role')
    if user_role == 'expert':
        return redirect(url_for('expert_dashboard'))
    elif user_role == 'worker':
        return redirect(url_for('worker_dashboard'))
    else:
        flash('Invalid user role', 'error')
        return redirect(url_for('login'))

@app.route('/create_project', methods=['POST'])
@login_required
def create_project():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'User not logged in'
            }), 401

        name = request.form.get('name')
        description = request.form.get('description')
        location = request.form.get('location')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        latitude = request.form.get('latitude', type=float)
        longitude = request.form.get('longitude', type=float)

        if not all([name, description, location, start_date, end_date, latitude, longitude]):
            return jsonify({
                'success': False,
                'error': 'All fields are required'
            }), 400

        project_ref = db.collection('users').document(user_id)\
            .collection('projects').document()
        
        project_data = {
            'name': name,
            'description': description,
            'location': location,
            'start_date': start_date,
            'end_date': end_date,
            'created_at': firestore.SERVER_TIMESTAMP,
            'created_by': user_id,
            'status': 'active',
            'current_stage': 'Not started',
            'current_sub_stage': 'Not started',
            'latitude': latitude,
            'longitude': longitude
        }
        
        project_ref.set(project_data)
        
        project = project_ref.get()
        response_data = project.to_dict()
        response_data['id'] = project.id
        
        return jsonify({
            'success': True,
            'message': 'Project created successfully',
            'project': response_data
        }), 200

    except Exception as e:
        app.logger.error(f"Error creating project: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error creating project: {str(e)}'
        }), 500

@app.route('/expert_dashboard')
@login_required
@role_required('expert')
def expert_dashboard():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return redirect(url_for('login'))
            
        projects_ref = db.collection('users').document(user_id).collection('projects')
        projects = []
        
        for doc in projects_ref.stream():
            project_data = doc.to_dict()
            project_data['id'] = doc.id
            
            validations_ref = projects_ref.document(doc.id).collection('validations')
            latest_validation = None
            
            for validation in validations_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream():
                latest_validation = validation.to_dict()
                break
                
            if latest_validation:
                current_stage = latest_validation.get('primary_stage', 'Not Started')
                current_sub_stage = latest_validation.get('specific_classification', '')
                project_data['current_stage'] = current_stage
                project_data['current_sub_stage'] = current_sub_stage
                
                stage_progress, overall_progress, completed_stages = calculate_progress(current_stage, current_sub_stage)
                project_data['progress_percentage'] = overall_progress
                project_data['completed_stages'] = completed_stages
            else:
                project_data['current_stage'] = 'Not Started'
                project_data['current_sub_stage'] = ''
                project_data['progress_percentage'] = 0
                project_data['completed_stages'] = []
                
            project_data['status'] = project_data.get('status', 'active')
            projects.append(project_data)
            
        return render_template('expert_dashboard.html', 
                             projects=projects, 
                             user_id=user_id,
                             user_role='expert')
                             
    except Exception as e:
        app.logger.error(f"Error in expert_dashboard: {str(e)}")
        flash('An error occurred while loading the dashboard', 'danger')
        return redirect(url_for('login'))

@app.route('/worker_dashboard')
@login_required
def worker_dashboard():
    try:
        user_id = session.get('user_id')
        if not user_id:
            flash('Please log in first', 'error')
            return redirect(url_for('login'))

        projects_ref = db.collection('users').document(user_id)\
            .collection('projects')\
            .order_by('created_at', direction=firestore.Query.DESCENDING)
        
        projects = []
        for doc in projects_ref.stream():
            project_data = doc.to_dict()
            project_data['id'] = doc.id
            
            validations_ref = db.collection('users').document(user_id)\
                .collection('projects').document(doc.id)\
                .collection('validations')\
                .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                .limit(1)
            
            validations = list(validations_ref.stream())
            if validations:
                latest_validation = validations[0].to_dict()
                current_stage = latest_validation.get('primary_stage', 'Not Started')
                current_sub_stage = latest_validation.get('specific_classification', 'Not Started')
                project_data['current_stage'] = current_stage
                project_data['current_sub_stage'] = current_sub_stage
                
                stage_progress, overall_progress, completed_stages = calculate_progress(current_stage, current_sub_stage)
                project_data['stage_progress'] = stage_progress
                project_data['progress_percentage'] = overall_progress
                project_data['completed_stages'] = completed_stages
                
                if latest_validation.get('timestamp'):
                    project_data['last_updated'] = latest_validation['timestamp']
            else:
                project_data['current_stage'] = 'Not Started'
                project_data['current_sub_stage'] = 'Not Started'
                project_data['stage_progress'] = 0
                project_data['progress_percentage'] = 0
                project_data['completed_stages'] = []
            
            projects.append(project_data)

        return render_template('worker_dashboard.html',
                             projects=projects,
                             user_id=user_id)
    except Exception as e:
        app.logger.error(f"Error in worker dashboard: {str(e)}")
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return redirect(url_for('login'))

@app.route('/project/<project_id>')
@login_required
def view_project(project_id):
    try:
        project_ref = db.collection('projects').document(project_id)
        project = project_ref.get()
        
        if not project.exists:
            flash('Project not found', 'error')
            return redirect(url_for('expert_dashboard' if session.get('role') == 'expert' else 'worker_dashboard'))
        
        project_data = project.to_dict()
        project_data['id'] = project_id
        
        validations_ref = db.collection('validations')\
            .where('project_id', '==', project_id)\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .limit(10)
        
        recent_validations = []
        for doc in validations_ref.stream():
            validation_data = doc.to_dict()
            validation_data['id'] = doc.id
            recent_validations.append(validation_data)
        
        return render_template(
            'project_details.html',
            project=project_data,
            recent_validations=recent_validations,
            user=session.get('email'),
            role=session.get('role')
        )
    except Exception as e:
        flash(f'Error loading project: {str(e)}', 'error')
        return redirect(url_for('expert_dashboard' if session.get('role') == 'expert' else 'worker_dashboard'))

@app.route('/project_details/<project_id>')
@login_required
def project_details(project_id):
    try:
        project_ref = db.collection('projects').document(project_id)
        project = project_ref.get()
        
        if not project.exists:
            flash('Project not found', 'error')
            return redirect(url_for('expert_dashboard' if session['role'] == 'expert' else 'worker_dashboard'))
        
        project_data = project.to_dict()
        project_data['id'] = project_id
        project_data['progress_percentage'] = project_data.get('progress_percentage', 0)
        
        if session['role'] == 'worker' and project_data.get('created_by_id') != session['user_id']:
            flash('You do not have permission to view this project', 'error')
            return redirect(url_for('worker_dashboard'))
        
        return render_template('home.html', 
                             project=project_data,
                             project_name=project_data.get('name', 'Unknown Project'))
    except Exception as e:
        flash(f'Error loading project: {str(e)}', 'error')
        return redirect(url_for('expert_dashboard' if session['role'] == 'expert' else 'worker_dashboard'))

@app.route('/project/<project_id>/validate', methods=['GET', 'POST'])
@login_required
@role_required(['expert', 'worker'])
def validate_project_images(project_id):
    try:
        user_id = session.get('user_id')
        project_ref = db.collection('users').document(user_id)\
            .collection('projects').document(project_id)
        project = project_ref.get()
        
        if not project.exists:
            flash('Project not found', 'error')
            return redirect(url_for('expert_dashboard' if session['role'] == 'expert' else 'worker_dashboard'))
        
        project_data = project.to_dict()
        project_data['id'] = project_id
        
        if session['role'] == 'worker' and project_data.get('created_by_id') != session['user_id']:
            flash('You do not have permission to validate this project', 'error')
            return redirect(url_for('worker_dashboard'))
        
        # Get all validations for comparison
        validations_ref = project_ref.collection('validations')\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .stream()
            
        validations = []
        for validation in validations_ref:
            validation_data = validation.to_dict()
            validation_data['id'] = validation.id
            # Convert timestamp to datetime if it exists
            if validation_data.get('timestamp'):
                validation_data['timestamp'] = validation_data['timestamp']
            validations.append(validation_data)
        
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file uploaded', 'error')
                return redirect(url_for('validate_project_images', project_id=project_id))
            
            uploaded_file = request.files['file']
            stage = request.form.get('stage')
            proceed_if_mismatch = request.form.get('proceed', 'false').lower() == 'true'
            generate_description = request.form.get('describe', 'false').lower() == 'true'
            
            if uploaded_file.filename == '':
                flash('No file selected', 'error')
                return redirect(url_for('validate_project_images', project_id=project_id))
            
            if uploaded_file and allowed_file(uploaded_file.filename):
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(file_path)
                
                try:
                    img = Image.open(file_path)
                    img = img.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    is_facade = stage.lower() == 'facade'
                    prediction = ensemble_predict(img_array[0], is_facade)
                    
                    predicted_stage = prediction[0]
                    stage_confidence = prediction[1]
                    
                    try:
                        predicted_sub_stage, sub_stage_confidence = classify_stage(img_array[0], predicted_stage)
                    except:
                        predicted_sub_stage = 'Not Specified'
                        sub_stage_confidence = 0
                    
                    matches = predicted_stage.lower() == stage.lower()
                    
                    validation_result = {
                        'matches': matches,
                        'stage_confidence': stage_confidence,
                        'sub_stage_confidence': sub_stage_confidence,
                        'predicted_sub_stage': predicted_sub_stage,
                        'predicted_stage': predicted_stage,
                        'selected_stage': stage,
                        'timestamp': firestore.SERVER_TIMESTAMP,
                        'validated_by': session['user_id'],
                        'project_id': project_id,
                        'image_path': file_path
                    }
                    
                    if generate_description:
                        try:
                            description = describe_image_with_gemini(file_path)
                            validation_result['ai_description'] = description
                        except Exception as e:
                            validation_result['description_error'] = str(e)
                    
                    user_ref = db.collection('users').document(session['user_id'])
                    projects_ref = user_ref.collection('projects')
                    project_doc_ref = projects_ref.document(project_id)
                    project_doc = project_doc_ref.get()
                    if not project_doc.exists:
                        project_data_to_store = {
                            'project_id': project_id,
                            'name': project_data.get('name', 'Unnamed Project'),
                            'location': project_data.get('location', 'Unknown Location'),
                            'current_stage': project_data.get('current_stage', 'Not Started'),
                            'created_at': firestore.SERVER_TIMESTAMP,
                            'created_by': session['user_id']
                        }
                        project_doc_ref.set(project_data_to_store)
                    
                    validations_ref = project_doc_ref.collection('validations')
                    validation_doc = validations_ref.add(validation_result)
                    added_doc_ref = validation_doc[1]
                    
                    if matches or proceed_if_mismatch:
                        project_ref.update({
                            'current_stage': stage,
                            'last_updated': firestore.SERVER_TIMESTAMP,
                            'last_validated_by': session['user_id']
                        })
                    
                    flash('Validation completed successfully', 'success')
                    return redirect(url_for('validate_project_images', project_id=project_id))
                    
                except Exception as e:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    flash(f'Error processing image: {str(e)}', 'error')
                    return redirect(url_for('validate_project_images', project_id=project_id))
            else:
                flash('Invalid file type', 'error')
                return redirect(url_for('validate_project_images', project_id=project_id))
        
        return render_template('home.html', 
                             project=project_data,
                             project_name=project_data.get('name', 'Unknown Project'),
                             stages=stages,
                             validations=validations)
                             
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('expert_dashboard' if session['role'] == 'expert' else 'worker_dashboard'))

@app.route('/get_validations', methods=['GET'])
@login_required
def get_validations():
    try:
        user_id = session.get('user_id')
        project_id = request.args.get('project_id')
        
        if not user_id or not project_id:
            return jsonify({
                'success': False,
                'error': 'User ID and Project ID are required'
            }), 400

        validations_ref = db.collection('users').document(user_id)\
            .collection('projects').document(project_id)\
            .collection('validations')\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .limit(10)
        
        validations = []
        for doc in validations_ref.stream():
            validation = doc.to_dict()
            validation['id'] = doc.id
            if validation.get('timestamp'):
                validation['timestamp'] = validation['timestamp'].astimezone(pytz.timezone('Asia/Kolkata'))
            validations.append(validation)
        
        return jsonify({
            'success': True,
            'validations': validations
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error getting validations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/validate_image', methods=['POST'])
@login_required
def validate_image():
    try:
        if not MODELS_LOADED:
            return jsonify({
                'success': False,
                'error': 'Models are not loaded properly'
            }), 500

        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG'
            }), 400

        user_id = session.get('user_id')
        project_id = request.form.get('project_id')
        if not user_id or not project_id:
            return jsonify({
                'success': False,
                'error': 'User ID and Project ID are required'
            }), 400

        project_ref = db.collection('users').document(user_id)\
            .collection('projects').document(project_id)
        project = project_ref.get()
        if not project.exists:
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = Image.open(filepath).convert('RGB')
            image_array = np.array(image)

            selected_stage = request.form.get('stage')
            proceed = request.form.get('proceed', 'false').lower() in ['true', 'on', '1', 'yes']
            describe = request.form.get('describe', 'false').lower() in ['true', 'on', '1', 'yes']

            if not selected_stage:
                return jsonify({
                    'success': False,
                    'error': 'Stage selection is required'
                }), 400

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

            validation_data = {
                'timestamp': firestore.SERVER_TIMESTAMP,
                'project_id': project_id,
                'image_path': filepath,
                'primary_stage': selected_stage,
                'specific_classification': predicted_class,
                'confidence_scores': {
                    'Stage_Confidence': f"{stage_confidence:.2f}",
                    'Global_Confidence': f"{global_confidence:.2f}"
                },
                'description_requested': describe,
                'ai_description': description if describe else None,
                'status': 'success',
                'validated_by': user_id,
                'validation_type': 'ai'
            }

            validation_ref = project_ref.collection('validations').document()
            validation_ref.set(validation_data)

            project_ref.update({
                'last_validation_id': validation_ref.id,
                'last_validation_timestamp': firestore.SERVER_TIMESTAMP,
                'current_stage': selected_stage,
                'current_sub_stage': predicted_class
            })

            ist = pytz.timezone('Asia/Kolkata')
            timestamp = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            return jsonify({
                'success': True,
                'message': f"The image matches the selected stage '{selected_stage}'. Predicted: '{predicted_class}' with confidence {stage_confidence:.2f}%.",
                'validation_id': validation_ref.id,
                'description': description,
                'timestamp': timestamp
            }), 200

        except Exception as e:
            if 'filepath' in locals():
                try:
                    os.remove(filepath)
                except:
                    pass
            return jsonify({
                'success': False,
                'error': f'Error processing image: {str(e)}'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/generate_report/<validation_id>', methods=['GET'])
def generate_report(validation_id):
    try:
        doc_ref = db.collection('validations').document(validation_id)
        validation = doc_ref.get()
        
        if not validation.exists:
            return jsonify({'error': 'Validation not found'}), 404
            
        validation_data = validation.to_dict()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            doc = SimpleDocTemplate(tmp.name, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            story.append(Paragraph('Construction Stage Analysis Report', title_style))
            story.append(Spacer(1, 12))
            
            story.append(Paragraph(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', styles['Normal']))
            story.append(Spacer(1, 12))
            
            image_path = validation_data.get('image_path', '')
            if os.path.exists(image_path):
                img = RLImage(image_path, width=400, height=300)
                story.append(img)
                story.append(Spacer(1, 12))
            
            story.append(Paragraph('Stage Analysis Results', styles['Heading2']))
            story.append(Paragraph(f'Primary Stage: {validation_data.get("primary_stage", "N/A")}', styles['Normal']))
            story.append(Paragraph(f'Stage-Specific Classification: {validation_data.get("specific_classification", "N/A")}', styles['Normal']))
            story.append(Spacer(1, 12))
            
            story.append(Paragraph('AI Description', styles['Heading2']))
            description = validation_data.get('ai_description', 'Description not generated')
            if not validation_data.get('description_requested', False):
                description = 'Description not generated'
            story.append(Paragraph(description, styles['Normal']))
            story.append(Spacer(1, 12))
            
            story.append(Paragraph('Confidence Scores', styles['Heading2']))
            confidence_scores = validation_data.get('confidence_scores', {})
            for model, score in confidence_scores.items():
                story.append(Paragraph(f'{model}: {score}%', styles['Normal']))
            
            doc.build(story)
            
            return send_file(
                tmp.name,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'construction_report_{validation_id}.pdf'
            )
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role', 'worker')
        
        result = create_user(email, password, role)
        if result:
            flash('Signup successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Signup failed. Please try again.', 'error')
    
    return render_template('signup.html')

@app.route('/geo-map')
@login_required
def geo_map():
    try:
        user_id = session.get('user_id')
        if not user_id:
            flash('Please log in first', 'error')
            return redirect(url_for('login'))

        # Get projects from Firestore
        projects_ref = db.collection('users').document(user_id).collection('projects')
        projects = []
        
        for doc in projects_ref.stream():
            data = doc.to_dict()
            # Only include projects with valid location data
            if data.get('latitude') and data.get('longitude'):
                try:
                    # Convert location data to float and validate
                    lat = float(data.get('latitude', 0))
                    lng = float(data.get('longitude', 0))
                    
                    if -90 <= lat <= 90 and -180 <= lng <= 180:
                        projects.append({
                            'id': doc.id,
                            'name': data.get('name', 'Unnamed Project'),
                            'latitude': lat,
                            'longitude': lng,
                            'status': data.get('status', 'Not Started'),
                            'stage': data.get('current_stage', 'Planning'),
                            'progress': data.get('progress_percentage', 0)
                        })
                except (ValueError, TypeError):
                    # Skip projects with invalid location data
                    continue
        
        if not projects:
            flash('No projects with location data found', 'info')
            
        return render_template('geo_map.html', 
                             projects=projects,
                             user_role=session.get('role', ''))
                             
    except Exception as e:
        app.logger.error(f'Error in geo_map route: {str(e)}')
        flash('Error loading project map', 'error')
        return redirect(url_for('dashboard'))

# Initialize ONNX and Segformer components
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'segformer.onnx')
processor = None
onnx_session = None

def init_onnx():
    """Initialize ONNX components only when needed"""
    global processor, onnx_session
    
    if processor is None or onnx_session is None:
        try:
            from transformers import SegformerImageProcessor
            import onnxruntime as ort
            
            # Initialize the image processor
            processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
            
            # Initialize ONNX Runtime session if model exists
            if os.path.exists(MODEL_PATH):
                onnx_session = ort.InferenceSession(MODEL_PATH)
                return True
            else:
                print(f"Warning: ONNX model not found at {MODEL_PATH}")
                return False
        except Exception as e:
            print(f"Warning: Could not initialize ONNX components: {str(e)}")
            return False
    return True

def segment_buildings_onnx(image_path, building_class_id=2):
    """
    Perform building segmentation using ONNX model.
    Args:
        image_path (str): Path to the input image.
        building_class_id (int): ID of the building class in ADE20K dataset.

    Returns:
        np.ndarray: Binary mask of segmented buildings.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    inputs = processor(images=image, return_tensors="np")

    # Perform inference
    ort_inputs = {"pixel_values": inputs["pixel_values"]}
    outputs = onnx_session.run(None, ort_inputs)
    logits = outputs[0]  # Shape: (1, num_classes, H, W)

    # Get predictions
    predictions = np.argmax(logits[0], axis=0)  # Shape: (H, W)
    
    # Create binary mask
    building_mask = (predictions == building_class_id).astype(np.uint8)
    
    # Resize mask to match original image dimensions
    building_mask = cv2.resize(
        building_mask,
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_NEAREST
    )

    return building_mask

@app.route('/visual_comparison', methods=['GET', 'POST'])
@login_required
def visual_comparison():
    if request.method == 'POST':
        try:
            # Initialize ONNX components only when needed
            if not init_onnx():
                return jsonify({
                    'success': False,
                    'error': 'Image processing components not initialized. Please ensure the ONNX model is properly set up.'
                })

            # Get uploaded images
            prev_image = request.files.get('prev_image')
            curr_image = request.files.get('curr_image')
            
            if not prev_image or not curr_image:
                return jsonify({
                    'success': False,
                    'error': 'Both previous and current images are required.'
                })
            
            if not allowed_file(prev_image.filename) or not allowed_file(curr_image.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type. Only PNG and JPG images are allowed.'
                })

            # Create upload folder if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save images temporarily
            prev_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(prev_image.filename))
            curr_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(curr_image.filename))
            
            try:
                prev_image.save(prev_path)
                curr_image.save(curr_path)

                # Process images using ONNX model
                prev_mask = segment_buildings_onnx(prev_path)
                curr_mask = segment_buildings_onnx(curr_path)

                # Create visualization
                curr_img = cv2.imread(curr_path)
                if curr_img is None:
                    raise ValueError("Failed to load current image")
                
                # Ensure mask dimensions match image dimensions
                curr_h, curr_w = curr_img.shape[:2]
                prev_mask = cv2.resize(prev_mask, (curr_w, curr_h), interpolation=cv2.INTER_NEAREST)
                curr_mask = cv2.resize(curr_mask, (curr_w, curr_h), interpolation=cv2.INTER_NEAREST)
                
                # Create overlay
                overlay = curr_img.copy()
                
                # Highlight changes
                changes = cv2.absdiff(prev_mask, curr_mask)
                changes = changes.astype(bool)
                
                # Apply changes to overlay
                overlay[changes] = [0, 255, 0]  # Green overlay for changes
                
                # Blend the images
                result = cv2.addWeighted(curr_img, 0.6, overlay, 0.4, 0)
                
                # Convert to base64 for display
                _, buffer = cv2.imencode('.png', result)
                result_b64 = base64.b64encode(buffer).decode('utf-8')
                
                return jsonify({
                    'success': True,
                    'result_image': f'data:image/png;base64,{result_b64}'
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Error processing images: {str(e)}'
                })
            finally:
                # Clean up temporary files
                if os.path.exists(prev_path):
                    os.remove(prev_path)
                if os.path.exists(curr_path):
                    os.remove(curr_path)
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Server error: {str(e)}'
            })
            
    return render_template('visual_comparison.html')

if __name__ == '__main__':
    app.run(debug=True)
