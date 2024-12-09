import os
import firebase_admin
from firebase_admin import credentials, auth, firestore
import pyrebase
from functools import wraps
from flask import session, redirect, url_for, request, flash
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Firebase configuration
config = {
    "apiKey": os.getenv('FIREBASE_API_KEY'),
    "authDomain": os.getenv('FIREBASE_AUTH_DOMAIN'),
    "projectId": os.getenv('FIREBASE_PROJECT_ID'),
    "storageBucket": os.getenv('FIREBASE_STORAGE_BUCKET'),
    "databaseURL": os.getenv('FIREBASE_DATABASE_URL'),
    "messagingSenderId": os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
    "appId": os.getenv('FIREBASE_APP_ID')
}

# Initialize Pyrebase
firebase = pyrebase.initialize_app(config)
pb_auth = firebase.auth()

# Firestore client will be set from app.py
db = None

def set_firestore_client(firestore_client):
    """Set the Firestore client from the main application"""
    global db
    db = firestore_client

def create_user(email, password, role='worker', display_name=None):
    """
    Create a new user with extended profile information
    Default role is 'worker' if not specified
    """
    if role not in ['expert', 'worker']:
        raise ValueError("Invalid role. Must be 'expert' or 'worker'")
    
    try:
        # Create user in Firebase Authentication
        user = pb_auth.create_user_with_email_and_password(email, password)
        
        # Prepare user profile data
        user_profile = {
            'email': email,
            'display_name': display_name or email.split('@')[0],
            'role': role,
            'active_projects': [],
            'status': 'active'
        }
        
        # Store user profile in Firestore
        db.collection('users').document(user['localId']).set(user_profile)
        
        return user['localId']
    
    except Exception as e:
        error = str(e)
        if 'EMAIL_EXISTS' in error:
            raise Exception("Email already exists")
        raise Exception(f"User creation failed: {error}")

def verify_user(email, password):
    """
    Verify user credentials and retrieve user profile
    Follows the specific database structure of the application
    """
    # Explicitly print received credentials
    print("\n" + "=" * 50)
    print("🔍 VERIFICATION INPUT VALIDATION:")
    print(f"Received Email: {email}")
    print(f"Received Password Length: {len(password)}")
    
    # Validate input data
    if not email:
        print("❌ ERROR: Email is empty")
        raise ValueError("Email cannot be empty")
    
    if not password:
        print("❌ ERROR: Password is empty")
        raise ValueError("Password cannot be empty")
    
    try:
        # Step 1: Find user by email in Firestore
        users_ref = db.collection('users')
        
        # Comprehensive query logging
        print("\n🕵️ FIRESTORE QUERY DETAILS:")
        print(f"Querying users collection for email: {email}")
        
        user_query = users_ref.where('email', '==', email).limit(1)
        user_docs = list(user_query.stream())
        
        # Comprehensive debugging for user lookup
        print("\n📊 QUERY RESULTS:")
        print(f"Number of matching users found: {len(user_docs)}")
        
        # If no users found, print all users in the database
        if not user_docs:
            print("❌ NO USERS FOUND WITH THIS EMAIL")
            
            # Fetch and print all users for debugging
            all_users = list(users_ref.stream())
            print("\n🌐 ALL USERS IN DATABASE:")
            for user_doc in all_users:
                user_data = user_doc.to_dict()
                print(f"User ID: {user_doc.id}")
                for key, value in user_data.items():
                    print(f"  {key}: {value}")
            
            raise Exception(f"No account found for {email}")
        
        # Get the user document
        user_doc = user_docs[0]
        user_data = user_doc.to_dict()
        
        # Detailed logging for debugging
        print("\n📋 FULL USER DOCUMENT DETAILS:")
        for key, value in user_data.items():
            print(f"{key}: {value}")
        
        # Validate required fields
        required_fields = ['email', 'role', 'status']
        missing_fields = [field for field in required_fields if field not in user_data]
        
        if missing_fields:
            print(f"❌ MISSING REQUIRED FIELDS: {missing_fields}")
            raise ValueError(f"User document is incomplete. Missing: {missing_fields}")
        
        # Step 2: Validate account status
        if user_data.get('status') != 'active':
            print(f"❌ Account for {email} is not active")
            raise Exception("Account is not active")
        
        # Step 3: Attempt authentication with Pyrebase
        try:
            # Attempt to authenticate with Pyrebase
            print("\n🔐 AUTHENTICATION ATTEMPT:")
            print(f"Attempting to authenticate: {email}")
            pyrebase_user = pb_auth.sign_in_with_email_and_password(email, password)
            print("✅ Pyrebase Authentication Successful")
        except Exception as auth_error:
            # Log authentication error details
            print(f"❌ Pyrebase Authentication Failed: {str(auth_error)}")
            
            # Additional error context
            try:
                error_details = auth_error.args[1] if len(auth_error.args) > 1 else "No additional details"
                print(f"🔍 Authentication Error Details: {error_details}")
            except Exception as debug_error:
                print(f"Could not extract error details: {debug_error}")
            
            # Raise a more informative error
            raise Exception("Invalid login credentials. Please check your email and password.")
        
        # Step 4: Return user profile information
        user_profile = {
            'user_id': user_doc.id,  # Use Firestore document ID
            'email': email,
            'role': user_data.get('role', 'unknown'),
            'display_name': user_data.get('display_name', email.split('@')[0]),
            'status': user_data.get('status', 'active')
        }
        
        print("\n👤 FINAL USER PROFILE:")
        for key, value in user_profile.items():
            print(f"{key}: {value}")
        
        return user_profile
    
    except Exception as e:
        # Comprehensive error logging
        print("\n❌ VERIFICATION ERROR:")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        raise

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(allowed_roles):
    """Decorator to require specific roles for routes"""
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            user_role = session.get('role')
            if not user_role or user_role not in allowed_roles:
                flash('You do not have permission to access this page', 'error')
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator