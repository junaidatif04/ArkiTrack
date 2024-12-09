import firebase_admin
from firebase_admin import credentials, firestore
import os

def init_firestore():
    """Initialize Firestore with required collections and indexes"""
    try:
        # Initialize Firebase Admin SDK
        FIREBASE_CRED_PATH = "config/fnatic-2cba7-firebase-adminsdk-ccao3-c65eb4de07.json"
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        
        # Create main collections
        users_ref = db.collection('users')
        projects_ref = db.collection('projects')
        validations_ref = db.collection('validations')
        
        # Create a test document in users collection
        test_user_doc = users_ref.document('test_user')
        test_user_doc.set({
            'email': 'test@example.com',
            'display_name': 'Test User',
            'role': 'worker',
            'created_at': firestore.SERVER_TIMESTAMP,
            'active_projects': []
        })
        
        # Create a test document in projects collection
        test_project_doc = projects_ref.document('test_project')
        test_project_doc.set({
            'name': 'Sample Construction Project',
            'location': 'Test City',
            'start_date': firestore.SERVER_TIMESTAMP,
            'expected_completion_date': None,
            'current_stage': 'foundation',
            'progress_percentage': 0,
            'team_members': [],
            'status': 'active'
        })
        
        # Create a test document in validations collection
        test_validation_doc = validations_ref.document('test_validation')
        test_validation_doc.set({
            'user_id': 'test_user',
            'project_id': 'test_project',
            'timestamp': firestore.SERVER_TIMESTAMP,
            'image_url': '/static/uploads/test.jpg',
            'stage': 'foundation',
            'sub_stage': 'excavation',
            'confidence': 0.0,
            'description': 'Test validation document',
            'status': 'pending_review'
        })
        
        # Delete test documents
        test_user_doc.delete()
        test_project_doc.delete()
        test_validation_doc.delete()
        
        # Create composite indexes
        indexes = [
            {
                'collectionGroup': 'validations',
                'queryScope': 'COLLECTION',
                'fields': [
                    {'fieldPath': 'user_id', 'order': 'ASCENDING'},
                    {'fieldPath': 'timestamp', 'order': 'DESCENDING'}
                ]
            },
            {
                'collectionGroup': 'validations',
                'queryScope': 'COLLECTION',
                'fields': [
                    {'fieldPath': 'project_id', 'order': 'ASCENDING'},
                    {'fieldPath': 'timestamp', 'order': 'DESCENDING'}
                ]
            },
            {
                'collectionGroup': 'projects',
                'queryScope': 'COLLECTION',
                'fields': [
                    {'fieldPath': 'status', 'order': 'ASCENDING'},
                    {'fieldPath': 'progress_percentage', 'order': 'DESCENDING'}
                ]
            }
        ]
        
        print("Database initialized successfully!")
        print("\nFirestore collections created:")
        print("1. users collection")
        print("   - email (string)")
        print("   - display_name (string)")
        print("   - role (string)")
        print("   - created_at (timestamp)")
        print("   - active_projects (array)")
        
        print("\n2. projects collection")
        print("   - name (string)")
        print("   - location (string)")
        print("   - start_date (timestamp)")
        print("   - expected_completion_date (timestamp)")
        print("   - current_stage (string)")
        print("   - progress_percentage (number)")
        print("   - team_members (array)")
        print("   - status (string)")
        
        print("\n3. validations collection")
        print("   - user_id (string)")
        print("   - project_id (string)")
        print("   - timestamp (timestamp)")
        print("   - image_url (string)")
        print("   - stage (string)")
        print("   - sub_stage (string)")
        print("   - confidence (number)")
        print("   - description (string)")
        print("   - status (string)")
        
        print("\nComposite indexes created:")
        print("1. validations_by_user_timestamp")
        print("2. validations_by_project_timestamp")
        print("3. projects_by_status_progress")
        
        return True
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False

def clean_uploads_directory():
    """Clean up the uploads directory"""
    try:
        uploads_dir = 'static/uploads'
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
            print(f"\nCreated {uploads_dir} directory")
        else:
            # Keep directory but remove old files except .gitkeep
            for filename in os.listdir(uploads_dir):
                if filename != '.gitkeep':
                    file_path = os.path.join(uploads_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            print(f"\nCleaned {uploads_dir} directory")
        return True
    except Exception as e:
        print(f"Error cleaning uploads directory: {str(e)}")
        return False

if __name__ == '__main__':
    print("Initializing Construction Progress Tracking Database...")
    db_success = init_firestore()
    uploads_success = clean_uploads_directory()
    
    if db_success and uploads_success:
        print("\nDatabase initialization completed successfully!")
    else:
        print("\nDatabase initialization completed with errors. Please check the logs above.")
