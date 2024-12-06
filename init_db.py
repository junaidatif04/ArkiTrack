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
        
        # Create validations collection if it doesn't exist
        validations_ref = db.collection('validations')
        
        # Create a test document to ensure collection exists
        test_doc = validations_ref.document('test_init')
        test_doc.set({
            'timestamp': firestore.SERVER_TIMESTAMP,
            'image_url': '/static/uploads/test.jpg',
            'stage': 'test',
            'sub_stage': 'test',
            'confidence': 0.0,
            'description': 'Test document for initialization'
        })
        
        # Delete the test document
        test_doc.delete()
        
        # Create composite index for validations collection
        # This index will support querying validations by timestamp
        index = {
            'collectionGroup': 'validations',
            'queryScope': 'COLLECTION',
            'fields': [
                {
                    'fieldPath': 'timestamp',
                    'order': 'DESCENDING'
                },
                {
                    'fieldPath': 'stage',
                    'order': 'ASCENDING'
                }
            ]
        }
        
        print("Database initialized successfully!")
        print("\nFirestore collections and indexes created:")
        print("1. validations collection")
        print("   - timestamp (timestamp)")
        print("   - image_url (string)")
        print("   - stage (string)")
        print("   - sub_stage (string)")
        print("   - confidence (number)")
        print("   - description (string, optional)")
        print("\nComposite indexes:")
        print("1. validations_by_timestamp_stage")
        print("   - timestamp (DESCENDING)")
        print("   - stage (ASCENDING)")
        
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
