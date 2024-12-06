import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import datetime

def migrate_data():
    """Migrate existing validation records to the new schema"""
    try:
        # Initialize Firebase Admin SDK if not already initialized
        FIREBASE_CRED_PATH = "config/fnatic-2cba7-firebase-adminsdk-ccao3-c65eb4de07.json"
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        validations_ref = db.collection('validations')
        
        # Get all existing validation documents
        docs = validations_ref.stream()
        
        migrated_count = 0
        skipped_count = 0
        
        for doc in docs:
            data = doc.to_dict()
            
            # Skip if document is already in new format
            if all(key in data for key in ['stage', 'sub_stage', 'image_url']):
                skipped_count += 1
                continue
                
            # Create new document data
            new_data = {
                'timestamp': data.get('timestamp', datetime.now()),
                'image_url': f"/static/uploads/{os.path.basename(data.get('file_path', ''))}",
                'stage': data.get('selected_stage', data.get('predicted_stage', '')),
                'sub_stage': data.get('predicted_class', ''),
                'confidence': data.get('stage_confidence', 0.0),
                'description': data.get('description', None)
            }
            
            # Update the document
            doc.reference.set(new_data)
            migrated_count += 1
            
        print(f"\nMigration completed:")
        print(f"- {migrated_count} documents migrated")
        print(f"- {skipped_count} documents already in new format")
        return True
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        return False

if __name__ == '__main__':
    print("Starting database migration...")
    if migrate_data():
        print("\nMigration completed successfully!")
    else:
        print("\nMigration completed with errors. Please check the logs above.")
