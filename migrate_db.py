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
        
        # Migrate Users Collection
        users_ref = db.collection('users')
        existing_users = db.collection('auth_users').stream()
        
        migrated_users = 0
        for user_doc in existing_users:
            user_data = user_doc.to_dict()
            new_user_doc = users_ref.document(user_doc.id)
            new_user_data = {
                'email': user_data.get('email', ''),
                'display_name': user_data.get('display_name', ''),
                'role': user_data.get('role', 'worker'),
                'created_at': user_data.get('created_at', datetime.now()),
                'active_projects': user_data.get('active_projects', [])
            }
            new_user_doc.set(new_user_data)
            migrated_users += 1
        
        # Migrate Projects Collection
        projects_ref = db.collection('projects')
        existing_projects = db.collection('construction_projects').stream()
        
        migrated_projects = 0
        for project_doc in existing_projects:
            project_data = project_doc.to_dict()
            new_project_doc = projects_ref.document(project_doc.id)
            new_project_data = {
                'name': project_data.get('project_name', 'Unnamed Project'),
                'location': project_data.get('location', ''),
                'start_date': project_data.get('start_date', datetime.now()),
                'expected_completion_date': project_data.get('expected_completion_date', None),
                'current_stage': project_data.get('current_stage', 'foundation'),
                'progress_percentage': project_data.get('progress_percentage', 0),
                'team_members': project_data.get('team_members', []),
                'status': project_data.get('status', 'active')
            }
            new_project_doc.set(new_project_data)
            migrated_projects += 1
        
        # Migrate Validations Collection
        validations_ref = db.collection('validations')
        existing_validations = db.collection('image_validations').stream()
        
        migrated_validations = 0
        skipped_validations = 0
        
        for validation_doc in existing_validations:
            data = validation_doc.to_dict()
            
            # Skip if document is already in new format
            if all(key in data for key in ['user_id', 'project_id', 'status']):
                skipped_validations += 1
                continue
            
            # Create new validation document
            new_validation_data = {
                'user_id': data.get('user_id', 'unknown_user'),
                'project_id': data.get('project_id', 'unknown_project'),
                'timestamp': data.get('timestamp', datetime.now()),
                'image_url': f"/static/uploads/{os.path.basename(data.get('file_path', ''))}",
                'stage': data.get('selected_stage', data.get('predicted_stage', '')),
                'sub_stage': data.get('predicted_class', ''),
                'confidence': data.get('stage_confidence', 0.0),
                'description': data.get('description', ''),
                'status': 'pending_review'
            }
            
            # Create new validation document
            new_validation_doc = validations_ref.document(validation_doc.id)
            new_validation_doc.set(new_validation_data)
            migrated_validations += 1
        
        print(f"\nMigration completed:")
        print(f"- {migrated_users} users migrated")
        print(f"- {migrated_projects} projects migrated")
        print(f"- {migrated_validations} validations migrated")
        print(f"- {skipped_validations} validations already in new format")
        
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
