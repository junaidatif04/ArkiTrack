ArkiTrack

Overview

ArkiTrack is a web application designed for monitoring the construction progress of buildings using deep learning convolutional neural network (CNN) models. It analyzes images from various construction phases, such as foundation, superstructure, facade, interior, and finishing, to track and visualize progress. The application leverages pre-trained CNN architectures like MobileNet, Inception, and VGG16 for image classification and analysis.

Key components include:





User authentication system



Dashboards for workers and experts



Project creation and management



Image upload and visual comparison tools



Geographic mapping of construction sites



Database initialization and migration scripts



Pre-built models stored in .keras format

Installation





Clone the repository:

git clone https://github.com/junaidatif04/ArkiTrack.git
cd ArkiTrack



Set up a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install dependencies: Assuming typical requirements for a Flask and TensorFlow/Keras app (update requirements.txt as needed):

pip install -r requirements.txt

Common dependencies might include: Flask, TensorFlow, Keras, SQLAlchemy, Firebase Admin, etc.



Configure environment variables:





Create or update .env with necessary keys (e.g., database URL, Firebase credentials).



Initialize the database:

python init_db.py

Run migrations if needed:

python migrate_db.py

Usage





Run the application:

python app.py



Access the web interface at http://localhost:5000 (or the configured port).



Sign up or log in.





Workers can upload images via their dashboard.



Experts can review progress, view visual comparisons, and monitor geo-locations.



To build or update models (if needed):

python build_models.py

This script handles CNN model creation or training for construction phase detection.

Directory Structure





config/: Contains configuration files, such as Firebase admin SDK JSON.



models/: Stores pre-trained .keras models for different construction phases (e.g., Foundation_mobile.keras, Superstructure_vgg16.keras).



static/: Static assets including CSS, JS, and uploads directory for images.



templates/: HTML templates for web pages (e.g., login.html, worker_dashboard.html, visual_comparison.html).



app.py: Main Flask application script.



auth.py: Handles user authentication.



build_models.py: Script for building and saving CNN models.



init_db.py & migrate_db.py: Database setup and migration scripts.



requirements.txt: List of Python dependencies.
