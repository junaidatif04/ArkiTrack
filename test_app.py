import pytest
from app import app
import os

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test that home page loads correctly"""
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'Construction Stage Classification' in rv.data

def test_validate_no_image(client):
    """Test validation endpoint without image"""
    rv = client.post('/validate')
    assert rv.status_code == 400
    assert b'No image uploaded' in rv.data

def test_validate_invalid_file(client):
    """Test validation with invalid file type"""
    data = {
        'image': (b'dummy data', 'test.txt'),
        'stage': 'foundation'
    }
    rv = client.post('/validate', data=data, content_type='multipart/form-data')
    assert rv.status_code == 400
    assert b'Invalid file type' in rv.data
