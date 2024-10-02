import pytest
from app import app  # Import your Flask app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home(client):
    """Test the home route"""
    response = client.get('/')
    assert response.status_code == 200

def test_predict(client):
    """Test the /predict endpoint"""
    response = client.post('/predict', json={'symbol': 'AAPL'})
    assert response.status_code == 200
    assert 'predictions' in response.get_json()
