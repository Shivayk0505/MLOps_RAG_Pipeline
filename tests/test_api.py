from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_upload_pdf_endpoint_exists():
    assert "/upload_pdf" in [route.path for route in app.routes]
