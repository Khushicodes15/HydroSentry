import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database.database import Base, get_db
from main import app

test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    # Setup test database
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=test_engine)
    yield
    # Teardown test database
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.pop(get_db, None)



@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_register_user_success(client):
    payload = {
        "username": "hydro_user",
        "email": "user@hydrosentry.org",
        "password": "Password123!",
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "hydro_user"
    assert data["email"] == "user@hydrosentry.org"
    assert "id" in data
    assert "created_at" in data
    assert "password" not in data
    assert "password_hash" not in data


def test_register_duplicate_username_rejected(client):
    payload = {
        "username": "dupe_user",
        "email": "dupe1@hydrosentry.org",
        "password": "Password123!",
    }
    res1 = client.post("/api/auth/register", json=payload)
    assert res1.status_code == 201

    payload_duplicate = {
        "username": "dupe_user",
        "email": "dupe2@hydrosentry.org",
        "password": "DifferentPassword123!",
    }
    res2 = client.post("/api/auth/register", json=payload_duplicate)
    assert res2.status_code == 400
    assert "Username already registered" in res2.json()["detail"]


def test_register_duplicate_email_rejected(client):
    payload = {
        "username": "email_user1",
        "email": "same@hydrosentry.org",
        "password": "Password123!",
    }
    res1 = client.post("/api/auth/register", json=payload)
    assert res1.status_code == 201

    payload_duplicate = {
        "username": "email_user2",
        "email": "same@hydrosentry.org",
        "password": "Password123!",
    }
    res2 = client.post("/api/auth/register", json=payload_duplicate)
    assert res2.status_code == 400
    assert "Email already registered" in res2.json()["detail"]


def test_login_success(client):
    # Register first
    client.post(
        "/api/auth/register",
        json={"username": "login_user", "email": "login@hydrosentry.org", "password": "SecretPassword1"},
    )
    # Login
    response = client.post(
        "/api/auth/login",
        json={"username": "login_user", "password": "SecretPassword1"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"].lower() == "bearer"
    assert len(data["access_token"]) > 20


def test_login_wrong_password_rejected(client):
    client.post(
        "/api/auth/register",
        json={"username": "wrong_pwd_user", "email": "wp@hydrosentry.org", "password": "RightPassword1"},
    )
    response = client.post(
        "/api/auth/login",
        json={"username": "wrong_pwd_user", "password": "IncorrectPassword"},
    )
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]


def test_login_nonexistent_user_rejected(client):
    response = client.post(
        "/api/auth/login",
        json={"username": "nobody", "password": "SomePassword"},
    )
    assert response.status_code == 401


def test_auth_me_endpoint(client):
    # Register and login
    client.post(
        "/api/auth/register",
        json={"username": "me_user", "email": "me@hydrosentry.org", "password": "MyPassword123"},
    )
    login_res = client.post(
        "/api/auth/login",
        json={"username": "me_user", "password": "MyPassword123"},
    )
    token = login_res.json()["access_token"]

    # Call /api/auth/me with Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200
    user_data = me_res.json()
    assert user_data["username"] == "me_user"
    assert user_data["email"] == "me@hydrosentry.org"

    # Call /api/auth/me without token -> rejected
    unauth_res = client.get("/api/auth/me")
    assert unauth_res.status_code == 401

    # Call /api/auth/me with invalid token -> rejected
    invalid_res = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid.fake.token"})
    assert invalid_res.status_code == 401


def test_oauth2_token_form_login_success(client):
    """Verify Swagger UI OAuth2 form-encoded login flow at /api/auth/token."""
    client.post(
        "/api/auth/register",
        json={"username": "form_user", "email": "form@hydrosentry.org", "password": "FormPassword123!"},
    )
    # Standard form data sent by Swagger UI
    form_data = {
        "username": "form_user",
        "password": "FormPassword123!",
        "grant_type": "password",
        "client_id": "",
        "client_secret": "",
    }
    response = client.post("/api/auth/token", data=form_data)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"].lower() == "bearer"

    # Verify token works for /api/auth/me
    headers = {"Authorization": f"Bearer {data['access_token']}"}
    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["username"] == "form_user"


def test_oauth2_token_form_wrong_password_rejected(client):
    client.post(
        "/api/auth/register",
        json={"username": "form_wrong_user", "email": "form_wrong@hydrosentry.org", "password": "CorrectPassword1"},
    )
    response = client.post(
        "/api/auth/token",
        data={"username": "form_wrong_user", "password": "BadPassword"},
    )
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]


def test_swagger_openapi_oauth2_token_url(client):
    """Verify Swagger UI OpenAPI specification defines tokenUrl pointing to /api/auth/token."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    sec_schemes = spec.get("components", {}).get("securitySchemes", {})
    assert "OAuth2PasswordBearer" in sec_schemes
    flow = sec_schemes["OAuth2PasswordBearer"].get("flows", {}).get("password", {})
    assert flow.get("tokenUrl") == "/api/auth/token"

