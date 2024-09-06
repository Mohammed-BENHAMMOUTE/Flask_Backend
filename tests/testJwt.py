import os
import jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Function to get the JWT secret key
def get_jwt_secret_key():
    key = os.getenv("JWT_SECRET_KEY")
    if key is None or not isinstance(key, str) or not key.strip():
        raise ValueError("JWT_SECRET_KEY must be a non-empty string")
    return key.strip()

# Get the JWT secret key
JWT_SECRET_KEY = get_jwt_secret_key()

# Create the payload
payload = {
    "exp": datetime.utcnow() + timedelta(hours=10),  # Token expires in 1 hour
    "iat": datetime.utcnow(),
    "claims": {
        "role": "ROLE_MEDECIN",
        "id": "test_doctor_id"
    }
}

try:
    # Generate the token
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS512")
    print(f"Your JWT token: {token}")
except Exception as e:
    print(f"An error occurred: {e}")