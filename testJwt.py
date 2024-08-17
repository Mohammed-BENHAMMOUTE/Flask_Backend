import os
import jwt
from datetime import datetime, timedelta


# Function to get the JWT secret key
def get_jwt_secret_key():
    # Try to get the key from environment variable, using the correct name with a dash
    key ='9faa372517ac1d389758d3750fc07acf00f542277f26fec1ce4593e93f64e338'

    if key is None:
        print("Error: JWT-SECRET-KEY not found in environment variables.")
        print("Please make sure you've set the environment variable correctly.")
        return None

    return key.strip()  # Remove any leading/trailing whitespace


# Get the JWT secret key
JWT_SECRET_KEY = get_jwt_secret_key()

if JWT_SECRET_KEY is None:
    print("Exiting due to missing JWT secret key.")
    exit(1)

# Create the payload
payload = {
    "exp": datetime.utcnow() + timedelta(hours=1),  # Token expires in 1 hour
    "iat": datetime.utcnow(),
    "claims": {
        "role": "MEDECIN",
        "id": "test_doctor_id"
    }
}

try:
    # Generate the token
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS512")
    print(f"Your JWT token: {token}")
except TypeError as e:
    print(f"Error: {e}")
    print("Make sure JWT-SECRET-KEY is a non-empty string.")
    print(f"Current value of JWT-SECRET-KEY: '{JWT_SECRET_KEY}'")
except Exception as e:
    print(f"An unexpected error occurred: {e}")