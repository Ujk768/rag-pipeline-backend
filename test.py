import psycopg2
import os
from dotenv import load_dotenv

# 👇 FORCE correct path to .env
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

url = os.getenv("DATA_BASE_URL")

print("Connecting...")
print("DATA_BASE_URL:", url)

if url is None:
    raise Exception("❌ DATA_BASE_URL not loaded. Check .env file location.")

conn = psycopg2.connect(
    url,
    sslmode="require",
    connect_timeout=5
)

print("✅ Connected!")