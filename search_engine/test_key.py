from haci_provider import HaciProvider
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("HACIPROVIDER_API_KEY")
print(f"Testing API Key: {api_key[:10]}...{api_key[-5:]}")

client = HaciProvider(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "user", "content": "Hi"}
        ],
    )
    print("✅ API Key is VALID!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print("❌ API Key is INVALID or EXPIRED!")
    print(f"Error: {e}")
