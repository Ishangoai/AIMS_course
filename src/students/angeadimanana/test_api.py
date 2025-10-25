from google import genai
from google.genai import types
import os

api_key= os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key="AIzaSyConloRzNoTi14lPOp9fr7G_95VUxu19EA")
response = client.models.generate_content(
    model='gemini-2.5-flash', contents='What is the capital of France?'
)
print(response.text)