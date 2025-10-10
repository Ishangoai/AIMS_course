import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel("models/gemini-2.0-flash-lite")
response = model.generate_content("What is the capital of France?")
print(response.text)