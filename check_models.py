# For me the working model is : 'gemini-flash-latest'
# but for you, you need to check the available model
# so let's just run this snippet

import google.generativeai as genai

api_key = "YOUR_API_KEY_HERE"
genai.configure(api_key=api_key)

print("Available Models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)