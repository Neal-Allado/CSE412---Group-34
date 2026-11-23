import sys
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Define series of fallback models
models = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]

# Define model context (WIP, needs to be refined)
instruction = "End your response with this phrase: 'Is this helpful?'"
config = types.GenerateContentConfig(system_instruction=instruction)

# API key should normally be an environment variable, but is hard-coded for ease of use and collaborative development. 
# Initialize client
client = genai.Client(api_key="INSERT API KEY")

# Get user prompt
prompt = input("Enter AI Prompt: ")

# Generate response
for model_name in models:
    try:
        response = client.models.generate_content(model=model_name, contents=prompt, config=config)
        print(f"{response.text}")
        break
    except APIError as e:
        if e.code == 503:
            print(f"{model_name} unavailable, trying next model...")
        continue