# rainbow_explanation.py
from transformers import pipeline

# Prompt for the model
prompt = "Explain how rainbows are formed"

# Create a text-generation pipeline with a small local model
# 'gpt2' is light enough for most CPUs; you can swap with another model if desired
generator = pipeline("text-generation", model="gpt2")

# Generate output
result = generator(prompt, max_new_tokens=100, temperature=0.7, do_sample=True)

# Print response
print("=== Model Response ===\n")
print(result[0]['generated_text'])
