from transformers import pipeline

prompt = "Write a small poem about the ocean"

# Create generator
generator = pipeline("text-generation", model="gpt2")

# Generate text
result = generator(prompt)

# Print the generated poem
print("=== Generated Poem ===\n")
print(result[0]['generated_text'])
