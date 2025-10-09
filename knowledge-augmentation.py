from transformers import AutoTokenizer, AutoModel

# Load the MedGemma model (this is a placeholder, replace with the actual model)
model_name = "huggingface/medgemma"  # Example model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example biomedical text (could be extracted from PubMed)
text = "Aspirin is used to reduce fever, pain, and inflammation."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Get the embeddings from the model
outputs = model(**inputs)
embeddings = outputs.last_hidden_state

# Print embeddings (can be used for further processing)
print(embeddings)
