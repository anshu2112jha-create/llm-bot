# %%
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


model_path = r"C:\Users\...\Hugging Face\modelsall-MiniLM-L6-v2"

# Load tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Mean pooling to convert token embeddings to sentence embedding
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

# Function to convert text into vector
def get_embedding(text):
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**encoded)
    return mean_pooling(output, encoded['attention_mask'])[0].numpy()


# %%
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_path = r"C:\Users\...\Hugging Face\modelsall-MiniLM-L6-v2"

# Load tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

# Embedding function
def get_embedding(text):
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**encoded)
    return mean_pooling(output, encoded['attention_mask'])[0].numpy()

# Define equipment and safety initiative descriptions
equipment_dict = {
   
    "Winter Traction": "Prevents slips and falls caused by snow or ice.",
    "Powered Lift Table": "Used to lift heavy items and prevent back injuries.",
    "Other": "Used when no equipment clearly applies."
}


# Optional: Keyword boost
boost_keywords = {
    "Winter Traction": [
        "ice", "snow", "slip", ...
    ],
    "Powered Lift Table": [
        "lift", "lifting", ...
    ],
    "DamoGuard Expansion": [
        "collision", ...
    ],
    "Wide Aisle Blockers": [
        "unauthorized access", ...
    ],
    "Dock Lock": [
        "dock", "trailer", ...
    ],
    "High Flow Fire Extinguisher": [
        "fire", "flame",...
    ],
    "Order Picker Platforms": [
        "order picker", "platform", ...
    ],
    "Better Ladders": [
        "ladder", "climb", ...
    ],
    "Hazmat Cage Rebuilds": [
        "hazmat", "chemical", ...
    ],
    "Customer Awareness Signage": [
        "customer", "sign", ...
    ],
    "Animal / Pest Control Measures": [
        "animal", "rodent", ...
    ],
    "PPE / Associate Safety Awareness": [
        "awareness", "training",...
    ],
    "Traffic Measure": [
        "vehicle", "forklift", ...
    ],
    "Other": []
}

# Precompute embeddings for equipment
equipment_embeddings = {k: get_embedding(v) for k, v in equipment_dict.items()}

# Matching function
def map_incident_to_equipment(text, threshold=0.35, top_n=1):
    incident_embedding = get_embedding(text)
    scores = {}

    for equip, emb in equipment_embeddings.items():
        sim = cosine_similarity([incident_embedding], [emb])[0][0]

        # Keyword bonus
        keywords = boost_keywords.get(equip, [])
        for word in keywords:
            if word in text.lower():
                sim += 0.05  # boost score

        scores[equip] = sim

    # Pick top results above threshold
    sorted_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    final = [k for k, v in sorted_matches[:top_n] if v > threshold]

    return ", ".join(final) if final else "Other"


# %%
df = pd.read_csv(r"C:\Users...\Input_2024.csv")
df["Preventable_By"] = df["Details"].apply(map_incident_to_equipment)


# %%
output_path = r"C:\Users\,,,\Output_2024.csv"
df.to_csv(output_path, index=False)



