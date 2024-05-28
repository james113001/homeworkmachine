# Load model directly
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from datasets import Dataset

import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("lschlessinger/bert-finetuned-math-prob-classification")
model = AutoModelForSequenceClassification.from_pretrained("lschlessinger/bert-finetuned-math-prob-classification")
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_df = pd.read_csv('AIMO/train.csv')
# test_df = pd.read_csv('ALLtraincompiled.csv').iloc[:100,:]

# Convert to Hugging Face dataset
test_dataset = Dataset.from_pandas(test_df)


label_mapping = {
    0: "Algebra",
    1: "Counting & Probability",
    2: "Geometry",
    3: "Intermediate Algebra",
    4: "Number Theory",
    5: "Prealgebra",
    6: "Precalculus"
}

def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Move inputs to GPU if available
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()

    predicted_label = label_mapping[predicted_class]
    
    return predicted_label

# Apply the prediction function to the dataset
test_dataset = test_dataset.map(lambda x: {'Type prediction': predict(x['problem'])})
df = pd.DataFrame(test_dataset) 

df