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


#import Data to predict on, change file as needed
test_df = pd.read_csv('ALLtestcompiled.csv')
# Convert 'Type' column to categorical if not already
test_df['Type'] = test_df['Type'].astype('category')

# Map the labels to integers
test_df['label'] = test_df['Type'].cat.codes

test_df = test_df[['label','Question']]
# Convert to Hugging Face dataset
test_dataset = Dataset.from_pandas(test_df)

# Function to tokenize data with truncation
def tokenize_data(examples):
    return tokenizer(examples['Question'], padding='max_length', truncation=True, max_length=512)

# Tokenize the datasets
tokenized_test_dataset = test_dataset.map(tokenize_data, batched=True, num_proc=4)
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# classifier = pipeline("text-classification", model="lschlessinger/bert-finetuned-math-prob-classification")
# classifier(tokenized_datasets['test']['New_Question'])

# Prepare DataLoader
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=16)

# Function to evaluate the model with a progress bar
def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions, true_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels

# Run evaluation
preds, labels = evaluate_model(model, test_dataloader)

# # Calculate evaluation metrics
# accuracy = accuracy_score(labels, preds)
# precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")