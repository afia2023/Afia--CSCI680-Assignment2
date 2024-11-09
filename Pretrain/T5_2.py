import torch
import logging
import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq ,TrainerCallback
from datasets import Dataset
from nltk.stem import WordNetLemmatizer
from evaluate import load
import nltk
import json
import shutil
from torch.cuda.amp import GradScaler, autocast
from transformers import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from datasets import load_metric
import gc
import torch.nn as nn
#from your_callbacks_module import PrintMetricsCallback, SaveBestCheckpointsCallback

# Ideally, set this environment variable outside your Python script, such as in your shell
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # This will map CUDA device 0 to your system's GPU 3

# Initialize your device after setting CUDA_VISIBLE_DEVICES
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # cuda:0 now refers to the GPU 3 on your system
#print(f"Using device: {device}")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Function to clear GPU cache
def clear_cuda_cache():
    torch.cuda.empty_cache()
    print("Cleared CUDA cache")

# Function to load and prepare the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def tokenize_data(dataframe, tokenizer):
    def tokenize_function(examples):
        model_inputs = tokenizer(examples['masked_method_code'], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        labels = tokenizer(examples['method_code'], max_length=128, padding="max_length", truncation=True, return_tensors="pt")

        # Set labels for non-masked tokens to -100 so they don't affect loss calculation
        labels[labels != tokenizer.mask_token_id] = -100
        return {
            'input_ids': model_inputs['input_ids'], 
            'attention_mask': model_inputs['attention_mask'], 
            'labels': labels['input_ids']
        }
    
    # Convert the pandas DataFrame to a Hugging Face dataset
    dataset = Dataset.from_pandas(dataframe)
    
    # Apply the tokenize_function across the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Remove any unnecessary columns and set the format for PyTorch
    tokenized_dataset = tokenized_dataset.remove_columns(
        [column for column in tokenized_dataset.column_names if column not in ['input_ids', 'attention_mask', 'labels']]
    )
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return tokenized_dataset

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
special_tokens_dict = {'additional_special_tokens': ['<TAB>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# Load the model and resize token embeddings
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
model.resize_token_embeddings(len(tokenizer))  # Adjust the model's embedding size to the new tokenizer
#model = model.to(device)
# Automatically use all available GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model.to("cuda") 

gc.collect()
torch.cuda.empty_cache()

# Load and prepare data
df = load_dataset('Big_Dataset1.json')
# First split to separate out the test set
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)  # Reserve 10% of the data as test set

# Second split to divide the remaining data into training and validation sets
train_df, val_df = train_test_split(train_val_df, test_size=0.11, random_state=42)  # 11% of 90% is about 10% of the original data

# Tokenize datasets
train_dataset = tokenize_data(train_df, tokenizer)
val_dataset = tokenize_data(val_df, tokenizer)
test_dataset = tokenize_data(test_df, tokenizer)

# Assuming test_dataset is a Hugging Face Dataset object after tokenization
test_dataset.save_to_disk('/home/afarjana/Code_Summarization/Assignment2/test_dataset')

gc.collect()
torch.cuda.empty_cache()

# Example of setting up a PyTorch DataLoader (optional here but useful for large datasets)
batch_size = 2
# Assuming train_dataset, val_dataset, and test_dataset are already defined and are instances of 'datasets.Dataset'
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

# Function to evaluate the model on the validation set
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss.mean()
            total_loss += loss.item() * batch['input_ids'].size(0)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == batch['labels']).sum().item()
            total_predictions += predictions.numel()

    avg_loss = total_loss / total_predictions
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


accumulation_steps=4
# Setup optimizer and scaler
optimizer = AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()
best_accuracy = 0.0  # Track the best validation accuracy

# Training loop
for epoch in range(3):  # Example for 3 epochs
    model.train()
    total_train_loss = 0  # Initialize the loss accumulation
    for i, batch in enumerate(train_loader):
        batch = {k: v.to("cuda") for k, v in batch.items()}

        with autocast():
            outputs = model(**batch)
            loss = outputs.loss.mean()

        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_train_loss += loss.item()  # Accumulate loss over the epoch

        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    avg_train_loss = total_train_loss / len(train_loader)  # Calculate average loss for the epoch
    print(f"Epoch {epoch}, Training Loss: {avg_train_loss:.2f}")  # Report training loss for the epoch

    eval_loss, eval_accuracy = evaluate(model, val_loader, "cuda")
    print(f"After Epoch {epoch + 1}: Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")

    # Save model if accuracy improves
    if eval_accuracy > best_accuracy:
        best_accuracy = eval_accuracy
        # Access the underlying model from DataParallel wrapper before saving
        if isinstance(model, torch.nn.DataParallel):
            model.module.save_pretrained('./best_model_1')
        else:
            model.save_pretrained('./best_model_1')
        print(f"New best model saved with accuracy: {best_accuracy:.4f}")

    torch.cuda.empty_cache()
    gc.collect()

print("Training complete.")