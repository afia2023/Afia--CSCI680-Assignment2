import torch
import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq ,TrainerCallback
from datasets import Dataset
from evaluate import load
import json
import shutil
import gc
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Initialize ROUGE metric
rouge = load('rouge', trust_remote_code=True)


# GPU device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleared CUDA cache and Python garbage")

# Function to save preprocessed data as a JSON array
def save_preprocessed_data(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Tokenize the data using the tokenizer
def tokenize_data(dataframe, tokenizer):

    dataframe = dataframe.reset_index(drop=True)

    def tokenize_function(examples):
        model_inputs = tokenizer(examples['input_method'], max_length=512, padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target_block'], max_length=512, padding="max_length", truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    tokenized_dataset = Dataset.from_pandas(dataframe).map(tokenize_function, batched=True, batch_size=5000)
    tokenized_dataset = tokenized_dataset.remove_columns(['input_method', 'target_block'])

    return tokenized_dataset



tokenizer = AutoTokenizer.from_pretrained("best_model_1")
model = AutoModelForSeq2SeqLM.from_pretrained("best_model_1")
model = model.cuda()  # Move model to default GPU
# model = nn.DataParallel(model)

# File paths
input_file = 'Finetune_processed_dataset.json'

# Load the preprocessed dataset from the JSON file
preprocessed_df = pd.read_json(input_file)
print("Data loaded and processed. First few rows:")
print(preprocessed_df.head(1))
print("Length of the entire preprocessed dataset:", len(preprocessed_df))
print("Shape of the entire preprocessed dataset:", preprocessed_df.shape)

# Split the dataset into training, validation, and test sets
train_df, temp_df = train_test_split(preprocessed_df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.20, random_state=42)

# Tokenize the datasets
train_dataset = tokenize_data(train_df, tokenizer)
val_dataset = tokenize_data(val_df, tokenizer)
test_dataset = tokenize_data(test_df, tokenizer)

print("Sample data points from train_dataset:")
for i in range(min(3, len(train_dataset))):  # Print 3 samples or fewer if dataset is small
    sample = train_dataset[i]
    print(f"Data point {i + 1}:")
    for key, value in sample.items():
        # If the data is tokenized text, print the shape if it's a tensor, else print the value itself
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {value}")

class SaveBestCheckpointsCallback(TrainerCallback):
    """A callback that saves the best N checkpoints based on a specific metric."""
    def __init__(self, save_path, max_save=2, metric="eval_loss"):
        self.best_scores = []
        self.checkpoint_paths = []
        self.save_path = save_path
        self.max_save = max_save
        self.metric = metric

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_score = metrics.get(self.metric)
        current_checkpoint_path = f"{self.save_path}/checkpoint-{state.global_step}"
        
        if len(self.best_scores) < self.max_save or current_score < max(self.best_scores):
            model.save_pretrained(current_checkpoint_path)
            self.best_scores.append(current_score)
            self.checkpoint_paths.append(current_checkpoint_path)
            
            if len(self.best_scores) > self.max_save:
                worst_idx = self.best_scores.index(max(self.best_scores))
                worst_checkpoint_path = self.checkpoint_paths[worst_idx]
                
                # Remove the worst checkpoint
                del self.best_scores[worst_idx]
                del self.checkpoint_paths[worst_idx]
                shutil.rmtree(worst_checkpoint_path)  # Delete the directory

def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleared CUDA cache and Python garbage")

# Configure the training arguments with smaller batch sizes and more gradient accumulation
training_args = Seq2SeqTrainingArguments(
    output_dir='./results1',
    eval_strategy='steps',
    eval_steps=50,  # Evaluate every 50 steps
    save_strategy='steps',
    save_steps=50,  # Save every 50 steps
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Reduced batch size
    gradient_accumulation_steps=8,  # Increased accumulation steps
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_runtime',
    logging_dir='./logs',
    fp16=True, # Mixed precision training
    remove_unused_columns=False
)

# Create the Trainer object
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, max_length=512),  # This should handle padding automatically
    callbacks=[SaveBestCheckpointsCallback('./results1', max_save=2)]
)

# Train the model
# Add the optimization memory management technique
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere GPUs to speed up training
torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuning for performance
torch.cuda.empty_cache()

# Train the model
# Train and save model
# Train the model
for _ in range(training_args.num_train_epochs):
    clear_cuda_cache()
    trainer.train()
    # Trainer handles saving based on TrainingArguments configuration
    clear_cuda_cache()

# Save the trained model to disk
model_path = './Finetuned_best_model_1'
if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save_pretrained(model_path)
print(f"Model saved to {model_path}")

