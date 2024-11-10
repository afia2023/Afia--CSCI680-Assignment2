import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
from transformers import AutoModelForSeq2SeqLM
from datasets import Dataset


import pandas as pd

# Load the test dataset (replace with the actual path to your test dataset file)
test_df = pd.read_json("/home/afarjana/Code_Summarization/Assignment2/Finetune_Test_dataset.json")


def tokenize_data_for_evaluation(dataframe, tokenizer):
    dataframe = dataframe.reset_index(drop=True)

    def tokenize_function(examples):
        model_inputs = tokenizer(examples['input_method'], max_length=512, padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target_block'], max_length=512, padding="max_length", truncation=True)
        model_inputs['labels'] = labels['input_ids']
        # Keep the 'input_method' and 'target_block' columns for evaluation
        model_inputs['input_method'] = examples['input_method']
        model_inputs['target_block'] = examples['target_block']
        return model_inputs

    tokenized_dataset = Dataset.from_pandas(dataframe).map(tokenize_function, batched=True, batch_size=5000)
    return tokenized_dataset

# Apply tokenization on the test dataset
tokenizer = AutoTokenizer.from_pretrained("best_model_1")  # Adjust if needed
test_dataset = tokenize_data_for_evaluation(test_df, tokenizer)

# Paths to your model checkpoints
checkpoint_paths = ["results1/checkpoint-50", "results1/checkpoint-2000", "results1/checkpoint-2030"]

def evaluate_and_save_output_whole_dataset(model, tokenizer, dataset, output_file):
    model.eval()  # Set the model to evaluation mode
    results = []
    total_correct = 0

    # Loop through all samples in the dataset
    for idx, item in enumerate(dataset):
        # Prepare input for model
        inputs = tokenizer(item['input_method'], return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate output
        outputs = model.generate(inputs['input_ids'], max_new_tokens=50)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check if the generated output is correct
        is_correct = decoded_output.strip() == item['target_block'].strip()
        if is_correct:
            total_correct += 1

        # Store result for this sample
        result = {
            'input_method': item['input_method'],
            'target_block': item['target_block'],
            'generated_output': decoded_output,
            'is_correct': is_correct
        }
        results.append(result)

    # Calculate accuracy for the entire dataset
    accuracy = (total_correct / len(dataset)) * 100

    # Convert results to a DataFrame and select only the first 5 rows to save
    df = pd.DataFrame(results[:5])  # Only save first 5 samples

    # Add overall accuracy as an additional row at the end
    accuracy_row = pd.DataFrame([{
        'input_method': 'Overall Accuracy',
        'target_block': '',
        'generated_output': '',
        'is_correct': f'{accuracy:.2f}%'
    }])
    df = pd.concat([df, accuracy_row], ignore_index=True)

    # Save to CSV
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Output saved to {output_file} with accuracy calculated on the entire dataset.")

# Loop through each checkpoint and save its output to a separate file
for checkpoint_path in checkpoint_paths:
    # Load model from the checkpoint
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define an output filename for each checkpoint
    checkpoint_name = checkpoint_path.split('/')[-1]
    output_file = f"output_samples_{checkpoint_name}_whole_dataset.csv"
    
    # Evaluate and save output for this checkpoint
    evaluate_and_save_output_whole_dataset(model, tokenizer, test_dataset, output_file)
