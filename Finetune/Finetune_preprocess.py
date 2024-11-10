import json
import pandas as pd
import re
import random

# Load JSON lines file
data = []
with open("Code_train.json", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Sample 7,000 examples randomly
sampled_data = random.sample(data, 500)

# Function to preprocess each function code
def preprocess_function(function_code):
    # Replace indentation and line breaks with <TAB> for input method
    function_code = function_code.replace("    ", "<TAB>")  # Replace 4-space indentation
    function_code = function_code.replace("\n", "<TAB>")    # Replace line breaks
    
    # Find the first `if` condition
    if_match = re.search(r"if\s+.*:", function_code)
    if if_match:
        # Extract the `if` condition as target block without any additional formatting
        if_condition = if_match.group(0)
        
        # Remove all `<TAB>` tokens from the `if_condition` to keep it in original form
        if_condition = if_condition.replace("<TAB>", "").strip()
        
        # Mask the `if` condition in the function code
        masked_code = function_code.replace(if_match.group(0), "<extra_id_0>", 1)
        
        # Return masked function code as input_method and the clean if condition as target_block
        return masked_code, if_condition
    else:
        # If no `if` condition is found, skip this example
        return None, None

# Initialize lists for processed data
input_methods = []
target_blocks = []

# Process each function in the sample
for example in sampled_data:
    function_code = example.get("code", "")  # Adjust based on the structure of your JSON
    input_method, target_block = preprocess_function(function_code)
    
    # Only add to the dataset if an `if` condition was found and processed
    if input_method and target_block:
        input_methods.append(input_method)
        target_blocks.append(target_block)

# Create a DataFrame for easy saving to CSV
df = pd.DataFrame({
    "input_method": input_methods,
    "target_block": target_blocks
})

# Save processed data to JSON format
processed_data = [{"input_method": im, "target_block": tb} for im, tb in zip(input_methods, target_blocks)]
with open("Finetune_processed_dataset.json", "w") as f:
    json.dump(processed_data, f, indent=4)

# Save processed data to CSV format
df.to_csv("Finetune_processed_dataset.csv", index=False)

print("Dataset preprocessing complete. Saved as JSON and CSV formats.")


