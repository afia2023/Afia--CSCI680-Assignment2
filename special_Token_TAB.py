import json

def preprocess_code(code):
    # Start of sequence
    processed_code = "<s> "
    # Replace new lines with <TAB> (if you intended to replace with <NEWLINE> change it accordingly)
    processed_code += code.replace("\n", " <TAB> ")
    # Replace tabs with <TAB>
    processed_code = processed_code.replace("\t", " <TAB> ")
    # Replace four consecutive spaces with <TAB>
    processed_code = processed_code.replace("    ", " <TAB> ")
    # End of sequence
    processed_code += " </s>"
    return processed_code

# Load the dataset from JSON
with open('T5+_Dataset.json', 'r') as file:
    dataset = json.load(file)

# Process the dataset
for entry in dataset:
    # Retrieve the original 'method_code'
    original_code = entry.get("method_code", "")
    # Apply the tokenization logic
    processed_code = preprocess_code(original_code)
    # Update the entry with the processed code
    entry["processed_method_code"] = processed_code

# Save the updated dataset to a new JSON file
with open('INDENT_dataset.json', 'w') as file:
    json.dump(dataset, file, indent=4)

print("Processing completed and saved to 'INDENT_dataset.json'.")
