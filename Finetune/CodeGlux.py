from datasets import load_dataset

# Load the dataset with the specified splits
dataset = load_dataset("code_x_glue_ct_code_to_text",'python')

# Access each split
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# Save each split to a local JSON file if needed
train_data.to_json("Code_train.json")
val_data.to_json("Code_validation.json")
test_data.to_json("Code_test.json")

print("Datasets downloaded and saved as JSON files.")
