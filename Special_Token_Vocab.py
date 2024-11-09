from transformers import AutoTokenizer

# Load the tokenizer (adjust the path to where your tokenizer is saved)
tokenizer = AutoTokenizer.from_pretrained("/home/afarjana/Code_Summarization/Assignment2/best_model_1")
# Check if '<TAB>' is in the tokenizer's vocabulary
if '<TAB>' in tokenizer.vocab:
    print("'<TAB>' is in the tokenizer's vocabulary.")
else:
    print("'<TAB>' is not in the tokenizer's vocabulary.")

# Example string with '<TAB>'
test_string = "def example_function():<TAB>return 42"

# Tokenize the string
tokenized_output = tokenizer.encode(test_string, return_tensors="pt")
decoded_tokens = tokenizer.convert_ids_to_tokens(tokenized_output[0])

print("Tokenized output (token IDs):", tokenized_output)
print("Decoded tokens:", decoded_tokens)



