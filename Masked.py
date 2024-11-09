import json
import ast
import random
import tokenize
from io import BytesIO
from transformers import AutoTokenizer

def get_identifiers(code):
    try:
        tree = ast.parse(code)
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Load, ast.Store)):
                identifiers.add(node.id)
        return identifiers
    except SyntaxError as e:
        print("Syntax error in the following code:", code)
        print(str(e))
        return set()

def mask_code(code, identifiers, tokenizer, mask_probability=1.0):
    tokens = ["<s>"]  # Start token
    source = code.encode('utf-8')
    try:
        tokens_gen = tokenize.tokenize(BytesIO(source).readline)
        for token in tokens_gen:
            token_str = token.string
            should_mask = (token.type == tokenize.NAME and token_str in identifiers and
                           token_str not in {"<s>", "</s>", "<TAB>"}) or \
                          (token_str in {'if', 'else', 'for', 'while'} and
                           token_str not in {"<s>", "</s>", "<TAB>"})
            if should_mask and random.random() < mask_probability:
                masked_token = tokenizer.mask_token
                tokens.append(masked_token)
            else:
                tokens.append(token_str)
    except tokenize.TokenError as e:
        print(f"Tokenization error in code: {code}")
        print(str(e))
        tokens = [code]  # Use original code in case of tokenization error
    tokens.append("</s>")  # End token
    return ' '.join(tokens)

def process_dataset(dataset, tokenizer):
    for entry in dataset:
        identifiers = get_identifiers(entry['processed_method_code'])
        masked_code = mask_code(entry['processed_method_code'], identifiers, tokenizer)
        entry['masked_method_code'] = masked_code

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")

    # Load dataset
    with open('INDENT_dataset.json', 'r') as file:
        dataset = json.load(file)

    # Process dataset with masking logic
    process_dataset(dataset, tokenizer)

    # Save the updated dataset with masked code
    with open('Big_Dataset1.json', 'w') as file:
        json.dump(dataset, file, indent=4)

    print("Dataset processed and saved with masking logic applied.")
