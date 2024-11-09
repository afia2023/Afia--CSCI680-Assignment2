import pandas as pd
from sklearn.model_selection import train_test_split
import re
from langdetect import detect
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_json("/home/afarjana/Code_Summarization/Assignment2/Dataset.json")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_documentation(text):
    """ Cleans documentation text using simple regex and lemmatization. """
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alpha characters
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def is_english(text):
    """ Check if the text is English. """
    try:
        return detect(text) == 'en'
    except:
        return False

def preprocess_data(row):
    """ Preprocess the dataset based on the given criteria. """
    # Check if documentation is None or not a string
    if row['documentation'] is None or not isinstance(row['documentation'], str):
        return None  # Skip rows with invalid or absent documentation

    # Remove examples with special tokens in the documentation
    if any(token in row['documentation'] for token in ['<img', 'https://']):
        return None

    # Ensure the documentation is in ASCII and English
    if not row['documentation'].isascii() or not is_english(row['documentation']):
        return None

    # Clean documentation
    documentation = clean_documentation(row['documentation'])

    # Count tokens in the cleaned documentation
    doc_tokens = documentation.split()
    if len(doc_tokens) < 3 or len(doc_tokens) > 256:
        return None

    # Count tokens in the method code using a simple whitespace split
    method_tokens = row['method_code'].split()
    if len(method_tokens) > 512:  # Check if method code is longer than 512 tokens
        return None

    return {
        'method_code': row['method_code'],
        'documentation': documentation
    }

# Apply preprocessing and drop any None values
processed_data = df.apply(preprocess_data, axis=1).dropna()

# Convert the resulting Series of dictionaries into a DataFrame
if not processed_data.empty:
    processed_df = pd.DataFrame(list(processed_data.values))
    
    # Save the whole processed dataset
    processed_df.to_json("T5+_Dataset.json", orient='records', indent=4)
    print("Preprocessed dataset saved successfully.")

    # Print the length of the preprocessed dataset
    print(f"Length of the preprocessed dataset: {len(processed_df)}")
else:
    print("No valid rows to process after filtering.")