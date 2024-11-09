import pandas as pd
import json

# List of full paths to your CSV files
file_paths = [
    '/home/afarjana/Code_Summarization/Assignment2/frappe.csv',
    '/home/afarjana/Code_Summarization/Assignment2/glances.csv',
    '/home/afarjana/Code_Summarization/Assignment2/khmer.csv',
    '/home/afarjana/Code_Summarization/Assignment2/linkchecker.csv',
    '/home/afarjana/Code_Summarization/Assignment2/linuxcnc.csv',
    '/home/afarjana/Code_Summarization/Assignment2/orange3.csv',
    '/home/afarjana/Code_Summarization/Assignment2/biopython.csv',
    '/home/afarjana/Code_Summarization/Assignment2/netbox.csv',
    '/home/afarjana/Code_Summarization/Assignment2/beets.csv',
    '/home/afarjana/Code_Summarization/Assignment2/conda.csv',
    '/home/afarjana/Code_Summarization/Assignment2/prefect.csv',
    '/home/afarjana/Code_Summarization/Assignment2/pyload.csv'
]

# Initialize a list to store the methods and their documentation
all_methods = []

# Read each file and extract the "Method Code" and "Documentation" columns
for path in file_paths:
    df = pd.read_csv(path)
    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Assuming 'Method Code' and 'Documentation' columns exist
        method_entry = {
            "method_code": row['Method Code'],
            "documentation": row['Documentation']
        }
        all_methods.append(method_entry)

# Convert the list of method entries to JSON
json_data = json.dumps(all_methods, indent=4)

# Print first few entries to check the data
print("Sample of the JSON data:")
print(json.dumps(all_methods[:5], indent=4))  # Adjust the slice [:5] to display more or less data

# Specify the directory and filename where you want to save the JSON file
save_path = '/home/afarjana/Code_Summarization/Assignment2/Dataset.json'  # Update this line with your desired file path

# Write the JSON data to the file at the specified path
with open(save_path, 'w') as json_file:
    json_file.write(json_data)

print(f"JSON file created successfully at {save_path}")
