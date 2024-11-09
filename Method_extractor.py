

import os
import csv
import uuid
import ast
import astor
import re

class MethodExtractor(ast.NodeVisitor):
    def __init__(self):
        self.methods = []

    def visit_FunctionDef(self, node):
        docstring = ast.get_docstring(node)
        method_code = ''.join(ast.get_source_segment(source, node).split('\n'))
        self.methods.append({
            'name': node.name,
            'start_line': node.lineno,
            'end_line': node.end_lineno,
            'code': remove_docstring(method_code),
            'docstring': docstring if docstring else "",
            'code_with_doc': f"{method_code}\n\n{docstring}" if docstring else method_code
        })

def extract_methods_from_file(file_path):
    global source
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            source = file.read()
        tree = ast.parse(source)
        extractor = MethodExtractor()
        extractor.visit(tree)
        return extractor.methods
    except SyntaxError as e:
        print(f"Syntax error in file {file_path}: {e}")
        return []
def remove_docstring(code):
    """
    Removes docstrings from the given function code.
    """
    # Regular expression to match and remove triple-quoted docstrings (""" or ''')
    pattern = r'""".*?"""|\'\'\'.*?\'\'\''
    
    # Substitute the docstring with an empty string
    cleaned_code = re.sub(pattern, '', code, flags=re.DOTALL)
    
    return cleaned_code.strip()

def process_directory(directory, output_csv):
    with open(output_csv, mode='w', encoding='utf-8', newline="") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        columnData = ['Project Name', 'File Name', 'Method Name', 'Start Line', 'End Line', 'Method Code', 'Documentation', 'Code With Doc', 'TokenID']
        writer.writerow(columnData)

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    methods = extract_methods_from_file(full_path)
                    for method in methods:
                        token_id = uuid.uuid4()
                        writer.writerow([os.path.basename(root), file, method['name'], method['start_line'], method['end_line'], method['code'], method['docstring'], method['code_with_doc'], token_id])

# Example usage
if __name__ == "__main__":
    process_directory('C:/Users/Afia Farjana/All_Git_Repos/sphinx', 'E:/AI for SE/sphinx.csv')


