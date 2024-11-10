# Assignment 2 : CodeT5: Training a Transformer model for Predicting if statements

1. Model Checkpoint-2030-https://drive.google.com/drive/folders/1Clo230JaRyDeZKLdvDHPkjBj_UnrvHqU?usp=sharing
2. Model Checkpoint-2000 - https://drive.google.com/drive/folders/10wl6dJwrBPS2U5VZfGpDZ4TLf91tBrnw?usp=sharing
3. Model Checkpoint-50- https://drive.google.com/drive/folders/1g4K3JMzbkAPxef_wOC6eDwyv-xUA4s0X?usp=sharing

## Introduction
This project enhances the CodeT5 model's ability to predict if conditions in Python code through specialized pre-training and fine-tuning phases. The dataset derived from CodeXGLUE was fine-tuned over 10 epochs, focusing on identifying and inserting logical conditions accurately. Output evaluations for three checkpoints are available in CSV format, showcasing the model's performance on the entire dataset, though only the first five indices are detailed in the CSV files. This README provides an overview of the methodologies and key findings, guiding users through the project's structured approach and its implications for automated coding tools.

## Table of Contents
- [Project Title](#project-title)

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
- [Usage](#usage)
  - [Pre-training](#pre-training)
  - [Fine-tuning](#fine-tuning)
  - -[Evaluation] 

### Pre-training
To start the pre-training process, execute the following commands in sequence:
```bash
# Extract method codes from source files
python Method_extractor.py

# Convert CSV data to JSON format
python json_conversion.py

# Preprocess the dataset
python Preprocess_dataset.py

# Maintain indentation with special tokens
python special_Token_TAB.py

# Apply the masking logic
python Masked.py

# Train the CodeT5 model
python T5_2.py





