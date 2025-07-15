import pandas as pd
import random

# File paths
train_file_path = "snli_train.csv"  # Update this if needed
test_file_path = "snli_test.csv"  # Update this if needed
modified_train_file_path = "snli_train_shuffled.csv"
modified_test_file_path = "snli_test_shuffled.csv"

# Function to shuffle words in a sentence
def shuffle_sentence(sentence):
    if isinstance(sentence, str):  # Ensure the value is a string
        words = sentence.split()
        random.shuffle(words)
        return " ".join(words)
    return ""  # Return empty string if the sentence is NaN or not a string

def process_file(file_path, modified_file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Handle missing values by replacing NaN with an empty string
    df = df.copy()
    df.loc[:, "premise"] = df["premise"].fillna("")
    df.loc[:, "hypothesis"] = df["hypothesis"].fillna("")
    
    # Apply the shuffling to both "premise" and "hypothesis" and store in "sentence1"
    df["sentence1"] = "PREMISE: " + df["premise"].apply(shuffle_sentence) + " HYPOTHESIS: " + df["hypothesis"].apply(shuffle_sentence)
    
    # Add 'Unnamed: 0' column only if it doesn't exist
    if "Unnamed: 0" not in df.columns:
        df.insert(0, "Unnamed: 0", df.index)
    
    # Save the modified file
    df.to_csv(modified_file_path, index=False)
    
    print(f"Modified file saved as: {modified_file_path}")

# Process both train and test files
process_file(train_file_path, modified_train_file_path)
process_file(test_file_path, modified_test_file_path)
