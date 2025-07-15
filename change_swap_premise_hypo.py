import pandas as pd

# Function to process SNLI dataset and swap `premise` & `hypothesis` in `sentence1`
def process_snli_file(input_file, output_file):
    # Load the dataset and keep only the necessary columns
    df = pd.read_csv(input_file).filter(["premise", "hypothesis", "label"])  

    # Fill NaN values with empty strings to prevent sentence1 from having NaN
    df["premise"] = df["premise"].fillna("")
    df["hypothesis"] = df["hypothesis"].fillna("")

    # Construct sentence1 by swapping premise and hypothesis
    df["sentence1"] = "Premise: " + df["hypothesis"] + " Hypothesis: " + df["premise"]

    # Remove rows where `sentence1` is empty after stripping whitespace
    df = df[df["sentence1"].str.strip() != ""]

    # Filter only valid labels
    df = df[df["label"].isin([0, 1, 2])]

    # Reset index and rename it to "Unnamed: 0"
    df_final = df.reset_index().rename(columns={"index": "Unnamed: 0"})

    # Save the modified dataset
    df_final.to_csv(output_file, index=False)

    print(f"Processed file saved as: {output_file}")

# Process both train and test datasets
process_snli_file("snli_train.csv", "snli_train_swap_premise_hypo.csv")
process_snli_file("snli_test.csv", "snli_test_swap_premise_hypo.csv")
