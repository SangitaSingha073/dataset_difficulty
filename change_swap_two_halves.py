import pandas as pd

def split_sentence(sentence):
    if not isinstance(sentence, str):  
        return "", ""
    
    words = sentence.split()
    mid = (len(words) + 1) // 2  
    return " ".join(words[:mid]), " ".join(words[mid:])

def process_snli_file(input_file, output_file):
    df = pd.read_csv(input_file)

  
    df["premise"] = df["premise"].fillna("")
    df["hypothesis"] = df["hypothesis"].fillna("")

    df["premise_part1"], df["premise_part2"] = zip(*df["premise"].apply(split_sentence))
    df["hypothesis_part1"], df["hypothesis_part2"] = zip(*df["hypothesis"].apply(split_sentence))

    df["swapped_premise"] = df["premise_part2"] + " " + df["premise_part1"]
    df["swapped_hypothesis"] = df["hypothesis_part2"] + " " + df["hypothesis_part1"]

    df["sentence1"] = "Premise: " + df["swapped_premise"] + " Hypothesis: " + df["swapped_hypothesis"]
    df.rename(columns={"index": "Unnamed: 0"}, inplace=True)
    df = df.reset_index()[["Unnamed: 0", "premise", "hypothesis", "label", "sentence1"]]

    df.to_csv(output_file, index=False)
    print(f"Processed file saved as: {output_file}")


process_snli_file("snli_train.csv", "snli_train_swap_two_halves.csv")
process_snli_file("snli_test.csv", "snli_test_swap_two_halves.csv")
