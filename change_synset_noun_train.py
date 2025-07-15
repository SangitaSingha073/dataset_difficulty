#pip install datasets
from datasets import load_dataset
import pandas as pd
import nltk 
from nltk.corpus import wordnet
from  nltk.tokenize import word_tokenize
from nltk import pos_tag


snli=load_dataset("snli")
train_data=snli["train"]
val_data=snli["validation"]
test_data=snli["test"]

train_data.to_csv('snli_train.csv')
test_data.to_csv('snli_test.csv')

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

def get_noun_synonym(word):
	synsets=wordnet.synsets(word, pos=wordnet.NOUN)
	if synsets:
		synonyms=[lemma.name() for lemma in synsets[0].lemmas() if lemma.name() != word ]
		return  synonyms[0] if synonyms else word 
	return word 

def replace_nouns_with_synonyms(sentence):
	words= word_tokenize(sentence)
	tagged_words=pos_tag(words)
	modified_words=[get_noun_synonym(word) if tag.startswith('NN') else word for word,tag in tagged_words]
	return " ".join(modified_words)




input_csv1="snli_train.csv"
df1=pd.read_csv(input_csv1)

input_csv2="snli_test.csv"
df2=pd.read_csv(input_csv2)

df1['premise'] = df1['premise'].apply(lambda x: replace_nouns_with_synonyms(str(x)))
df1['hypothesis'] = df1['hypothesis'].apply(lambda x: replace_nouns_with_synonyms(str(x)))
df1=df1[['premise', 'hypothesis','label']]
df1.index.name = 'Unnamed'
df1['sentence1'] = "PREMISE: " + df1['premise'] + " HYPOTHESIS: " + df1['hypothesis']
df1 = df1.reset_index()[['Unnamed', 'premise', 'hypothesis', 'label', 'sentence1']]
output_csv="snli_train_modified_noun.csv"
df1.to_csv(output_csv , index = False)
print(f"Modified sentences saved to {output_csv}")

df2['premise'] = df2['premise'].apply(lambda x: replace_nouns_with_synonyms(str(x)))
df2['hypothesis'] = df2['hypothesis'].apply(lambda x: replace_nouns_with_synonyms(str(x)))
df2=df2[['premise', 'hypothesis','label']]
df2.index.name = 'Unnamed'
df2['sentence1'] = "PREMISE: " + df2['premise'] + " HYPOTHESIS: " + df2['hypothesis']
df2 = df2.reset_index()[['Unnamed', 'premise', 'hypothesis', 'label', 'sentence1']]
output_csv="snli_test_modified_noun.csv"
df2.to_csv(output_csv , index = False)
print(f"Modified sentences saved to {output_csv}")
