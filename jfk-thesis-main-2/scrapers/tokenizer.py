import re
import spacy
import pandas as pd
from nltk.corpus import stopwords
from string import punctuation

# Load English model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # only for tokenization

# Extend stopwords list
custom_stopwords = set(stopwords.words('english')).union({
    'mr', 'ms', 'mrs', 'dr', 'etc', 'may', 'said', 'would',
    'could', 'also', 'one', 'two', 'three', 're', 'subject', 'document', 
    'file', 'record', 'memo', 'memorandum', 'pdf', 'return', 'background',
    'use', 'only', 'page', 'release'
})

def preprocess_for_topic_modeling(text):
    # 1. Remove numeric references and document codes
    text = re.sub(r'\b\d{2,}\b', ' ', text)  
    text = re.sub(r'104-\d+-\d+', ' ', text)
    text = re.sub(r'\bpage\s*\d+\b', ' ', text, flags=re.IGNORECASE)
    
    # 2. Normalize and clean punctuation
    text = re.sub(r'[\[\]\(\){}_/\\]', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    # 3. Tokenize using spaCy (handles punctuation and compound words better than split)
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        # keep only alphabetic words longer than 2 characters
        if token.is_alpha and len(token.text) > 2:
            lemma = token.lemma_.lower()
            if lemma not in custom_stopwords:
                tokens.append(lemma)
                
    return tokens

# Load your CSV
df = pd.read_csv("/Users/furkandemir/Desktop/Thesis/files/1/simple/extracted_texts.csv")

# Apply preprocessing
df["Tokens"] = df["Extracted_Text"].apply(preprocess_for_topic_modeling)

# Save preprocessed version
df.to_csv("/Users/furkandemir/Desktop/Thesis/files/1/simple/extracted_tokens.csv", index=False)

# Check result
print(df[["Document_ID", "Tokens"]].head(3))