import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
from nltk import pos_tag

import numpy as np
from tqdm import tqdm
import ast
tqdm.pandas(desc="My Progress Bar")


file_path = "addrec/anonymized_dataset/"
subreddit = ['ADHD','Blind','Disability']

""" 
#Consolidate as a single .csv for each subreddit
for s in subreddit:
    year_list = os.listdir(file_path+s)
    df = pd.DataFrame()
    for y in year_list:
        file_dir = os.listdir(file_path+s+'/'+y)
        for f in file_dir:
            df = pd.concat([df, pd.read_csv(file_path+s+'/'+y+'/'+f)])
            
    df.reset_index(inplace=True)
    df.to_csv(f'datasets/{s}.csv', index=False)

#Tokenizer & lemmatization
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return ADJ
    elif treebank_tag.startswith('V'):
        return VERB
    elif treebank_tag.startswith('N'):
        return NOUN
    elif treebank_tag.startswith('R'):
        return ADV
    else:  
        return NOUN
def lemmatize_text(text):
    try:
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_tokens]
        return lemmatized_tokens
    except (TypeError, ValueError):
        return ''
for s in subreddit:
    df = pd.read_csv(f'datasets/{s}.csv')
    df['anonymized_body'].fillna(0, inplace=True)
    df['anonymized_body_lemmatized'] = df['anonymized_body'].progress_apply(lemmatize_text)
    df.to_csv(f'datasets/{s}_lemma.csv', index=False)
"""

#Reaading & filtering keyword 
keywords = []
keyword_path = "Keywords.txt"
with open(keyword_path, 'r') as file:
    for line in file:
        keywords.append(line.strip())
filtered_list = [word for word in keywords if len(word.split()) < 2]
filtered_list = list(set(filtered_list))

#keyword-based filtering
def contains_keyword(tokens, keyword):
    keyword_lower = keyword.lower()
    return any(keyword_lower in token.lower() for token in tokens)
def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []  
for s in subreddit:
    s ='Disability'
    df = pd.read_csv(f'datasets/{s}_lemma.csv')
    df['anonymized_body_lemmatized'].fillna('', inplace=True)
    tqdm.pandas(desc="listfy")
    df['anonymized_body_lemmatized'] = df['anonymized_body_lemmatized'].progress_apply(safe_literal_eval)
    for k in tqdm(filtered_list):
        keyword = k
        filtered_df = df[df['anonymized_body_lemmatized'].apply(lambda tokens: keyword.lower() in [token.lower() for token in tokens])]
        if len(filtered_df) >=1:
            filtered_df.to_csv(f'datasets/keyword_filtered/{s}_lemma_{k}.csv', index=False)
        else:
            pass
    break
        

