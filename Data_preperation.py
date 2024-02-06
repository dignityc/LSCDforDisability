import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import numpy as np


file_path = "addrec/anonymized_dataset/"
subreddit = ['ADHD','Blind','Disability']

#Consolidate as a single .csv for each subreddit
for s in subreddit:
    year_list = os.listdir(file_path+s)
    df = pd.DataFrame()
    for y in year_list:
        file_dir = os.listdir(file_path+s+'/'+y)
        for f in file_dir:
            df = pd.concat([df, pd.read_csv(file_path+s+'/'+y+'/'+f)])
            
    df.reset_index(inplace=True)
    df.to_csv(f'datasets/{s}', index=False)

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
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_tokens]
    return lemmatized_tokens
for s in subreddit:
    s = 'Disability'
    df = pd.read_csv(f'datasets/{s}.csv')
    df['anonymized_body_lemmatized'] = df['anonymized_body'].apply(lemmatize_text)
    break

#keyword-based filtering
keyword = 'dumb'
def contains_keyword(tokens, keyword):
    keyword_lower = keyword.lower()
    return any(keyword_lower in token.lower() for token in tokens)
filtered_df = df[df['anonymized_body_lemmatized'].apply(lambda tokens: contains_keyword(tokens, keyword))]


#LSA 
filtered_df['anonymized_body_stringfied'] = filtered_df['anonymized_body_lemmatized'].apply(lambda x: ' '.join(x))

tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
svd_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=100, random_state=122)

lsa_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('svd', svd_model)
])

lsa_matrix = lsa_pipeline.fit_transform(filtered_df['anonymized_body_stringfied'])

#print("LSA Topic Matrix:")
#print(lsa_matrix)

print("\nTopic Word Weights:")
feature_names = tfidf_vectorizer.get_feature_names_out()
for i, topic in enumerate(svd_model.components_):
    print(f"Topic {i}:")
    print([feature_names[index] for index in topic.argsort()[:-6:-1]])
    print(np.sort(topic)[:-6:-1])


#word embedding-based simliarty
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import pandas as pd


model = Word2Vec(vector_size=100, min_count=1)
model.build_vocab(df['anonymized_body_lemmatized'])
model.train(df['anonymized_body_lemmatized'], total_examples=model.corpus_count, epochs=model.epochs)

word1 = 'dumb'
similarities = {word: model.wv.similarity(word1, word) 
                for word in model.wv.index_to_key if word != word1}

sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

N = 10  
for word, similarity in sorted_similarities[:N]:
    print(f"Similarity between '{word1}' and '{word}': {similarity:.4f}")