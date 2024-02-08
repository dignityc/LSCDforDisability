import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import ast
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


path = 'datasets/'
subreddit = 'ADHD'
n_components=10

#merging sets into a single dataframe for a subreddit
df = pd.read_csv(f"{path}{subreddit}_lemma.csv")
df['anonymized_body_lemmatized'] = df['anonymized_body_lemmatized'].fillna('[]')
#LSA 
def LSA(df, n_components, filtered_list):
    df['anonymized_body_stringfied'] = df['anonymized_body_lemmatized'].apply(ast.literal_eval)
    df['anonymized_body_textualized'] = df['anonymized_body_stringfied'].apply(lambda x: ' '.join(x))
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['anonymized_body_textualized'])

    svd_model = TruncatedSVD(n_components=n_components, algorithm='randomized', n_iter=100, random_state=122)
    lsa_matrix = svd_model.fit_transform(tfidf_matrix)

    word_topic_matrix = svd_model.components_.T
    word_similarity_matrix = cosine_similarity(word_topic_matrix)

    words = tfidf_vectorizer.get_feature_names_out()

    word_similarity_df = pd.DataFrame(word_similarity_matrix, index=words, columns=words)



    selected_keywords = list(set(filtered_list) & set(words))

    filtered_similarity_df = word_similarity_df.loc[selected_keywords, selected_keywords]

    return filtered_similarity_df

##### Top-n keyword selection to represent better visualization
#Reaading & filtering keyword 
keywords = []
keyword_path = "Keywords.txt"
with open(keyword_path, 'r') as file:
    for line in file:
        keywords.append(line.strip())
filtered_list = [word for word in keywords if len(word.split()) < 2]
filtered_list = list(set(filtered_list))
filtered_list = [word.lower() for word in filtered_list]

df['anonymized_body_stringfied'] = df['anonymized_body_lemmatized'].apply(ast.literal_eval)
def calculate_total_tf(df):
    total_word_counts = Counter()
    for word_list in df['anonymized_body_stringfied']:
        total_word_counts.update(word_list)
    return pd.Series(total_word_counts)

total_tf_series = calculate_total_tf(df)
total_tf_series = total_tf_series.reindex(filtered_list, fill_value=0)
sorted_series = total_tf_series.sort_values(ascending=False)


#time spliting
df_sorted = df.sort_values(by='created_utc')
total_rows = len(df_sorted)
half_rows = total_rows // 2
df_first_half = df_sorted.iloc[:half_rows].copy()
df_second_half = df_sorted.iloc[half_rows:].copy()

sim_matrix = LSA(df, 10, filtered_list)
sim_matrix1 = LSA(df_first_half, 10, filtered_list)
sim_matrix2 = LSA(df_second_half, 10, filtered_list)

key_set = list(set(sim_matrix1.columns) & set(sim_matrix2.columns))

top_n = 20
sorted_series = sorted_series[sorted_series.index.isin(key_set)]
top_values = sorted_series.head(top_n).keys()

sim_matrix = sim_matrix.loc[top_values,top_values]
sim_matrix1 = sim_matrix1.loc[top_values,top_values]
sim_matrix2 = sim_matrix2.loc[top_values,top_values]


plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, cmap='coolwarm', annot=False, fmt=".2f")
plt.title('all time')
plt.xlabel('Keywords')
plt.ylabel('Keywords')
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix1, cmap='coolwarm', annot=False, fmt=".2f")
plt.title('t1')
plt.xlabel('Keywords')
plt.ylabel('Keywords')
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix2, cmap='coolwarm', annot=False, fmt=".2f")
plt.title('t2')
plt.xlabel('Keywords')
plt.ylabel('Keywords')
plt.show()


