import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm


path = 'datasets/keyword_filtered'
file_list = os.listdir(path)
subreddit = 'Disability'
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
sample_cutoff_value = 10

#merging sets into a single dataframe for a subreddit
filtered_files = [file for file in file_list if file.startswith(subreddit)]
all_df = pd.DataFrame()
for f in tqdm(filtered_files):
    df = pd.read_csv(f'datasets/keyword_filtered/{f}')
    df['keyword'] = f.split('_')[-1].split('.')[0]
    all_df = pd.concat([all_df,df], axis=0)

all_df = all_df.drop('index', axis=1)
all_df = all_df.reset_index()

word_count_dict = all_df['keyword'].value_counts().to_dict()
filtered_word_count_dict = {key: value for key, value in word_count_dict.items() if value > sample_cutoff_value}


#LDA
result = []
for key, value in tqdm(filtered_word_count_dict.items()):
    row = []
    row.append(key)
    filtered_df = all_df[all_df['keyword']==key].copy()
    filtered_df['anonymized_body_stringfied'] = filtered_df['anonymized_body_lemmatized'].apply(ast.literal_eval)
    filtered_df['anonymized_body_textualized'] = filtered_df['anonymized_body_stringfied'].apply(lambda x: ' '.join(x))

    text_data = filtered_df['anonymized_body_textualized']
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(text_data)

    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda_model.components_):
        #print(f"topic {idx+1}:")
        word_set = [feature_names[i] for i in topic.argsort()[-20:]]
        #print(word_set)
        scores = sid.polarity_scores(', '.join(word_set))['compound']
        row.append(scores)
        #print(f"Sentiment_Score:{scores}")
    result.append(row)
    
result_df = pd.DataFrame(result)
result_df.to_csv(f'datasets/LDA_{subreddit}.csv')