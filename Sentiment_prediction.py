from transformers import pipeline
import torch
import pandas as pd
from tqdm import tqdm




dataset = 'ADHD'
path = 'Sentiment_prediction/Merged_sets_for_all_keywords'
File_path = f'{path}/{dataset}_merged_for_all_keywords.csv'

df = pd.read_csv(File_path)
df = df[df['sentiment']==1].reset_index().copy()

device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU

model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=device)

print(sentiment_task('This is a block')[0])

labels = []
predictions = []
for index, row in tqdm(df.iterrows(), total=len(df)):
    result = sentiment_task(row['anonymized_body'][:800])[0]
    labels.append(result['label'])
    predictions.append(result['score'])

r_df = df.loc[:(len(labels)-1)].copy()
r_df['post_sentiment_label'] = labels
r_df['post_sentiment_score'] = predictions

r_df.to_csv(f"{path}/{dataset}_post_sentiment.csv")