from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from huggingface_hub import login
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from transformers.pipelines.pt_utils import KeyDataset



# load model
model_name = 'Llama-2-13b'
base_model_name = f"meta-llama/{model_name}-chat-hf"

model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map='cuda:0', load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

dataset = 'Blind'
original_df = pd.read_csv(f'datasets/2nd_filtering/{dataset}_post_sentiment.csv')
df = original_df[original_df['sentiment']==1].copy().reset_index(drop=True)

# Load the model pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2048, temperature=0.1,repetition_penalty=1.19)
#repetition_penalty=1.19

# generate text
generated_texts = []
classification_result = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    instance = row['anonymized_body']
    question = f'### Below is a text that needs classification. Your task is to determine whether the text is relevant to the domain of physical disability. Please respond with only one word: "Relevant" or "Irrelevant". ###Text:{instance} ###Classificaiton result:'


    results = pipe(question)
    generated_text = results[0]['generated_text']  # Extract generated text

    # Append to the list as a tuple
    generated_texts.append((instance, generated_text, generated_text.split(':')[-1]))
    classification_result.append(generated_text.split(':')[-1])
    if index%5000 == 0:
        binary_result = [1 if 'Relevant' in result else 0 for result in classification_result]
        r_df = df.loc[:(len(classification_result)-1)].copy()
        r_df['Relevance'] = binary_result
        r_df.to_csv(f'datasets/2nd_filtering/{dataset}_relevance.csv')
                
        

binary_result = [1 if 'Relevant' in result else 0 for result in classification_result]
r_df = df.loc[:(len(classification_result)-1)].copy()
r_df['Relevance'] = binary_result
r_df = r_df.drop(columns=['Unnamed: 0', 'index'])
#r_df.to_csv(f'datasets/2nd_filtering/{dataset}_relevance.csv')
