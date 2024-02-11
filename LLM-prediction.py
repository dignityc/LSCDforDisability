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

model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(base_model_name)


df = pd.read_csv('datasets/Disability_merged_for_all_keywords.csv')

# Load the model pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024, temperature=0.2,repetition_penalty=1.19)
#repetition_penalty=1.19

# generate text
generated_texts = []
classification_result = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    instance = row['anonymized_body']
    question = f'###Below is a sentence that needs classification. Your task is to determine whether the sentene is relevant to domain of disability. Please respond with only one word: "Relevant" or "Irrelvant". ###Sentence:{instance} ###Classificaiton result:'
    #question = f'###Below is a sentence that needs classification. Your task is to determine whether the sentiment expressed in the sentence is "Neutral" or "Offensive". Please respond with only one word: "Neutral" or "Offensive". ###Sentence:{instance} ###Classificaiton result:'

    results = pipe(question)
    generated_text = results[0]['generated_text']  # Extract generated text

    # Append to the list as a tuple
    generated_texts.append((instance, generated_text, generated_text.split(':')[-1]))
    classification_result.append(generated_text.split(':')[-1])
    if index%5 ==0:
        results_df = pd.DataFrame(generated_texts, columns=['input', 'output', 'results'])


binary_result = [0 if 'Neutral' in result else 1 for result in classification_result]
r_df = df.loc[:len(classification_result)-1].copy()
r_df[model_name] = classification_result