from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from huggingface_hub import login
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from transformers.pipelines.pt_utils import KeyDataset


#load dataset 
dataset_name = 'Blind'
df = pd.read_csv(f'datasets/{dataset_name}_merged_for_all_keywords.csv')

# load model

llama2_7b_name = f"meta-llama/Llama-2-7b-chat-hf"
llama2_13b_name = f"meta-llama/Llama-2-13b-chat-hf"

llama2_7b_model = AutoModelForCausalLM.from_pretrained(llama2_7b_name, device_map='cuda', load_in_8bit=True)
llama2_7b_tokenizer = AutoTokenizer.from_pretrained(llama2_7b_name)
llama2_13b_model = AutoModelForCausalLM.from_pretrained(llama2_13b_name, device_map='cuda', load_in_8bit=True)
llama2_13b_tokenizer = AutoTokenizer.from_pretrained(llama2_13b_name)


# Load the model pipeline
llama2_7b_pipe = pipeline(task="text-generation", model=llama2_7b_model, tokenizer=llama2_7b_tokenizer, max_length=1024, temperature=0.2,repetition_penalty=1.19)
llama2_13b_pipe = pipeline(task="text-generation", model=llama2_13b_model, tokenizer=llama2_13b_tokenizer, max_length=1024, temperature=0.2,repetition_penalty=1.19)
#repetition_penalty=1.19

# generate text
generated_texts = []
llama2_7b_label = []
llama2_13b_label = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    instance = row['anonymized_body']
    question = f'###Below is a sentence that needs classification. Your task is to determine whether the sentene is offensive or not. Please respond with only one word: "Neutral" or "Offensive". ###Sentence:{instance} ###Classificaiton result:'

    llama2_7b_label.append(llama2_7b_pipe(question)[0]['generated_text'].split(':')[-1])
    llama2_13b_label.append(llama2_13b_pipe(question)[0]['generated_text'].split(':')[-1])


    # Append to the list as a tuple
    if index == 25:
        break

llama2_7b_label_encoding = [0 if 'Neutral' in result else 1 for result in llama2_7b_label]
llama2_13b_label_encoding = [0 if 'Neutral' in result else 1 for result in llama2_13b_label]
r_df = df.loc[:len(llama2_7b_label)-1].copy()
r_df['llama2_7b_pred'] = llama2_7b_label_encoding
r_df['llama2_13b_pred'] = llama2_13b_label_encoding