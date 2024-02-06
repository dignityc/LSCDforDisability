from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from huggingface_hub import login
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from transformers.pipelines.pt_utils import KeyDataset


# load model
base_model_name = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(base_model_name)


df = pd.read_csv('datasets/Disability')

# Load the model pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512, temperature=0.1, repetition_penalty=1.19)

# generate text
generated_texts = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    instance = row['anonymized_body']
    question = f'###Below is an instruction that describes a task, paired with an input text. Write a response that appropriately completes the instruction. Instruction: Classify the input text as "hate speech", "offensive" or "neutral". Input text:{instance}'

    results = pipe(question)
    generated_text = results[0]['generated_text']  # Extract generated text

    # Append to the list as a tuple
    generated_texts.append((instance, generated_text))
    if index%100 ==0:
        results_df = pd.DataFrame(generated_texts, columns=['input', 'output'])

    
