from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from huggingface_hub import login
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load model
llama2_7b_name = f"meta-llama/Llama-2-7b-chat-hf"
llama2_13b_name = f"meta-llama/Llama-2-13b-chat-hf"

llama2_7b_model = AutoModelForCausalLM.from_pretrained(llama2_7b_name, device_map='cuda:0', load_in_8bit=True,use_auth_token=True)
llama2_7b_tokenizer = AutoTokenizer.from_pretrained(llama2_7b_name)
llama2_13b_model = AutoModelForCausalLM.from_pretrained(llama2_13b_name, device_map='cuda:0', load_in_8bit=True)
llama2_13b_tokenizer = AutoTokenizer.from_pretrained(llama2_13b_name)


# Load the model pipeline
llama2_7b_pipe = pipeline(task="text-generation", model=llama2_7b_model, tokenizer=llama2_7b_tokenizer, max_length=1024, temperature=0.7,repetition_penalty=1.19)
llama2_13b_pipe = pipeline(task="text-generation", model=llama2_13b_model, tokenizer=llama2_13b_tokenizer, max_length=1024, temperature=0.2,repetition_penalty=1.19)
#repetition_penalty=1.19
TimeLMs_pipe = pipeline('text-classification', model="cardiffnlp/twitter-roberta-base-hate-latest-st")
hatexplain_pipe = pipeline('text-classification', model="Hate-speech-CNERG/bert-base-uncased-hatexplain")


#load dataset 
Path = "datasets/annotatedSets"
datasets = ['ADHD', 'Blind', 'Disability']
df = pd.DataFrame()
for d in datasets:
    file_path = f"{Path}/{d}_Hate.csv"
    reading = pd.read_csv(file_path, header=None, index_col=0)
    reading['True_label'] = 1
    reading['datasets'] = d
    df = pd.concat([df,reading], axis=0)
    file_path = f"{Path}/{d}_Neutral.csv"
    reading = pd.read_csv(file_path, header=None, index_col=0)
    reading['True_label'] = 0
    reading['datasets'] = d
    df = pd.concat([df,reading], axis=0)

df = df.reset_index(drop=True)   

login(token='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')

# generate text
original_df = df.copy()
results = pd.DataFrame()
score = []
for iter in range(0,10):
    print(f"=========iter:{iter} running==========")

    generated_texts = []
    llama2_7b_label = []
    llama2_13b_label = []
    TimeLMs_label = []
    hatexplain_label = []
    llama2_7b_label_FL = []
    llama2_13b_label_FL = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        instance = row[3]

        question = f'###Below is a text that needs classification. Your task is to determine whether the text is offensive or not. Please respond with only one word: "Neutral" or "Offensive". ###Text:{instance} ###Classification result:'

        llama2_7b_label.append(llama2_7b_pipe(question)[0]['generated_text'].split(':')[-1])
        llama2_13b_label.append(llama2_13b_pipe(question)[0]['generated_text'].split(':')[-1])
        TimeLMs_label.append(TimeLMs_pipe(instance)[0]['label'])
        hatexplain_label.append(hatexplain_pipe(instance)[0]['label'])

        #Few-shot learning for llama2
        question = f'###Below is a text that needs classification. Your task is to determine whether the sentene is offensive or not. Please respond with only one word: "Neutral" or "Offensive". an exmaple of offensive text: "Public schools can fucking suck sometimes." and an example of neutral text: "Good response.  I think it is all about context.  Also, keep in mind that retard is a French word meaning "late". ###Text:{instance} ###Classificaiton result:'
        llama2_7b_label_FL.append(llama2_7b_pipe(question)[0]['generated_text'].split(':')[-1])
        llama2_13b_label_FL.append(llama2_13b_pipe(question)[0]['generated_text'].split(':')[-1])



    llama2_7b_label_encoding = [0 if 'Neutral' in result else 1 for result in llama2_7b_label]
    llama2_13b_label_encoding = [0 if 'Neutral' in result else 1 for result in llama2_13b_label]
    r_df = df.loc[:len(llama2_7b_label)-1].copy()
    r_df['llama2_7b_pred_text'] = llama2_7b_label
    r_df['llama2_13b_pred_text'] = llama2_13b_label
    r_df['llama2_7b_pred'] = llama2_7b_label_encoding
    r_df['llama2_13b_pred'] = llama2_13b_label_encoding

    TimeLMs_label_encoding = [0 if 'not_hate' in result else 1 for result in TimeLMs_label]
    r_df['TimeLMs_label'] = TimeLMs_label_encoding

    hatexplain_label_encoding = [0 if 'not_hate' in result else 1 for result in hatexplain_label]
    r_df['hatexplain_label'] = hatexplain_label_encoding

    llama2_7b_label_encoding_FL = [0 if 'Neutral' in result else 1 for result in llama2_7b_label_FL]
    llama2_13b_label_encoding_FL = [0 if 'Neutral' in result else 1 for result in llama2_13b_label_FL]
    r_df['llama2_7b_pred_FL_text'] = llama2_7b_label_FL
    r_df['llama2_13b_pred_FL_text'] = llama2_13b_label_FL
    r_df['llama2_7b_FL_pred'] = llama2_7b_label_encoding_FL
    r_df['llama2_13b_FL_pred'] = llama2_13b_label_encoding_FL

    #Calculation of acc, recall, f1..
    prediction_df = r_df[[3, 'True_label','llama2_7b_pred','llama2_13b_pred','TimeLMs_label','hatexplain_label','llama2_7b_FL_pred','llama2_13b_FL_pred', 'datasets']]
    result = prediction_df.copy()
    result['iteration'] = iter
    results = pd.concat([results,result], axis=0)
    original_df = prediction_df.copy()
    for d in ['all', 'ADHD', 'Blind', 'Disability']:
        if d != 'all': 
            prediction_df = original_df[original_df['datasets']==d].copy()
        elif d == 'all':
            prediction_df = original_df.copy()
        print(f'dataset:{d}, shape:{prediction_df.shape}')
        true_labels = prediction_df['True_label']
        predicted_labels = prediction_df.drop([3, 'True_label', 'datasets'], axis=1)
        for i in predicted_labels.columns:
            print(f"===={i}====")

            traget_label_list = predicted_labels[i]
            accuracy = accuracy_score(true_labels, traget_label_list)
            precision = precision_score(true_labels, traget_label_list, average='binary')
            recall = recall_score(true_labels, traget_label_list, average='binary')
            f1 = f1_score(true_labels, traget_label_list, average='binary')
            score.append({'model': i, 'dataset': d, 'iteration': iter, 'acc': accuracy, 'prec': precision, 'rec': recall, 'f1-score': f1})
score = pd.DataFrame(score)
results.to_csv('all_predictions.csv')
score.to_csv('all_scores.csv')
