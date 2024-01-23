
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib
import lime
from lime import lime_text

model = SentenceTransformer('bert-base-nli-mean-tokens')
mlp = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(512, 512, 512), max_iter=500, learning_rate="constant", n_iter_no_change=100,random_state=0)

df = pd.read_excel('datasets/hatexplain_processed_kgs.xlsx')
df['Wikipedia_description'].fillna('', inplace=True)
df['Wikidata_description'].fillna('', inplace=True)
df['ConceptNet_terms'].fillna('', inplace=True)
df['KnowledJe_context'].fillna('', inplace=True)
df['context'] = df['Wikipedia_description'] + df['Wikidata_description'] + df['ConceptNet_terms'] + df['KnowledJe_context']

combined_bk = df['cleaned_text']+df['KnowledJe_context']
combined_bk = combined_bk.tolist()

df['class'].replace({'hate_speech':0, 'neutral':1, 'offensive':2}, inplace=True) # change for other dataset's labels
labels=df['class'].tolist()
labels=np.array(labels)
print(labels.shape)

# Sentence BERT embeddings

combined_bk_emb = model.encode(combined_bk)
#np.save('embeddings/hatexplain/sentenceBERT_KnowledJe.npy', combined_bk_emb) #If we need to reuse embeddings, turn on this line

# Get indices for each set
train_indices = np.where(df['split_set'] != 'test')[0]
test_indices = np.where(df['split_set'] == 'test')[0]

# Split document embeddings based on indices
train_embeddings = combined_bk_emb[train_indices]
test_embeddings = combined_bk_emb[test_indices]

assert len(train_embeddings) == (df.split_set.value_counts()['train']+df.split_set.value_counts()['val'])
assert len(test_embeddings) == df.split_set.value_counts()['test']

# get labels
train_labels = labels[train_indices]
test_labels = labels[test_indices]

mlp.fit(train_embeddings, train_labels)
score=mlp.score(test_embeddings, test_labels) #acc
bin_preds=mlp.predict(test_embeddings)

print(classification_report(test_labels, bin_preds, digits=4))

#Lime-based explanations

joblib.dump(mlp, 'MLP_Jessica_hatexplain.joblib')
# joblib.load('models/MLP_Jessica_hatexplain.joblib')

explainer = lime_text.LimeTextExplainer(class_names=['hate', 'non-hate', 'offensive']) # or other implicit hate class names
def pred_fn(text): # take a list of string
    text_transformed = model.encode(text) 
    return mlp.predict_proba(text_transformed)

test_texts = df[df['split_set']=='test']['cleaned_text'].tolist()
s = test_texts[25]

explanation = explainer.explain_instance(s, classifier_fn=pred_fn, top_labels =1, num_samples=5000) #first arg: string, second arg: 2d array
explanation.show_in_notebook()