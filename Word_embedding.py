from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast
from sklearn.feature_extraction.text import CountVectorizer


path = 'datasets/'
subreddit = 'Disability'

df = pd.read_csv(f"{path}{subreddit}_lemma.csv")
df['anonymized_body_stringfied'] = df['anonymized_body_lemmatized'].apply(ast.literal_eval)
df['anonymized_body_textualized'] = df['anonymized_body_stringfied'].apply(lambda x: ' '.join(x))

target_keyword = 'deaf'

# Word2Vec 모델 훈련
model = Word2Vec(sentences=df['anonymized_body_textualized'].apply(lambda x: x.split()), vector_size=100, window=5, min_count=1, workers=4)
model.train(df['anonymized_body_textualized'].apply(lambda x: x.split()), total_examples=model.corpus_count, epochs=10)

# target_keyword와 가장 유사한 단어 찾기
similar_words_1hop = model.wv.most_similar(target_keyword, topn=10)

# NetworkX 그래프 생성
G = nx.Graph()


# 1-hop 단어들과의 관계 추가
for word, _ in similar_words_1hop:
    G.add_edge(target_keyword, word)

    # 각 1-hop 단어에 대한 2-hop 유사 단어 찾기
    similar_words_2hop = model.wv.most_similar(word, topn=5)
    for word_2hop, _ in similar_words_2hop:
        # 2-hop 단어를 네트워크에 추가 (1-hop 단어와 연결)
        # 이미 존재하는 단어와의 관계는 중복 추가되지 않음
        G.add_edge(word, word_2hop)

# 네트워크 시각화
plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, k=0.5)  # k 값을 조정하여 노드 간 거리 조절
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=12, font_weight='bold')
plt.title(f'Word Similarity Network for "{target_keyword}" (Including up to 2-hop relationships)', size=20)
plt.show()