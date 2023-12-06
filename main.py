import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import re
import nltk
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

patterns = "[«»A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()


def lemmatize(review: str):
    review = re.sub(patterns, ' ', review)
    tokens = nltk.word_tokenize(review.lower())
    preprocessed_text = []
    for token in tokens:
        lemma = morph.parse(token)[0].normal_form
        if lemma not in stopwords_ru:
            preprocessed_text.append(lemma)
    return preprocessed_text

def join_lemmatized_words(words_list):
    return ' '.join(words_list)


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(7752, 100)
        self.linear2 = nn.Linear(100, 10)
        self.linear3 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# 1
df_csv = pd.read_csv(r"C:\Users\natal\lab_2_python\annotation1.csv")
texts = []
for absolute_path, rating in zip(df_csv['absolute_path'], df_csv['rating']):
    with open(absolute_path, 'r', encoding='utf-8') as file:
        text = file.read()
        texts.append((text, rating))

df = pd.DataFrame(texts, columns=['review', 'rating'])
df['review'] = df['review'].apply(lemmatize)
print(df.head())
vectorizer = CountVectorizer(stop_words=stopwords_ru)
sparse_matrix = vectorizer.fit_transform(df['review']).toarray()

# 2
print(sparse_matrix.shape)
train, test_valid = train_test_split(sparse_matrix, test_size=0.2)
test, valid = train_test_split(test_valid, test_size=0.5)

model = LogisticRegression()
criterion = nn.CrossEntropyLoss()
x_train = torch.Tensor(test).float()
# print(train.shape)
# print(valid.shape)
# print(test.shape)
