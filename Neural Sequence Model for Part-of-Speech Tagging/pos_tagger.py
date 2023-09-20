import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import conllu
from tensorflow.keras.models import load_model

MAX_SEQ_LENGTH = 46

model = load_model("my_model.h5")

data_file = open("./UD_English-Atis/en_atis-ud-train.conllu", "r", encoding="utf-8")
data = conllu.parse(data_file.read())

X_train = []
y_train = []
tags = set([])
for sentence in data:
    sentence_words = []
    sentence_labels = []
    for token in sentence:
        tags.add(token["upos"])
        sentence_words.append(token["form"])
        sentence_labels.append(token["upos"])
    X_train.append(sentence_words)
    y_train.append(sentence_labels)

all_words = []
for words in X_train:
    all_words.extend(words)
freq_dist = nltk.FreqDist(all_words)
vocab = {word: count for word, count in freq_dist.items()}

for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        if X_train[i][j] in vocab and vocab[X_train[i][j]]<2:
            X_train[i][j] = 'ukn'

word_tokenizer = Tokenizer()              
word_tokenizer.fit_on_texts(X_train)      
tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(y_train)



sentence = input("Enter a sentence: ")

def printPreds(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens_st = tokens[:]
    new_sentence = ""
    for j in range(len(tokens)):
            if tokens[j] not in vocab or vocab[tokens[j]]<2:
                tokens[j] = 'ukn'
            new_sentence+=tokens[j]
            if j!=len(tokens)-1:
                new_sentence+=" "
        

    sentence_tokens = word_tokenizer.texts_to_sequences([new_sentence])

    padded_sentence = pad_sequences(sentence_tokens, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
    predicted_tags_encoded = model.predict(padded_sentence)
    length = len(sentence_tokens[0])
    predicted_tags = []
    for i in range(length):
        idx = np.argmax(predicted_tags_encoded[0][i])
        for (word, i) in tag_tokenizer.word_index.items():
            if idx == i:
                predicted_tags.append(word)
    print(predicted_tags)

    for i in range(len(predicted_tags)):
        print(tokens_st[i], '\t', predicted_tags[i])

printPreds(sentence)