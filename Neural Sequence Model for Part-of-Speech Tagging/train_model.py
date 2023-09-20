import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical
import gensim.models as gm
import nltk
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
import conllu
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report


# Read the CoNLL-U file
data_file = open("./UD_English-Atis/en_atis-ud-train.conllu", "r", encoding="utf-8")
data = conllu.parse(data_file.read())

val_file = open("./UD_English-Atis/en_atis-ud-dev.conllu", "r", encoding="utf-8")
val = conllu.parse(val_file.read())

test_file = open("./UD_English-Atis/en_atis-ud-test.conllu", "r", encoding="utf-8")
test = conllu.parse(test_file.read())

# Extract the features and labels
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

X_val = []
y_val = []
for sentence in val:
    sentence_words = []
    sentence_labels = []
    for token in sentence:
        sentence_words.append(token["form"])
        sentence_labels.append(token["upos"])
    X_val.append(sentence_words)
    y_val.append(sentence_labels)

X_test = []
y_test = []
for sentence in test:
    sentence_words = []
    sentence_labels = []
    for token in sentence:
        sentence_words.append(token["form"])
        sentence_labels.append(token["upos"])
    X_test.append(sentence_words)
    y_test.append(sentence_labels)

all_words = []
for words in X_train:
    all_words.extend(words)
freq_dist = nltk.FreqDist(all_words)
vocab = {word: count for word, count in freq_dist.items()}

for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        if X_train[i][j] in vocab and vocab[X_train[i][j]]<2:
            X_train[i][j] = 'ukn'

for i in range(len(X_val)):
    for j in range(len(X_val[i])):
        if X_val[i][j] in vocab and vocab[X_val[i][j]]<2:
            X_val[i][j] = 'ukn'

for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        if X_test[i][j] in vocab and vocab[X_test[i][j]]<2:
            X_test[i][j] = 'ukn'

word_tokenizer = Tokenizer()              
word_tokenizer.fit_on_texts(X_train)       

X_train_encoded = word_tokenizer.texts_to_sequences(X_train)  
X_val_encoded = word_tokenizer.texts_to_sequences(X_val)
X_test_encoded = word_tokenizer.texts_to_sequences(X_test)

tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(y_train)

y_train_encoded = tag_tokenizer.texts_to_sequences(y_train)
y_val_encoded = tag_tokenizer.texts_to_sequences(y_val)
y_test_encoded = tag_tokenizer.texts_to_sequences(y_test)

MAX_SEQ_LENGTH = 46
padded_X_train = pad_sequences(X_train_encoded, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
padded_y_train = pad_sequences(y_train_encoded, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
print(padded_X_train[0])
print(padded_y_train[0])
print()
padded_X_val = pad_sequences(X_val_encoded, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
padded_y_val = pad_sequences(y_val_encoded, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
print(padded_X_val[0])
print(padded_y_val[0])
print()
padded_X_test = pad_sequences(X_test_encoded, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
padded_y_test = pad_sequences(y_test_encoded, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
print(padded_X_test[0])
print(padded_y_test[0])

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(word_tokenizer.word_index) + 1, output_dim=64, input_length=MAX_SEQ_LENGTH))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
# model.add(Dense(units=32))
model.add(Dense(units=len(tag_tokenizer.word_index) + 1, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(padded_X_train, padded_y_train, validation_data=(padded_X_val, padded_y_val), epochs=10, batch_size=32)

loss, accuracy = model.evaluate(padded_X_test, padded_y_test, verbose = 1)

model.save("my_model2.h5")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Predict on validation set
pred = model.predict(padded_X_val)
y_p = []
for i in range(len(y_val_encoded)):
    no_words = len(y_val_encoded[i])
    all_words = pred[i]
    vals = []
    for j in range(no_words):
        vals.append(np.argmax(all_words[j]))
    for j in range(no_words, MAX_SEQ_LENGTH):
        vals.append(0)
    y_p.append(vals)

y_pred = np.array(y_p)
y_true = padded_y_val

y_true_flat = y_true.ravel()
y_pred_flat = y_pred.ravel()

report = classification_report(y_true_flat, y_pred_flat)
print(report)