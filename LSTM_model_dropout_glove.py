"""
Note: this model requires the use of the Stanford GloVe word embeddings found here: https://nlp.stanford.edu/projects/glove/
"""

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers

embed_size = 50 # how big is each word vector
max_features = 25000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 150 # max number of words in a comment to use


print "importing datasets"
# import datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# fill in null values to make keras not grumble
filled_train = train["comment_text"].fillna("unknown").values
filled_test = test["comment_text"].fillna("unknown").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values

print "preparing comments for LSTM input"
# make a tokenizer and fit it on the training set
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(filled_train))

# convert these texts to tokenized sequences of words in order to pad them and make them the same size
tokenized_train = tokenizer.texts_to_sequences(filled_train) 
tokenized_test = tokenizer.texts_to_sequences(filled_test)
padded_train = pad_sequences(tokenized_train, maxlen=maxlen)
padded_test = pad_sequences(tokenized_test, maxlen=maxlen)

# using the GloVe word vector representations to get more accurate fits...
def get_glove_coefs(word, *arr):
	return word, np.asarray(arr, dtype='float32')

print "getting glove coefficients"
# get word embeddings from the glove file
embeddings_index = dict(get_glove_coefs(*o.strip().split()) for o in open("glove.6B.50d.txt"))

print "extracting glove metadata"
all_embeddings = np.stack(embeddings_index.values())
emb_mean = all_embeddings.mean(), 
emb_std = all_embeddings.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))

print "creating embedding matrix"
# set up an embedding matrix that is initialized to random values corresponding roughly with glove
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

# fill in embedding matrix with glove-defined word vectors, else keep them random to train
for word, i in word_index.items():
	if i >= max_features:
		continue
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

print "embedding matrix created with glove coefficients"

model = Sequential()
model.add(Embedding(max_features, embed_size, weights=[embedding_matrix], input_shape=(maxlen,)))
model.add(Bidirectional(LSTM(60, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
model.add(LSTM(60, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
model.add(GlobalMaxPool1D())
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print "model compiled"

model.fit(padded_train, y, batch_size=100, epochs=2, validation_split=0.08);

model.save("complex_lstm_weights.h5")

print "model fit"
