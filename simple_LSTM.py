import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping

embed_size = 50
max_features = 20000
maxlen = 100
batch_size = 32
epochs = 2

# import datasets
train = pd.read_csv("train.csv")
print train
test = pd.read_csv("test.csv")

filled_train = train["comment_text"].fillna("unknown", inplace=True)
filled_test = test["comment_text"].fillna("unknown", inplace=True)
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values

# make a tokenizer and fit it on the training set
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train))

# convert these texts to tokenized sequences of words in order to pad them and make them the same size
tokenized_train = tokenizer.texts_to_sequences(filled_train)
tokenized_test = tokenizer.texts_to_sequences(filled_test)
padded_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
padded_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)

inp = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(LSTM(50, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

model.fit(tokenized_train, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[early])

model.predict(tokenized_test, verbose=1)
