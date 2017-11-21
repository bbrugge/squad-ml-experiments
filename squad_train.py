import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from squad_utils import squad_parse

import numpy as np
import json
import os
import pickle

#SQUAD='/mnt/data/datasets/squad/train-v1.1.json'
SQUAD='/mnt/data/datasets/squad/dev-v1.1.json'
GLOVE='/mnt/data/datasets/glove/glove.6B.100d.txt'

NETWORK_TYPE = 'lstm'     # lstm or conv1d
EPOCHS = 100
BATCH_SIZE = 512

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 3

def main():

    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(os.path.join(GLOVE))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    print('Processing text dataset.')

    texts, labels, labels_index = squad_parse(SQUAD)

    print('Found %s labels.' % len(labels_index))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    pickle.dump(tokenizer, open('models/tokenizer.p', 'wb'))

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

    print('Training model.')

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    if NETWORK_TYPE == 'lstm':
        x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
    if NETWORK_TYPE == 'conv1d':
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.summary(line_length=80, positions=[.33, .65, .8, 1.])

    model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['acc'])

    cb_tb = keras.callbacks.TensorBoard(log_dir='./logs',
        histogram_freq=0, write_graph=True, write_images=True)
    cb_es = keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    model.fit(x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[cb_tb, cb_es])

    model.save('models/squad.h5')

if __name__ == '__main__':
    main()
