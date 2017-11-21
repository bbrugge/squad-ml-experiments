from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
from squad_utils import squad_parse

import numpy as np
import json
import os
import pickle

#SQUAD='/mnt/data/datasets/squad/train-v1.1.json'
SQUAD='/mnt/data/datasets/squad/dev-v1.1.json'

MAX_SEQUENCE_LENGTH = 1000

def main():

    print('Processing text dataset.')
    texts, labels, labels_index = squad_parse(SQUAD)

    while True:
        query = raw_input('> ').decode('utf-8')
        if query != '':
            predict(query, labels_index)


def predict(text, labels_index):

    tokenizer = pickle.load(open('models/tokenizer.p', 'rb'))
    sequences = tokenizer.texts_to_sequences([text])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    model = load_model('models/squad.h5')

    preds = model.predict(data)

    preds_final = {}
    for i, p in enumerate(preds[0]):
        preds_final[labels_index[i]] = format(p, '.8f')

    print 'question:'
    print ' ' + text
    print 'top answers:'

    limit = 20
    for x in sorted(preds_final.items(), key=lambda x: x[1], reverse=True):
        if limit == 0:
            break
        limit -= 1
        print ' ' + x[1] + '    ' + x[0]

if __name__ == '__main__':
    main()
