# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import unittest
import os
from random import randint
import codecs
from data import PROJECT_PATH
# import pickle
from sklearn.externals import joblib
from models.train import *
from models.tokenizer import Tokenizer
import unittest
import os
import codecs
from data import PROJECT_PATH
# import pickle
from sklearn.externals import joblib

from models.train import *
from models.tokenizer import Tokenizer

def load_word_dictionary():
    word_dictionary = dict()
    with codecs.open(os.path.join(PROJECT_PATH, 'data', 'Viet39K.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            # token = self.normalize_text(token)
            word = token.split(' ')
            # word_dictionary.append(word)
            word_dictionary[token.lower()] = '0'

    return word_dictionary

def load_word_entity():
    word_dictionary = dict()
    with codecs.open(os.path.join(PROJECT_PATH, 'data', 'commune.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            # token = self.normalize_text(token)
            # word = token.split(' ')
            # word_dictionary.append(word)
            word_dictionary[token.lower()] = 'commune'

    with codecs.open(os.path.join(PROJECT_PATH, 'data', 'district.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            # token = self.normalize_text(token)
            # word = token.split(' ')
            # word_dictionary.append(word)
            word_dictionary[token.lower()] = 'district'
    with codecs.open(os.path.join(PROJECT_PATH, 'data', 'province.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            # token = self.normalize_text(token)
            # word = token.split(' ')
            # word_dictionary.append(word)
            word_dictionary[token.lower()] = 'province'
    with codecs.open(os.path.join(PROJECT_PATH, 'data', 'product.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            # token = self.normalize_text(token)
            # word = token.split(' ')
            # word_dictionary.append(word)
            word_dictionary[token.lower()] = 'product'

    return word_dictionary

# def removekey(d, key):
#     r = dict(d)
#     del r[key]
#     return r
def load_data_set():
    entity = load_word_entity()
    dictionary = load_word_dictionary()
    print(len(dictionary))

    for token in entity:
        if token in dictionary:
            del dictionary[token]
    z = dict(dictionary.items() + entity.items())
    data = []
    for key,value in z.items():
        data.append([key,value])
        if value in ['province','district','commune','0','product']:
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
        if value in ['province','district','commune','product']:
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
        if value in ['province','district','product']:
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
        if value in ['province','product']:
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
            data.append([key, value])
    from random import shuffle
    shuffle(data)
    # print(data)
    # number_word = randint(10, 29)
    chunks = [data[x:x+randint(10, 20)] for x in xrange(0, len(data), 100)]
    return chunks

def datasource():
    dataset = load_data_set()
    word_dictionary = dict()

    return dataset, word_dictionary


def test_train_postagger():
    trainer = TrainPosTagger()
    trainer.datasource = datasource
    trainer.is_overfitting = True
    trainer.train()


    with open(os.path.join(PROJECT_PATH, 'pretrained_models/NER.model'), 'w') as f:
        joblib.dump(trainer.model, f)


test_train_postagger()