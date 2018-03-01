# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from random import randint
import os
import codecs
from data import PROJECT_PATH
from sklearn.externals import joblib
from models.train import *
from models.conect_db import get_entities
from random import shuffle


def load_word_dictionary():
    word_dictionary = dict()
    with codecs.open(os.path.join(PROJECT_PATH, 'data', 'Viet39K.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            # token = self.normalize_text(token)
            word_dictionary[token.lower()] = '0'
    return word_dictionary

def load_word_entity():
    word_dictionary = dict()
    with codecs.open(os.path.join(PROJECT_PATH, 'data', 'commune.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            # token = self.normalize_text(token)
            word_dictionary[token.lower()] = 'commune'
            word_dictionary[token.title()] = 'commune'

    with codecs.open(os.path.join(PROJECT_PATH, 'data', 'district.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            # token = self.normalize_text(token)
            word_dictionary[token.lower()] = 'district'
            word_dictionary[token.title()] = 'district'

    with codecs.open(os.path.join(PROJECT_PATH, 'data', 'province.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            # token = self.normalize_text(token)
            word_dictionary[token.lower()] = 'province'
            word_dictionary[token.title()] = 'province'

    with codecs.open(os.path.join(PROJECT_PATH, 'data', 'product.txt'), 'r', encoding='utf-8') as fin:
        for token in fin.read().split('\n'):
            # token = self.normalize_text(token)
            word_dictionary[token.lower()] = 'product'
            word_dictionary[token.title()] = 'product'
    for ket, value in word_dictionary.items():
        print(ket,value)
    return word_dictionary


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
        if value in ['province','district','commune','product']:
            data.append([key, value])
        if value in ['province','district','product']:
            data.append([key, value])
        if value in ['province','product']:
            data.append([key, value])
    shuffle(data)
    chunks = [data[x:x + randint(10, 20)] for x in xrange(0, len(data), 100)]
    return chunks


def load_data_set_fromdb():
    datadb = get_entities()
    dictionary = load_word_dictionary()
    for token in datadb:
        if token[0].lower() in dictionary:
            del dictionary[token[0].lower()]
    import collections
    import operator
    result = collections.Counter(map(operator.itemgetter(1), datadb))
    result = sorted(result.items(), key=operator.itemgetter(1))
    data = []
    for content,entity in datadb:
        for key, value in result:
            if entity==key:
                for i in range(len(dictionary)/(value*2)):
                    data.append([content,entity])
                    data.append([content.lower(), entity])

    for key,value in dictionary.items():
        data.append([key,value])
    data = [x for pair in zip(data, data) for x in pair] #duplicate elements
    shuffle(data)
    chunks = [data[x:x + randint(10, 25)] for x in xrange(0, len(data), 100)]
    # result = collections.Counter(map(operator.itemgetter(1), data))
    # print(result)
    return chunks

def datasource():
    dataset = load_data_set_fromdb()
    # for data in dataset:
    #     for key,value in data:
    #         print(key,value)
    word_dictionary = dict()
    return dataset, word_dictionary


def test_train_postagger():
    trainer = TrainPosTagger()
    trainer.datasource = datasource
    trainer.is_overfitting = True
    trainer.train()
    with open(os.path.join(PROJECT_PATH, 'pretrained_models/NER.model'), 'w') as f:
        joblib.dump(trainer.model, f)


# data = test_train_postagger()
# for i in data:
#     print(i)
datadb = get_entities()
data2 = load_word_entity()
for data in data2:
    print(data.title())
