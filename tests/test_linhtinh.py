# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from random import randint
import os
import codecs
from data import PROJECT_PATH
from sklearn.externals import joblib
from models.train import *


def normalize_text(text):
    # return text
    dict = {
        u'òa': u'oà', u'óa': u'oá', u'ỏa': u'oả', u'õa': u'oã', u'ọa': u'oạ', u'òe': u'oè',u'óe': u'oé',
        u'ỏe': u'oẻ',u'õe': u'oẽ', u'ọe': u'oẹ', u'ùy': u'uỳ', u'úy': u'uý',u'ủy': u'uỷ', u'ũy': u'uỹ',u'ụy': u'uỵ'
    }
    for k, v in dict.iteritems():
        text = text.replace(k, v)
    return text

word_dictionary = dict()
with codecs.open(os.path.join(PROJECT_PATH, 'data', 'product.txt'), 'r', encoding='utf-8') as fin:
    for token in fin.read().split('\n'):
        token = normalize_text(token)
        word_dictionary[token.title()] = 'commune'
for key,value in word_dictionary.iteritems():
    print('"","'+key+'","11"')